import json
import os
import sys
import time
import datetime

# Add root directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from old.prepare import preprocess_receipt

from ocr_engine import get_onnx_ocr_engine
from parser import parse_receipt_lines
from utils import load_config, get_output_folder_and_prefix

def extract_receipt_data(img_path: str, config_path: str = "walmart/config.yaml") -> str:
    """
    Executes the receipt process pipeline:
    1. Sets up dedicated output directory for the image index.
    2. Preprocessing and noise reduction.
    3. OCR text detection and bounding polygon output via ONNX RapidOCR.
    4. Spatial center calculation and Y-axis line grouping.
    5. Annotation image generation.
    6. Structural regex parsing, raw JSON generation, and transformed JSON generation.

    Args:
        img_path (str): Target image file path.
        config_path (str): Target YAML config path.

    Returns:
        str: Formatted JSON string of the transformed payload.
    """
    timing_stats = {}
    total_start_time = time.time()

    # Create target folder 'i' from image name
    output_dir, base_name = get_output_folder_and_prefix(img_path)

    # Load execution parameters
    config = load_config(config_path)
    y_threshold = config.get("y_threshold", 15)
    # Give a bit more leeway for RapidOCR due to box differences
    # y_threshold = max(y_threshold, 35)
    save_annotated_image = config.get("save_annotated_image", True)
    use_orientation = config.get("use_orientation", True)
    language = config.get("language", "english")
    
    # -------------------------------------------------------------------------
    # STEP 0: Preprocess Image
    # -------------------------------------------------------------------------
    t0_start = time.time()
    preprocessed_path = os.path.join(output_dir, f"{base_name}_preprocessed.png")
    processed_img_path = preprocess_receipt(img_path, output_path=preprocessed_path, binarize=False)
    timing_stats["Step 0: Preprocess Image"] = time.time() - t0_start
    
    # -------------------------------------------------------------------------
    # STEP 1: Execute ONNX OCR Inference
    # -------------------------------------------------------------------------
    t1_start = time.time()
    ocr_engine = get_onnx_ocr_engine(use_cls=use_orientation, language=language)
    
    # RapidOCR returns tuple: (result_list, elapse_list)
    ocr_result, _ = ocr_engine(processed_img_path)
    timing_stats["Step 1: OCR Inference"] = time.time() - t1_start
    
    # Handle empty detection output
    if not ocr_result:
        return json.dumps({"error": "No text detected", "lines": []}, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # STEP 2: Polygon Processing & Center Point Calculation
    # RapidOCR result format: [[box_points, text, confidence_score], ...]
    # -------------------------------------------------------------------------
    t2_start = time.time()
    boxes_data = []
    
    for item in ocr_result:
        box = item[0]        # List of 4 points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        text = str(item[1])  # Extracted text string
        
        center_y = (box[0][1] + box[2][1]) / 2.0
        center_x = (box[0][0] + box[1][0]) / 2.0
        
        boxes_data.append({
            'text': text.strip(),
            'center_y': float(center_y),
            'center_x': float(center_x),
            'box': box
        })
        
    timing_stats["Step 2: Polygon Processing"] = time.time() - t2_start

    # -------------------------------------------------------------------------
    # STEP 3: Y-Axis Spatial Line Reconstruction
    # -------------------------------------------------------------------------
    t3_start = time.time()
    boxes_data.sort(key=lambda x: x['center_y'])
    
    lines = []
    current_line = []
    
    for box in boxes_data:
        if not current_line:
            current_line.append(box)
        else:
            line_y = current_line[0]['center_y']
            if abs(box['center_y'] - line_y) <= y_threshold:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]
    
    if current_line:
        lines.append(current_line)
        
    formatted_lines = []
    for line_boxes in lines:
        line_boxes.sort(key=lambda x: x['center_x'])
        full_line_text = " ".join([b['text'] for b in line_boxes])
        
        elements = []
        for b in line_boxes:
            box_coords = [[float(pt[0]), float(pt[1])] for pt in b['box']]
            elements.append({
                "text": b['text'],
                "center_x": float(b['center_x']),
                "center_y": float(b['center_y']),
                "box": box_coords
            })
            
        formatted_lines.append({
            "line_text": full_line_text,
            "elements": elements
        })
    timing_stats["Step 3: Line Reconstruction"] = time.time() - t3_start

    # -------------------------------------------------------------------------
    # STEP 4: Render Bounding-Box Overlay Output Image
    # -------------------------------------------------------------------------
    t4_start = time.time()
    if save_annotated_image:
        try:
            import cv2
            import numpy as np
            img = cv2.imread(processed_img_path)
            if img is not None:
                for b_data in boxes_data:
                    box = b_data['box']
                    text = b_data['text']
                    pts = np.array(box, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                    org = (int(box[0][0]), max(int(box[0][1]) - 5, 0))
                    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                out_path = os.path.join(output_dir, f"{base_name}_annotated.png")
                cv2.imwrite(out_path, img)
                print(f"-> Saved annotated image to: {out_path}")
        except Exception as e:
            print(f"Could not save annotated image: {e}")
    timing_stats["Step 4: Render Image"] = time.time() - t4_start

    # -------------------------------------------------------------------------
    # STEP 4.5: Save Raw OCR Output (Lines & Bounding Boxes)
    # -------------------------------------------------------------------------
    t45_start = time.time()
    raw_ocr_path = os.path.join(output_dir, f"{base_name}_raw_output.txt")
    try:
        with open(raw_ocr_path, "w") as f:
            f.write(json.dumps({"raw_lines": formatted_lines}, indent=2))
        print(f"-> Saved raw unparsed OCR output to: {raw_ocr_path}")
    except Exception as e:
        print(f"Could not save raw OCR output: {e}")
    timing_stats["Step 4.5: Save Raw Output"] = time.time() - t45_start

    # -------------------------------------------------------------------------
    # STEP 5: Configuration-Driven Rule Engine & Dual JSON Generation
    # -------------------------------------------------------------------------
    t5_start = time.time()
    parsing_results = parse_receipt_lines(formatted_lines, config)
    timestamp = datetime.datetime.now().isoformat()

    # 1. Output WITHOUT transformations (Raw extraction)
    raw_parsed = parsing_results["raw_parsed"]
    raw_parsed["time_of_output"] = timestamp
    raw_parsed_json = json.dumps(raw_parsed, indent=2)
    
    raw_parsed_path = os.path.join(output_dir, f"{base_name}_raw_parsed_output.json")
    try:
        with open(raw_parsed_path, "w") as f:
            f.write(raw_parsed_json)
        print(f"-> Saved RAW parsed output to: {raw_parsed_path}")
    except Exception as e:
        print(f"Could not save raw parsed output: {e}")

    # 2. Output WITH transformations (Inferences applied)
    transformed = parsing_results["transformed"]
    transformed["time_of_output"] = timestamp
    transformed_json = json.dumps(transformed, indent=2)
    
    final_output_path = os.path.join(output_dir, f"{base_name}_final_output.json")
    try:
        with open(final_output_path, "w") as f:
            f.write(transformed_json)
        print(f"-> Saved FINAL parsed output to: {final_output_path}")
    except Exception as e:
        print(f"Could not save final output: {e}")
    timing_stats["Step 5: Rule Engine & JSON"] = time.time() - t5_start
    
    timing_stats["Total Pipeline Execution"] = time.time() - total_start_time

    print("\n--- Pipeline Timing Stats ---")
    for section, duration in timing_stats.items():
        print(f"{section}: {duration:.4f} seconds")

    return transformed_json


if __name__ == "__main__":
    receipt_image_path = "images/store-4/image-4.png"
    config_file = "configs/store-4.yaml"
    
    try:
        json_output = extract_receipt_data(receipt_image_path, config_path=config_file)
        print("\n--- FINAL TRANSFORMED OUTPUT ---")
        print(json_output)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
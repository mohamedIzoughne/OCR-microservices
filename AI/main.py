import json
import re
import yaml
import os
import sys
import time
import datetime
from typing import Dict, List, Any, Optional
from rapidocr_onnxruntime import RapidOCR

# Add parent directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from old.prepare import preprocess_receipt


# -------------------------------------------------------------------------
# RAPIDOCR CONFIGURATION PATCH FOR RECOGNIZER KEYS
# -------------------------------------------------------------------------
from rapidocr_onnxruntime.utils import UpdateParameters
original_call = UpdateParameters.__call__

def patched_call(self, config, **kwargs):
    new_config = original_call(self, config, **kwargs)
    if 'rec_keys_path' in kwargs:
        new_config['Rec']['keys_path'] = kwargs['rec_keys_path']
    return new_config

UpdateParameters.__call__ = patched_call

import urllib.request
def ensure_latin_model():
    model_dir = os.path.join(os.path.dirname(__file__), "models", "latin")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "latin_PP-OCRv3_rec_infer.onnx")
    keys_path = os.path.join(model_dir, "latin_dict.txt")
    
    if not os.path.exists(model_path):
        print(f"Downloading Latin ONNX OCR Model to {model_path}...")
        urllib.request.urlretrieve("https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/latin/rec.onnx", model_path)
    if not os.path.exists(keys_path):
        print(f"Downloading Latin OCR Dictionary to {keys_path}...")
        urllib.request.urlretrieve("https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/latin/dict.txt", keys_path)
    
    return model_path, keys_path

# -------------------------------------------------------------------------
# ONNX OCR ENGINE GLOBAL CACHE
# Keeps initialized engines in memory across function calls
# -------------------------------------------------------------------------
ONNX_OCR_CACHE: Dict[str, RapidOCR] = {}


def get_onnx_ocr_engine(use_cls: bool = True, language: str = "english") -> RapidOCR:
    """
    Retrieves a cached RapidOCR engine or creates a new one if not loaded yet.
    The ONNX Runtime backend automatically utilizes CPU acceleration (Intel MKL/OpenVINO)
    or GPU acceleration (NVIDIA CUDA/TensorRT) depending on hardware availability.
    """
    cache_key = f"onnx_cls_{use_cls}_{language}"
    
    if cache_key not in ONNX_OCR_CACHE:
        print(f"--> [ONNX Cache Miss] Initializing RapidOCR ONNX Engine ({cache_key})...")
        
        kwargs = {
            "use_cls": use_cls,
            "use_det": True,
            "use_rec": True
        }
        
        if language.lower() in ["english", "french", "latin", "en", "fr"]:
            model_path, keys_path = ensure_latin_model()
            kwargs["rec_model_path"] = model_path
            kwargs["rec_keys_path"] = keys_path
            
        ONNX_OCR_CACHE[cache_key] = RapidOCR(**kwargs)
    return ONNX_OCR_CACHE[cache_key]


def load_config(config_path: str = "walmart/config.yaml") -> Dict[str, Any]:
    """
    Loads YAML configuration settings for receipt spatial thresholds and regex rules.

    Args:
        config_path (str): File system path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration key-value dictionary, or empty dict on failure.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            return config if config else {}
    except Exception as e:
        print(f"Could not load config from {config_path}: {e}")
        return {}


def parse_receipt_lines(formatted_lines: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses spatial text lines into structured metadata and itemized product records.
    Returns both raw extracted results and transformed/arithmetically-inferred results.

    Args:
        formatted_lines (List[Dict[str, Any]]): Array of horizontally reconstructed line objects.
        config (Dict[str, Any]): Parsing parameters loaded from YAML configuration.

    Returns:
        Dict[str, Any]: Object containing both 'raw_parsed' and 'transformed' payloads.
    """
    raw_metadata: Dict[str, Any] = {}
    raw_items: List[Dict[str, Any]] = []
    unmapped_lines: List[str] = []
    
    fields_config = config.get("fields", {})
    items_section_config = config.get("items_section", {})
    item_patterns = config.get("item_patterns", [])
    
    start_trigger = items_section_config.get("start_trigger")
    end_trigger = items_section_config.get("end_trigger")
    
    currency = config.get("currency")
    
    in_items_section = False if start_trigger else True
    raw_lines = [line["line_text"] for line in formatted_lines]
    
    # -------------------------------------------------------------------------
    # STAGE 1: Pure Key-Value Field Metadata Extraction (Raw)
    # -------------------------------------------------------------------------
    for field_name, rules in fields_config.items():
        regex_pattern = rules.get("regex")
        group_id = rules.get("group", 1)
        split_trigger = rules.get("split_trigger")
        split_regex = rules.get("split_regex")
        
        for idx, line_text in enumerate(raw_lines):
            # 1. Match on same line
            if regex_pattern:
                match = re.search(regex_pattern, line_text)
                if match:
                    try:
                        val = match.group(group_id)
                        if val:
                            if currency and currency in val:
                                val = val.replace(currency, "").strip()
                            
                            val = val.strip()
                            
                            # Clean up OCR artifacts in specific fields
                            if field_name in ["unit_price", "total_price"]:
                                val = re.sub(r',\s+', ',', val)
                            elif field_name == "description":
                                val = val.replace("_", " ")
                                
                            raw_metadata[field_name] = val
                            break
                    except IndexError:
                        pass
            
            # 2. Match split across next or previous line
            if split_trigger and split_regex:
                if split_trigger in line_text:
                    val_match = None
                    split_dir = rules.get("split_direction", "down")
                    
                    if split_dir == "down" and idx + 1 < len(raw_lines):
                        next_line = raw_lines[idx + 1]
                        val_match = re.search(split_regex, next_line)
                    elif split_dir == "up" and idx - 1 >= 0:
                        prev_line = raw_lines[idx - 1]
                        val_match = re.search(split_regex, prev_line)
                    elif split_dir == "both":
                        if idx + 1 < len(raw_lines):
                            val_match = re.search(split_regex, raw_lines[idx + 1])
                        if not val_match and idx - 1 >= 0:
                            val_match = re.search(split_regex, raw_lines[idx - 1])
                        
                    if val_match:
                        try:
                            val = val_match.group(group_id)
                            if val:
                                if currency:
                                    val = val.replace(currency, '')
                                raw_metadata[field_name] = val.strip()
                                break
                        except IndexError:
                            pass

    # -------------------------------------------------------------------------
    # STAGE 2: Itemized Line Extraction & Multi-line Merging (Raw)
    # -------------------------------------------------------------------------
    pending_description = None

    for line_info in formatted_lines:
        line_text = line_info["line_text"].strip()
        
        # Check start section trigger
        if start_trigger and not in_items_section:
            if re.search(start_trigger, line_text):
                in_items_section = True
                unmapped_lines.append(line_text)
                continue
                
        # Check end section trigger
        if end_trigger and in_items_section:
            if re.search(end_trigger, line_text):
                in_items_section = False
                if pending_description:
                    raw_items.append({
                        "quantity": None,
                        "unit_price": None,
                        "total_price": None,
                        "description": pending_description
                    })
                    pending_description = None
                
        if in_items_section:
            matched_item = False
            
            for pattern_entry in item_patterns:
                pattern = pattern_entry.get("regex")
                if not pattern:
                    continue
                
                match = re.match(pattern, line_text)
                if match:
                    item_dict = match.groupdict()
                    
                    # Case A: Line contains price/quantity breakdown
                    if item_dict.get("unit_price") is not None:
                        if currency:
                            item_dict["unit_price"] = item_dict["unit_price"].replace(currency, '').strip()
                        if item_dict.get("total_price") and currency:
                            item_dict["total_price"] = item_dict["total_price"].replace(currency, '').strip()
                        
                        # Clean OCR spaces in prices
                        item_dict["unit_price"] = re.sub(r',\s+', ',', item_dict["unit_price"])
                        if item_dict.get("total_price"):
                            item_dict["total_price"] = re.sub(r',\s+', ',', item_dict["total_price"])

                            
                        if pending_description:
                            item_dict["description"] = pending_description
                            pending_description = None
                        
                        # Infer total price if missing from the OCR
                        if not item_dict.get("total_price") and item_dict.get("quantity") and item_dict.get("unit_price"):
                            try:
                                qty = int(item_dict["quantity"])
                                u_price = float(item_dict["unit_price"].replace(',', '.'))
                                item_dict["total_price"] = f"{(qty * u_price):.2f}".replace('.', ',')
                            except ValueError:
                                pass
                                
                        raw_items.append(item_dict)
                        matched_item = True
                        break
                    
                    # Case B: Line contains description only
                    elif item_dict.get("description") is not None and item_dict.get("unit_price") is None:
                        # Clean OCR underscores in descriptions
                        clean_desc = item_dict["description"].replace("_", " ")
                        if pending_description:
                            pending_description += f" {clean_desc}"
                        else:
                            pending_description = clean_desc
                        matched_item = True
                        break
            
            if not matched_item:
                unmapped_lines.append(line_text)
        else:
            unmapped_lines.append(line_text)

    if pending_description:
        raw_items.append({
            "quantity": None,
            "unit_price": None,
            "total_price": None,
            "description": pending_description
        })
        pending_description = None

    raw_output_data = {
        "items": json.loads(json.dumps(raw_items)),
        "subtotal": raw_metadata.get("subtotal"),
        "tax": raw_metadata.get("tax"),
        "total": raw_metadata.get("total")
    }
    if currency:
        raw_output_data["currency"] = currency

    # -------------------------------------------------------------------------
    # STAGE 3: Multi-Directional Arithmetic Integrity Engine (Transformed)
    # -------------------------------------------------------------------------
    transformed_metadata = dict(raw_metadata)
    transformed_items = json.loads(json.dumps(raw_items))

    def parse_float(val: Optional[str]) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(str(val).replace(',', '.').replace('$', '').strip())
        except ValueError:
            return None

    subtotal = parse_float(transformed_metadata.get("subtotal"))
    tax = parse_float(transformed_metadata.get("tax"))
    total = parse_float(transformed_metadata.get("total"))

    # Extract item prices
    item_prices = [parse_float(i.get("total_price")) for i in transformed_items]
    has_unpriced_items = any(p is None for p in item_prices)
    valid_item_prices = [p for p in item_prices if p is not None]

    # Rule 1: Deduce missing single item price if Subtotal is known
    missing_items = [i for i in transformed_items if parse_float(i.get("total_price")) is None]
    if len(missing_items) == 1 and subtotal is not None and subtotal > sum(valid_item_prices):
        inferred_item_price = round(subtotal - sum(valid_item_prices), 2)
        missing_items[0]["total_price"] = f"{inferred_item_price:.2f}"
        missing_items[0]["inferred"] = True
        item_prices = [parse_float(i.get("total_price")) for i in transformed_items]
        has_unpriced_items = any(p is None for p in item_prices)
        valid_item_prices = [p for p in item_prices if p is not None]

    # Rule 2: Calculate missing Subtotal from sum of items (if ALL items have valid prices)
    if subtotal is None and not has_unpriced_items and len(valid_item_prices) > 0:
        subtotal = round(sum(valid_item_prices), 2)
        transformed_metadata["subtotal"] = f"{subtotal:.2f}"
        transformed_metadata["subtotal_inferred"] = True

    # Rule 3: Infer missing Subtotal (Subtotal = Total - Tax)
    if subtotal is None and total is not None and tax is not None:
        subtotal = round(total - tax, 2)
        transformed_metadata["subtotal"] = f"{subtotal:.2f}"
        transformed_metadata["subtotal_inferred"] = True

    # Rule 4: Infer missing Total (Total = Subtotal + Tax)
    if total is None and subtotal is not None and tax is not None:
        total = round(subtotal + tax, 2)
        transformed_metadata["total"] = f"{total:.2f}"
        transformed_metadata["total_inferred"] = True

    # Rule 5: Infer missing Tax (Tax = Total - Subtotal)
    if tax is None and total is not None and subtotal is not None:
        tax = round(total - subtotal, 2)
        transformed_metadata["tax"] = f"{tax:.2f}"
        transformed_metadata["tax_inferred"] = True

    transformed_output_data = {
        "items": transformed_items,
        "subtotal": transformed_metadata.get("subtotal"),
        "tax": transformed_metadata.get("tax"),
        "total": transformed_metadata.get("total")
    }
    if currency:
        transformed_output_data["currency"] = currency

    return {
        "raw_parsed": raw_output_data,
        "transformed": transformed_output_data
    }


def get_output_folder_and_prefix(img_path: str) -> tuple:
    """
    Extracts index 'i' from image path like 'image-2.png' or 'image_2.jpg'
    and creates an output directory named after the index 'i' (e.g. 'output/2/').
    Returns (output_dir, file_prefix).
    """
    filename = os.path.basename(img_path)
    base_name, _ = os.path.splitext(filename)
    
    # Extract 'i' from pattern image-i or image_i
    match = re.search(r'image[-_](\d+)', base_name, re.IGNORECASE)
    if match:
        index_folder = match.group(1)
    else:
        index_folder = base_name

    # Root output directory where index subfolders are placed
    base_dir = os.path.dirname(img_path)
    output_dir = os.path.join(base_dir, base_name)
    os.makedirs(output_dir, exist_ok=True)

    return output_dir, index_folder


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
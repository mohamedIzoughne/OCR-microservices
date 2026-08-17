import json
import os
import sys
import time
import datetime
from confluent_kafka import Consumer, Producer, KafkaError, KafkaException

# Add root directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from old.prepare import preprocess_receipt

from ocr_engine import get_onnx_ocr_engine
from parser import parse_receipt_lines
from utils import load_config, get_output_folder_and_prefix

def extract_receipt_data(img_path: str, config_path: str = "walmart/config.yaml") -> dict:
    """
    Executes the receipt process pipeline:
    1. Sets up dedicated output directory for the image index.
    2. Preprocessing and noise reduction.
    3. OCR text detection and bounding polygon output via ONNX RapidOCR.
    4. Spatial center calculation and Y-axis line grouping.
    5. Structural regex parsing and dictionary generation.

    Args:
        img_path (str): Target image file path.
        config_path (str): Target YAML config path.

    Returns:
        dict: The raw_parsed dictionary.
    """
    timing_stats = {}
    total_start_time = time.time()

    # Create target folder 'i' from image name
    output_dir, base_name = get_output_folder_and_prefix(img_path)

    # Load execution parameters
    config = load_config(config_path)
    y_threshold = config.get("y_threshold", 15)
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
        return {"error": "No text detected", "lines": []}

    # -------------------------------------------------------------------------
    # STEP 2: Polygon Processing & Center Point Calculation
    # -------------------------------------------------------------------------
    t2_start = time.time()
    boxes_data = []
    
    for item in ocr_result:
        box = item[0]
        text = str(item[1])
        
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
    # STEP 5: Configuration-Driven Rule Engine
    # -------------------------------------------------------------------------
    t5_start = time.time()
    parsing_results = parse_receipt_lines(formatted_lines, config)
    timestamp = datetime.datetime.now().isoformat()

    # 1. Output WITHOUT transformations (Raw extraction)
    raw_parsed = parsing_results["raw_parsed"]
    raw_parsed["time_of_output"] = timestamp
    
    # Transformed output commented out per user request
    # transformed = parsing_results["transformed"]
    # transformed["time_of_output"] = timestamp
    # final_output = transformed

    timing_stats["Step 5: Rule Engine"] = time.time() - t5_start
    timing_stats["Total Pipeline Execution"] = time.time() - total_start_time

    print("\n--- Pipeline Timing Stats ---")
    for section, duration in timing_stats.items():
        print(f"{section}: {duration:.4f} seconds")

    return raw_parsed


if __name__ == "__main__":
    KAFKA_BROKER = 'localhost:9092'
    INPUT_TOPIC = 'ocr_image_requests'
    OUTPUT_TOPIC = 'ocr_parsed_results'
    
    consumer_config = {
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': 'ocr_service_group',
        'auto.offset.reset': 'earliest'
    }
    producer_config = {
        'bootstrap.servers': KAFKA_BROKER
    }
    
    print(f"Initializing Kafka Consumer for topic: {INPUT_TOPIC}")
    try:
        consumer = Consumer(consumer_config)
        consumer.subscribe([INPUT_TOPIC])
        
        producer = Producer(producer_config)
    except Exception as e:
        print(f"Failed to initialize Kafka: {e}")
        sys.exit(1)
        
    print("Starting Kafka consumer loop...")
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"Consumer error: {msg.error()}")
                    break
            
            try:
                # Expecting payload: {"image_path": "path/to/image.png"}
                payload = json.loads(msg.value().decode('utf-8'))
                image_path = payload.get("image_path")
                
                if image_path and os.path.exists(image_path):
                    print(f"\nProcessing received image: {image_path}")
                    config_file = payload.get("config_path", "configs/store-4.yaml")
                    
                    result_dict = extract_receipt_data(image_path, config_path=config_file)
                    
                    # Produce result
                    producer.produce(
                        OUTPUT_TOPIC, 
                        key=image_path.encode('utf-8'),
                        value=json.dumps(result_dict).encode('utf-8')
                    )
                    producer.poll(0)
                    print(f"Successfully processed and sent result to {OUTPUT_TOPIC}.")
                else:
                    print(f"Invalid or missing image_path in message: {payload}")
                    
            except Exception as ex:
                print(f"Error processing message: {ex}")
                
    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down...")
    finally:
        consumer.close()
        producer.flush()
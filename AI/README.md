# OCR AI Service

This project contains the AI pipeline for extracting receipt data using OCR (Optical Character Recognition) and structural regex parsing. The pipeline is split into multiple modules for easier maintenance and structure.

## Prerequisites

- Python 3.8+ recommended
- Virtual Environment (recommended)

## Setup Guide

1. **Create and Activate a Virtual Environment:**
   Depending on your OS, you can create and activate a python virtual environment:

   ```bash
   # Create
   python -m venv .venv
   
   # Activate (Linux/Mac)
   source .venv/bin/activate
   
   # Activate (Windows)
   .venv\Scripts\activate
   ```

2. **Install Dependencies:**
   Ensure you are in the directory containing `requirements.txt`. Install the required libraries using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

The entrypoint to test the OCR pipeline is `main.py`.

1. Navigate to the `AI` directory.
2. Run the main script:
   ```bash
   python main.py
   ```

### What it does

When you run `main.py`, it executes the `extract_receipt_data()` function which:
- Takes a sample image (e.g. `images/store-4/image-4.png`) and configuration (`configs/store-4.yaml`).
- Pre-processes the image for better OCR results.
- Runs ONNX RapidOCR to detect text and their bounding boxes.
- Calculates spatial centers and parses out the fields (metadata and items) using regex rules defined in the config.
- Outputs annotated images and structured JSON files in the same directory as the input image.

### Project Structure

- `main.py`: The pipeline orchestrator and entrypoint.
- `ocr_engine.py`: Handles caching and patching for the ONNX OCR engine.
- `parser.py`: Contains the engine for parsing text using YAML configs and arithmetic transformations.
- `utils.py`: Helper functions for file loading and path manipulation.

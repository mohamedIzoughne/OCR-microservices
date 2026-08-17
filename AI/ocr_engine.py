import os
import urllib.request
from typing import Dict
from rapidocr_onnxruntime import RapidOCR

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

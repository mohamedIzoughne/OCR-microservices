import os
import re
import yaml
from typing import Dict, Any

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

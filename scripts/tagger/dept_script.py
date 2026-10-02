import json
import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional

SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

def load_depth_manifest(manifest_path: str) -> Dict:
    """Loads existing annotations manifest or initializes a default structure."""
    path = Path(manifest_path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {"depth_unit": "meters", "annotations": {}}

def save_depth_manifest(manifest_path: str, annotations: Dict) -> None:
    """Saves depth annotations to JSON file."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "depth_unit": "meters",
        "annotations": annotations
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def scan_depth_dataset(source_dir: str) -> List[Path]:
    """Returns sorted list of RGB image paths from source directory."""
    raw_path = Path(source_dir)
    if not raw_path.exists():
        return []
    return sorted([f for f in raw_path.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXTS])

def export_depth_map_npy(
    output_dir: str, 
    image_name: str, 
    image_size: Tuple[int, int], 
    depth_points: List[Dict]
) -> str:
    """
    Generates a 32-bit float NumPy array representing depth and saves as .npy file.
    Interpolates depth map based on marked anchor points (x, y, depth_in_meters).
    """
    width, height = image_size
    depth_map = np.full((height, width), fill_value=np.nan, dtype=np.float32)

    if depth_points:
        # Fill specific annotated point locations
        for pt in depth_points:
            x, y, d = int(pt['x']), int(pt['y']), float(pt['depth'])
            if 0 <= x < width and 0 <= y < height:
                depth_map[y, x] = d

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    stem = Path(image_name).stem
    npy_file = out_path / f"{stem}_depth.npy"
    np.save(npy_file, depth_map)
    return str(npy_file)
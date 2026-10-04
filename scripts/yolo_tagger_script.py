import json
import os
import cv2
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO

# ==========================================
# 1. CONSTANTS & DEFAULT MANIFESTS
# ==========================================
SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# Default keypoint definitions (COCO 17-point)
DEFAULT_POSE_KPTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
DEFAULT_POSE_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), 
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), 
    (13, 15), (12, 14), (14, 16)
]

def get_default_manifest(task_type: str) -> Dict:
    """Returns the default JSON structure for a specific YOLO task."""
    if task_type == "pose":
        return {
            "classes": ["person"],
            "keypoint_names": DEFAULT_POSE_KPTS,
            "skeleton": DEFAULT_POSE_SKELETON,
            "annotations": {}
        }
    elif task_type == "obb":
        return {"classes": ["vehicle", "ship", "building"], "annotations": {}}
    elif task_type == "segm":
        return {"classes": ["object", "background"], "annotations": {}}
    elif task_type == "cls":
        return {"classes": ["class_a", "class_b"], "annotations": {}}
    elif task_type == "dept":
        return {"depth_unit": "meters", "annotations": {}}
    else: # bbox
        return {"classes": ["object"], "annotations": {}}

# ==========================================
# 2. UNIFIED I/O & SCANNING
# ==========================================
def scan_dataset_images(source_dir: str) -> List[Path]:
    """Returns sorted list of image file paths from source directory."""
    raw_path = Path(source_dir)
    if not raw_path.exists():
        return []
    return sorted([f for f in raw_path.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXTS])

def load_task_manifest(manifest_path: str, task_type: str) -> Dict:
    """Loads existing manifest or initializes default structure based on task."""
    path = Path(manifest_path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return get_default_manifest(task_type)

def save_task_manifest(manifest_path: str, data: Dict) -> None:
    """Saves annotations and configuration to JSON file."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# ==========================================
# 3. GEOMETRY & CONVERSION HELPERS
# ==========================================
def bbox_to_polygon(x1, y1, x2, y2) -> List[Tuple[float, float]]:
    """Converts a bounding box to a 4-point polygon (for segm fallback)."""
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

def get_obb_corners(cx, cy, w, h, angle_deg) -> List[Tuple[float, float]]
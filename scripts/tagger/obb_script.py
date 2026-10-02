import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

def load_obb_manifest(manifest_path: str) -> Dict:
    """Loads existing annotations manifest or initializes a default structure."""
    path = Path(manifest_path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {"classes": ["vehicle", "ship", "building"], "annotations": {}}

def save_obb_manifest(manifest_path: str, classes: List[str], annotations: Dict) -> None:
    """Saves OBB annotations and active classes to JSON file."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "classes": sorted(list(set(classes))),
        "annotations": annotations
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def scan_obb_dataset(source_dir: str) -> List[Path]:
    """Returns sorted list of image paths from source directory."""
    raw_path = Path(source_dir)
    if not raw_path.exists():
        return []
    return sorted([f for f in raw_path.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXTS])

def get_obb_corners(cx: float, cy: float, w: float, h: float, angle_deg: float) -> List[Tuple[float, float]]:
    """Calculates 4 corner coordinates (x, y) for a rotated box given center, size, and angle."""
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    # Half dimensions
    hw = w / 2.0
    hh = h / 2.0

    # Corners relative to center (unrotated)
    local_corners = [
        (-hw, -hh),
        (hw, -hh),
        (hw, hh),
        (-hw, hh)
    ]

    # Rotate corners around center (cx, cy)
    corners = []
    for lx, ly in local_corners:
        rx = cx + (lx * cos_a - ly * sin_a)
        ry = cy + (lx * sin_a + ly * cos_a)
        corners.append((rx, ry))

    return corners
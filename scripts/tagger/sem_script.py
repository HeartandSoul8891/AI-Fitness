import json
from pathlib import Path
from typing import Dict, List, Tuple

SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

def load_sem_manifest(manifest_path: str) -> Dict:
    """Loads existing segmentation manifest or creates a default structure."""
    path = Path(manifest_path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {"classes": ["object", "background"], "annotations": {}}

def save_sem_manifest(manifest_path: str, classes: List[str], annotations: Dict) -> None:
    """Saves active classes and segmentation polygon annotations to JSON."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "classes": sorted(list(set(classes))),
        "annotations": annotations
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def scan_sem_dataset(source_dir: str) -> List[Path]:
    """Returns sorted list of image paths from source directory."""
    raw_path = Path(source_dir)
    if not raw_path.exists():
        return []
    return sorted([f for f in raw_path.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXTS])

def normalize_polygon(points: List[Dict[str, int]], img_w: int, img_h: int) -> List[Tuple[float, float]]:
    """Converts pixel coordinate vertices to normalized [0, 1] range for YOLO format."""
    return [(round(pt["x"] / img_w, 6), round(pt["y"] / img_h, 6)) for pt in points]
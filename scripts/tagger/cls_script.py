import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

def load_manifest(manifest_path: str) -> Dict:
    """Loads existing tags manifest or creates a default structure."""
    path = Path(manifest_path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {"classes": ["class_a", "class_b"], "annotations": {}}

def save_manifest(manifest_path: str, classes: List[str], annotations: Dict[str, str]) -> None:
    """Saves active classes and filename-to-class annotations to JSON file."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "classes": sorted(list(set(classes))),
        "annotations": annotations
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def scan_images(source_dir: str) -> List[Path]:
    """Returns sorted list of image file paths from source directory."""
    raw_path = Path(source_dir)
    if not raw_path.exists():
        return []
    return sorted([f for f in raw_path.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS])
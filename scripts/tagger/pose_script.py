import json
from pathlib import Path
from typing import Dict, List, Tuple

SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# Default COCO 17-Keypoint definition
DEFAULT_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Default Skeleton Links (Pairs of keypoint indices to draw connection lines)
DEFAULT_SKELETON_LINKS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Upper body / Arms
    (5, 11), (6, 12), (11, 12),               # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)    # Legs
]

def load_pose_manifest(manifest_path: str) -> Dict:
    """Loads existing pose manifest or initializes default structure."""
    path = Path(manifest_path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {
        "classes": ["person"],
        "keypoint_names": DEFAULT_KEYPOINT_NAMES,
        "skeleton": DEFAULT_SKELETON_LINKS,
        "annotations": {}
    }

def save_pose_manifest(
    manifest_path: str,
    classes: List[str],
    keypoint_names: List[str],
    skeleton: List[Tuple[int, int]],
    annotations: Dict
) -> None:
    """Saves pose annotations and skeleton configuration to JSON."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "classes": sorted(list(set(classes))),
        "keypoint_names": keypoint_names,
        "skeleton": skeleton,
        "annotations": annotations
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def scan_pose_dataset(source_dir: str) -> List[Path]:
    """Returns sorted list of image file paths from source directory."""
    raw_path = Path(source_dir)
    if not raw_path.exists():
        return []
    return sorted([f for f in raw_path.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXTS])
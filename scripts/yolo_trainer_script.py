
import json
import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

# ==========================================
# PATH RESOLUTION
# ==========================================
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent if FILE_PATH.parent.name == "scripts" else FILE_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ==========================================
# DEFAULT MODELS & SETTINGS
# ==========================================
TASK_BASE_MODELS = {
    "bbox": "yolo26n.pt",
    "segm": "yolo26n-seg.pt",
    "obb": "yolo26n-obb.pt",
    "pose": "yolo26n-pose.pt",
    "cls": "yolo26n-cls.pt",
    "dept": "yolo26n.pt" 
}

def load_settings() -> dict:
    """Loads the unified settings.json."""
    settings_file = PROJECT_ROOT / "user" / "settings.json"
    if settings_file.exists():
        try:
            with open(settings_file, "r", encoding="utf-8") as f:
                raw_settings = json.load(f)
                return {str(k).strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw_settings.items()}
        except Exception:
            pass
    return {}

def resolve_model_save_dir() -> Path:
    """Determines where training outputs and weights are saved."""
    settings = load_settings()
    # Use the unified training_folder or ultralytics_models_folder
    target_dir = settings.get("training_folder") or settings.get("ultralytics_models_folder")
    
    if not target_dir or not os.path.exists(target_dir):
        target_dir = PROJECT_ROOT / "training"
        os.makedirs(target_dir, exist_ok=True)
        
    return Path(target_dir)

def get_device_config(device_selection: str = "auto"):
    """Resolves compute device."""
    if device_selection == "cpu":
        return "cpu"
    if device_selection in ["gpu", "auto"]:
        if torch.cuda.is_available():
            return 0
        # Add MPS/ROCm checks here if needed in the future
    return "cpu"

# ==========================================
# CORE TRAINING ENGINE
# ==========================================
def start_yolo_training(
    task_type: str,
    dataset_path: str,
    model_weights: str = None,
    project_name: str = "yolo_experiments",
    run_name: str = "train_run",
    device: str = "auto",
    **kwargs # Captures ALL hyperparameters and augmentations from the GUI
):
    """Executes model training for ANY YOLO task type with full parameter support."""
    save_dir = resolve_model_save_dir()
    target_device = get_device_config(device)

    # 1. Resolve Model Weights
    if not model_weights or not os.path.exists(model_weights):
        model_weights = TASK_BASE_MODELS.get(task_type, "yolo11n.pt")

    # 2. Resolve Dataset Path
    if task_type != "cls":
        data_yaml = Path(dataset_path) / "data.yaml" if os.path.isdir(dataset_path) else Path(dataset_path)
        if not data_yaml.exists():
            return False, f"Missing data.yaml file at `{data_yaml}`", None
        # Ultralytics requires forward slashes for paths
        dataset_target = str(data_yaml.resolve()).replace("\\", "/")
    else:
        dataset_target = str(Path(dataset_path).resolve()).replace("\\", "/")

    # 3. Build Training Arguments
    train_kwargs = {
        "data": dataset_target,
        "device": target_device,
        "project": str(save_dir / project_name),
        "name": run_name,
        "exist_ok": True,
        "save": True,
        "plots": True
    }

    # Merge UI kwargs (epochs, batch, augmentations, etc.)
    # Filter out None values to let Ultralytics use its internal defaults
    for k, v in kwargs.items():
        if v is not None:
            train_kwargs[k] = v

    try:
        model = YOLO(model_weights)
        results = model.train(**train_kwargs)
        output_run_path = save_dir / project_name / run_name
        return True, f"Training complete for {task_type.upper()} task. Artifacts saved to `{output_run_path}`", str(output_run_path)
    except Exception as e:
        return False, f"Training failed with error: {str(e)}", None
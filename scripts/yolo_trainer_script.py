import json
import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

# Resolve Project Root dynamically
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent.parent if FILE_PATH.parent.name == "scripts" else FILE_PATH.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TASK_BASE_MODELS = {
    "bbox": "yolo11n.pt",
    "segm": "yolo11n-seg.pt",
    "obb": "yolo11n-obb.pt",
    "pose": "yolo11n-pose.pt",
    "cls": "yolo11n-cls.pt",
    "dept": "yolo11n.pt"
}

def load_settings():
    settings_file = PROJECT_ROOT / "user" / "settings.json"
    if settings_file.exists():
        try:
            with open(settings_file, "r", encoding="utf-8") as f:
                raw_settings = json.load(f)
            return {str(k).strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw_settings.items()}
        except Exception:
            pass
    return {}

def resolve_model_save_dir(task_type):
    settings = load_settings()
    folder_key = f"ultralytics_{task_type}_folder"
    configured_dir = settings.get(folder_key)
    if configured_dir and os.path.exists(configured_dir):
        target_dir = Path(configured_dir)
    else:
        target_dir = PROJECT_ROOT / "output" / "models" / task_type
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir

def get_device_config(device_selection="auto"):
    if device_selection == "cpu":
        return "cpu"
    if device_selection == "gpu" or device_selection == "auto":
        if torch.cuda.is_available():
            return 0
        # Add ROCm/MPS checks here if needed
    return "cpu"

def start_yolo_training(
    task_type, 
    dataset_path, 
    model_weights=None, 
    project_name="yolo_experiments", 
    run_name="train_run",
    device="auto", 
    **kwargs # Captures ALL hyperparameters, augmentations, and advanced settings
):
    """Executes model training for ANY YOLO task type with full parameter support."""
    save_dir = resolve_model_save_dir(task_type)
    target_device = get_device_config(device)
    
    # 1. Resolve Model Weights
    if not model_weights or not os.path.exists(model_weights):
        model_weights = TASK_BASE_MODELS.get(task_type, "yolo11n.pt")
        
    # 2. Resolve Dataset Path
    if task_type != "cls":
        data_yaml = Path(dataset_path) / "data.yaml" if os.path.isdir(dataset_path) else Path(dataset_path)
        if not data_yaml.exists():
            return False, f"Missing data.yaml file at `{data_yaml}`", None
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
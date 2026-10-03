import json
import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent.parent
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
                return json.load(f)
        except Exception:
            pass
    return {}

def resolve_model_save_dir(task_type):
    settings = load_settings()
    folder_key = f"ultralytics_{task_type}_folder"
    configured_dir = settings.get(folder_key)
    
    if configured_dir:
        target_dir = Path(configured_dir)
    else:
        target_dir = PROJECT_ROOT / "output" / "models" / task_type
        
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir

def get_device_config(device_selection="auto"):
    if device_selection == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return 0
    return "cpu"

def start_yolo_training(
    task_type,
    dataset_path,
    model_weights=None,
    epochs=50,
    batch_size=16,
    imgsz=640,
    device="auto",
    learning_rate=0.01,
    patience=50,
    optimizer="auto",
    workers=4,
    seed=0,
    cos_lr=False,
    augment=True,
    val_evaluation=True,
    project_name="yolo_experiments",
    run_name="train_run",
    exist_ok=True,
    extra_args=None
):
    """Executes model training with fully configurable parameters across all YOLO task types."""
    save_dir = resolve_model_save_dir(task_type)
    target_device = get_device_config(device)

    if not model_weights or not os.path.exists(model_weights):
        model_weights = TASK_BASE_MODELS.get(task_type, "yolo11n.pt")

    if task_type != "cls":
        data_yaml = Path(dataset_path) / "data.yaml" if os.path.isdir(dataset_path) else Path(dataset_path)
        if not data_yaml.exists():
            return False, f"Missing data.yaml file at `{data_yaml}`", None
        dataset_target = str(data_yaml.resolve()).replace("\\", "/")
    else:
        dataset_target = str(Path(dataset_path).resolve()).replace("\\", "/")

    # Build kwargs dictionary passed directly to Ultralytics YOLO.train()
    train_kwargs = {
        "data": dataset_target,
        "epochs": int(epochs),
        "batch": int(batch_size),
        "imgsz": int(imgsz),
        "device": target_device,
        "lr0": float(learning_rate),
        "patience": int(patience),
        "optimizer": str(optimizer),
        "workers": int(workers),
        "seed": int(seed),
        "cos_lr": bool(cos_lr),
        "augment": bool(augment),
        "val": bool(val_evaluation),
        "project": str(save_dir / project_name),
        "name": run_name,
        "exist_ok": exist_ok,
        "save": True,
        "plots": True
    }

    if extra_args and isinstance(extra_args, dict):
        train_kwargs.update(extra_args)

    try:
        model = YOLO(model_weights)
        results = model.train(**train_kwargs)
        
        output_run_path = save_dir / project_name / run_name
        return True, f"Training complete for {task_type.upper()} task. Artifacts saved to `{output_run_path}`", str(output_run_path)

    except Exception as e:
        return False, f"Training failed with error: {str(e)}", None
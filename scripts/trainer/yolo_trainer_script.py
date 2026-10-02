import os
import shutil
import torch

torch.backends.cudnn.enabled = False
from ultralytics import YOLO


def get_model_path(model_name, model_folder="models"):
  os.makedirs(model_folder, exist_ok=True)
  if os.path.exists(model_name):
    return os.path.abspath(model_name)

  base_name = os.path.basename(model_name)
  target_path = os.path.join(model_folder, base_name)

  if os.path.exists(target_path):
    return os.path.abspath(target_path)

  if os.path.exists(base_name):
    shutil.move(base_name, target_path)
    return os.path.abspath(target_path)

  try:
    temp_model = YOLO(base_name)
    if os.path.exists(base_name):
      shutil.move(base_name, target_path)
      return os.path.abspath(target_path)
  except Exception as e:
    print(f"Could not auto-download model {base_name}: {e}")

  return os.path.abspath(model_name)


def run_yolo_training(data_path, model_name="yolov8n.pt", **kwargs):
  if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"Dataset configuration file not found: {data_path}"
    )

  abs_data_path = os.path.abspath(data_path)

  # Extract project and model_folder from kwargs if present, with defaults
  project = kwargs.pop("project", "output/ultralytics/bbox")
  model_folder = kwargs.pop("model_folder", "models")
  abs_project_path = os.path.abspath(project)

  resolved_model_path = get_model_path(model_name, model_folder)
  model = YOLO(resolved_model_path)

  # Pass all user settings dynamically to Ultralytics model.train
  results = model.train(data=abs_data_path, project=abs_project_path, **kwargs)
  return results
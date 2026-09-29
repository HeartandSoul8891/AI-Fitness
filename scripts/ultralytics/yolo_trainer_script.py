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


def run_yolo_training(
    data_path,
    model_name="yolov8n.pt",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    single_cls=False,
    freeze=0,
    weight_decay=0.0005,
    patience=15,
    project="output/ultralytics/bbox",
    name="exp",
    model_folder="models",
    # Augmentation Parameters
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    erasing=0.4,
    close_mosaic=10,
):
  if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"Dataset configuration file not found: {data_path}"
    )

  abs_data_path = os.path.abspath(data_path)
  abs_project_path = os.path.abspath(project)

  resolved_model_path = get_model_path(model_name, model_folder)
  model = YOLO(resolved_model_path)

  results = model.train(
      data=abs_data_path,
      epochs=epochs,
      imgsz=imgsz,
      batch=batch,
      device=device,
      single_cls=single_cls,
      freeze=freeze,
      weight_decay=weight_decay,
      patience=patience,
      project=abs_project_path,
      name=name,
      # Augmentation Settings
      hsv_h=hsv_h,
      hsv_s=hsv_s,
      hsv_v=hsv_v,
      degrees=degrees,
      translate=translate,
      scale=scale,
      shear=shear,
      perspective=perspective,
      flipud=flipud,
      fliplr=fliplr,
      mosaic=mosaic,
      mixup=mixup,
      copy_paste=copy_paste,
      erasing=erasing,
      close_mosaic=close_mosaic,
  )
  return results
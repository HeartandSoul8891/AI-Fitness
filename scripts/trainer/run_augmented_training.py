#!/usr/bin/env python3
# trigger training without opening streamlit
import os
from scripts.trainer.yolo_trainer_script import run_yolo_training


def main():
  # Paths
  DATA_YAML = "datasets/my_dataset/data.yaml"  # Adjust to your dataset path
  MODEL_NAME = "yolov8n.pt"
  PROJECT_OUTPUT = "output/ultralytics/bbox"
  RUN_NAME = "augmented_run_8k"

  # Hyperparameters
  EPOCHS = 50
  BATCH = 16
  IMGSZ = 640
  DEVICE = 0  # GPU index

  print("🚀 Starting YOLO Augmented Training Run...")

  results = run_yolo_training(
      data_path=DATA_YAML,
      model_name=MODEL_NAME,
      epochs=EPOCHS,
      imgsz=IMGSZ,
      batch=BATCH,
      device=DEVICE,
      patience=15,
      project=PROJECT_OUTPUT,
      name=RUN_NAME,
      # Augmentations
      hsv_h=0.015,  # Hue shift
      hsv_s=0.7,  # Saturation shift
      hsv_v=0.4,  # Value/brightness shift
      degrees=10.0,  # Small rotation jitter
      translate=0.1,  # Translation
      scale=0.5,  # Scale gain
      fliplr=0.5,  # Horizontal flip probability
      flipud=0.0,  # Vertical flip probability
      mosaic=1.0,  # 4-image mosaic stitching
      mixup=0.15,  # Blend images
      erasing=0.4,  # Random erasing
      close_mosaic=10,  # Disable mosaic during last 10 epochs for fine-tuning
  )

  print(f"✅ Training completed! Artifacts saved to {PROJECT_OUTPUT}/{RUN_NAME}")


if __name__ == "__main__":
  main()
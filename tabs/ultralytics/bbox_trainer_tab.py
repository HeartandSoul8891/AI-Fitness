import json
import glob
import os
import streamlit as st
from scripts.ultralytics.yolo_trainer_script import run_yolo_training

PRESETS_DIR = "presets/ultralytics"


def get_yaml_files(dataset_dir="datasets"):
  """Scans the dataset directory recursively for YAML files."""
  yaml_files = []
  if os.path.exists(dataset_dir):
    pattern = os.path.join(dataset_dir, "**", "*.yaml")
    yaml_files = glob.glob(pattern, recursive=True)

  # Also scan root for dataset configs if needed
  yaml_files.extend(glob.glob("*.yaml"))
  return sorted(list(set(yaml_files)))


def get_preset_files():
  os.makedirs(PRESETS_DIR, exist_ok=True)
  return [f for f in os.listdir(PRESETS_DIR) if f.endswith(".json")]


def render_yolo_trainer_ui(mode="bbox"):
  st.header(f"YOLO Training ({mode.upper()})")

  # --- Dataset & Model Configuration ---
  st.subheader("1. Data & Model Selection")

  yaml_options = get_yaml_files()
  if yaml_options:
    data_path = st.selectbox(
        "Dataset YAML File", yaml_options, key=f"{mode}_dataset_path"
    )
  else:
    st.warning("No YAML files found in 'datasets/' directory or root.")
    data_path = st.text_input(
        "Dataset Path manually",
        "datasets/data.yaml",
        key=f"{mode}_manual_data_path",
    )

  col_m1, col_m2 = st.columns(2)
  with col_m1:
    model_name = st.text_input(
        "Base Model", f"yolov8n{'' if mode == 'bbox' else '-seg'}.pt"
    )
  with col_m2:
    project = st.text_input("Output Project Path", f"output/ultralytics/{mode}")

  exp_name = st.text_input("Experiment Name", "exp")

  # --- Hyperparameters & Augmentations ---
  st.subheader("2. Hyperparameters & Augmentations")

  tab_basic, tab_aug = st.tabs(["Basic Hyperparameters", "Augmentations"])

  with tab_basic:
    col1, col2, col3 = st.columns(3)
    with col1:
      epochs = st.number_input("Epochs", min_value=1, value=50)
      batch = st.number_input("Batch Size", min_value=1, value=16)
      imgsz = st.number_input("Image Size", min_value=32, value=640, step=32)
    with col2:
      device = st.text_input("Device (0, cpu, etc.)", "0")
      patience = st.number_input("Patience", min_value=0, value=15)
      freeze = st.number_input("Freeze Layers", min_value=0, value=0)
    with col3:
      conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)
      weight_decay = st.number_input("Weight Decay", value=0.0005, format="%.4f")
      single_cls = st.checkbox("Single Class Mode", value=False)

  with tab_aug:
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
      hsv_h = st.slider("HSV-Hue", 0.0, 1.0, 0.015)
      hsv_s = st.slider("HSV-Saturation", 0.0, 1.0, 0.7)
      hsv_v = st.slider("HSV-Value", 0.0, 1.0, 0.4)
      degrees = st.slider("Rotation Degrees", 0.0, 180.0, 0.0)
    with col_a2:
      translate = st.slider("Translate", 0.0, 1.0, 0.1)
      scale = st.slider("Scale", 0.0, 1.0, 0.5)
      shear = st.slider("Shear", 0.0, 180.0, 0.0)
      perspective = st.slider("Perspective", 0.0, 0.001, 0.0, format="%.4f")
    with col_a3:
      flipud = st.slider("Flip Up-Down", 0.0, 1.0, 0.0)
      fliplr = st.slider("Flip Left-Right", 0.0, 1.0, 0.5)
      mosaic = st.slider("Mosaic", 0.0, 1.0, 1.0)
      mixup = st.slider("Mixup", 0.0, 1.0, 0.0)
      erasing = st.slider("Random Erasing", 0.0, 1.0, 0.4)
      close_mosaic = st.number_input("Close Mosaic Epochs", value=10)

  st.divider()

  # --- Preset Management (Placed right above Start Training) ---
  st.subheader("3. Presets")

  preset_cols = st.columns([2, 2, 1, 1])

  current_settings = {
      "epochs": int(epochs),
      "batch": int(batch),
      "imgsz": int(imgsz),
      "device": device,
      "patience": int(patience),
      "freeze": int(freeze),
      "conf": float(conf),
      "weight_decay": float(weight_decay),
      "single_cls": single_cls,
      "hsv_h": float(hsv_h),
      "hsv_s": float(hsv_s),
      "hsv_v": float(hsv_v),
      "degrees": float(degrees),
      "translate": float(translate),
      "scale": float(scale),
      "shear": float(shear),
      "perspective": float(perspective),
      "flipud": float(flipud),
      "fliplr": float(fliplr),
      "mosaic": float(mosaic),
      "mixup": float(mixup),
      "erasing": float(erasing),
      "close_mosaic": int(close_mosaic),
  }

  with preset_cols[0]:
    preset_files = get_preset_files()
    selected_preset = st.selectbox(
        "Load Preset", ["None"] + preset_files, key=f"{mode}_preset_select"
    )

  with preset_cols[1]:
    new_preset_name = st.text_input(
        "Preset Name", placeholder="default_settings", key=f"{mode}_preset_name"
    )

  with preset_cols[2]:
    st.write(" ")
    st.write(" ")
    if st.button("Save Preset", key=f"{mode}_save_preset"):
      if new_preset_name:
        filename = f"{new_preset_name}.json" if not new_preset_name.endswith(".json") else new_preset_name
        filepath = os.path.join(PRESETS_DIR, filename)
        with open(filepath, "w") as f:
          json.dump(current_settings, f, indent=4)
        st.success(f"Saved: {filename}")
        st.rerun()

  with preset_cols[3]:
    st.write(" ")
    st.write(" ")
    if st.button("Load", key=f"{mode}_load_preset"):
      if selected_preset != "None":
        filepath = os.path.join(PRESETS_DIR, selected_preset)
        with open(filepath, "r") as f:
          loaded_settings = json.load(f)
        st.session_state[f"{mode}_loaded_preset"] = loaded_settings
        st.success(f"Loaded {selected_preset}!")

  st.divider()

  # --- Start Training ---
  if st.button(
      "🚀 Start Training", type="primary", use_container_width=True
  ):
    if not data_path or not os.path.exists(data_path):
      st.error(f"Selected YAML dataset file was not found: {data_path}")
      return

    try:
      with st.spinner("Training in progress..."):
        results = run_yolo_training(
            data_path=data_path,
            model_name=model_name,
            project=project,
            name=exp_name,
            **current_settings,
        )
      st.success("Training Completed Successfully!")
      st.write(results)
    except Exception as e:
      st.error(f"Error during execution: {e}")
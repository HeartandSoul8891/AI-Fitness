import glob
import json
import os
import streamlit as st
from scripts.trainer.yolo_trainer_script import run_yolo_training
from UI.settings_tab import USER_DIR, load_settings

PRESETS_DIR = USER_DIR / "presets" / "ultralytics"


def get_target_dataset_dir():
  if "settings_training_folder" in st.session_state:
    return st.session_state["settings_training_folder"]
  if "settings_datasets_folder" in st.session_state:
    return st.session_state["settings_datasets_folder"]

  saved_settings = load_settings()
  return saved_settings.get(
      "training_folder",
      saved_settings.get("datasets_folder", "datasets"),
  )


def get_yaml_files(dataset_dir=None):
  if dataset_dir is None:
    dataset_dir = get_target_dataset_dir()

  yaml_files = []
  if dataset_dir and os.path.exists(dataset_dir):
    pattern = os.path.join(dataset_dir, "**", "*.yaml")
    yaml_files = glob.glob(pattern, recursive=True)

  if not yaml_files:
    yaml_files.extend(glob.glob("*.yaml"))

  return sorted(list(set(yaml_files)))


def get_model_files(mode="bbox"):
  """Scans configured model/checkpoint directories for .pt files."""
  saved_settings = load_settings()

  key = (
      "ultralytics_bbox_folder"
      if mode == "bbox"
      else "ultralytics_segm_folder"
  )
  target_dir = saved_settings.get(
      key, saved_settings.get("checkpoint_folder", "models")
  )

  pt_files = []
  if target_dir and os.path.exists(target_dir):
    pattern = os.path.join(target_dir, "**", "*.pt")
    pt_files = glob.glob(pattern, recursive=True)

  defaults = (
      ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
      if mode == "bbox"
      else [
          "yolov8n-seg.pt",
          "yolov8s-seg.pt",
          "yolov8m-seg.pt",
          "yolov8l-seg.pt",
          "yolov8x-seg.pt",
      ]
  )

  combined = sorted(list(set(pt_files))) + defaults
  return combined


def get_preset_files():
  os.makedirs(PRESETS_DIR, exist_ok=True)
  return [f for f in os.listdir(PRESETS_DIR) if f.endswith(".json")]


def render_bbox_trainer_ui(mode="bbox"):
  st.header(f"YOLO Training ({mode.upper()})")

  # --- Data & Model Selection ---
  st.subheader("1. Data & Model Selection")

  target_dir = get_target_dataset_dir()
  yaml_options = get_yaml_files(dataset_dir=target_dir)

  if yaml_options:
    data_path = st.selectbox(
        "Dataset YAML File", yaml_options, key=f"{mode}_dataset_path"
    )
  else:
    st.warning(f"No YAML files found in directory: '{target_dir}'")
    data_path = st.text_input(
        "Dataset Path manually",
        os.path.join(target_dir, "data.yaml"),
        key=f"{mode}_manual_data_path",
    )

  col_m1, col_m2 = st.columns(2)
  with col_m1:
    model_options = get_model_files(mode=mode)
    model_name = st.selectbox(
        "Base Model", model_options, key=f"{mode}_model_select"
    )
  with col_m2:
    exp_name = st.text_input(
        "Experiment Name", "exp", key=f"{mode}_exp_name"
    )

  # Auto-resolve base output folder cleanly behind the scenes
  saved_settings = load_settings()
  out_base = saved_settings.get("output_folder", "output")
  project = os.path.join(out_base, "ultralytics", mode)

  # --- Hyperparameters & Augmentations ---
  st.subheader("2. Hyperparameters & Augmentations")

  tab_basic, tab_aug = st.tabs(["Basic Hyperparameters", "Augmentations"])

  with tab_basic:
    col1, col2, col3 = st.columns(3)
    with col1:
      epochs = st.number_input(
          "Epochs", min_value=1, value=50, key=f"{mode}_epochs"
      )
      batch = st.number_input(
          "Batch Size", min_value=1, value=16, key=f"{mode}_batch"
      )
      imgsz = st.number_input(
          "Image Size", min_value=32, value=640, step=32, key=f"{mode}_imgsz"
      )
    with col2:
      device = st.text_input("Device (0, cpu, etc.)", "0", key=f"{mode}_device")
      patience = st.number_input(
          "Patience", min_value=0, value=15, key=f"{mode}_patience"
      )
      freeze = st.number_input(
          "Freeze Layers", min_value=0, value=0, key=f"{mode}_freeze"
      )
    with col3:
      conf = st.slider(
          "Confidence Threshold", 0.0, 1.0, 0.25, key=f"{mode}_conf"
      )
      weight_decay = st.number_input(
          "Weight Decay",
          value=0.0005,
          format="%.4f",
          key=f"{mode}_weight_decay",
      )
      single_cls = st.checkbox(
          "Single Class Mode", value=False, key=f"{mode}_single_cls"
      )

  with tab_aug:
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
      hsv_h = st.slider("HSV-Hue", 0.0, 1.0, 0.015, key=f"{mode}_hsv_h")
      hsv_s = st.slider("HSV-Saturation", 0.0, 1.0, 0.7, key=f"{mode}_hsv_s")
      hsv_v = st.slider("HSV-Value", 0.0, 1.0, 0.4, key=f"{mode}_hsv_v")
      degrees = st.slider(
          "Rotation Degrees", 0.0, 180.0, 0.0, key=f"{mode}_degrees"
      )
    with col_a2:
      translate = st.slider("Translate", 0.0, 1.0, 0.1, key=f"{mode}_translate")
      scale = st.slider("Scale", 0.0, 1.0, 0.5, key=f"{mode}_scale")
      shear = st.slider("Shear", 0.0, 180.0, 0.0, key=f"{mode}_shear")
      perspective = st.slider(
          "Perspective",
          0.0,
          0.001,
          0.0,
          format="%.4f",
          key=f"{mode}_perspective",
      )
    with col_a3:
      flipud = st.slider("Flip Up-Down", 0.0, 1.0, 0.0, key=f"{mode}_flipud")
      fliplr = st.slider("Flip Left-Right", 0.0, 1.0, 0.5, key=f"{mode}_fliplr")
      mosaic = st.slider("Mosaic", 0.0, 1.0, 1.0, key=f"{mode}_mosaic")
      mixup = st.slider("Mixup", 0.0, 1.0, 0.0, key=f"{mode}_mixup")
      erasing = st.slider("Random Erasing", 0.0, 1.0, 0.4, key=f"{mode}_erasing")
      close_mosaic = st.number_input(
          "Close Mosaic Epochs", value=10, key=f"{mode}_close_mosaic"
      )

  st.divider()

  # --- Preset Management ---
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
        "Preset Name",
        placeholder="default_settings",
        key=f"{mode}_preset_name",
    )

  with preset_cols[2]:
    st.write(" ")
    st.write(" ")
    if st.button("Save Preset", key=f"{mode}_save_preset"):
      if new_preset_name:
        filename = (
            f"{new_preset_name}.json"
            if not new_preset_name.endswith(".json")
            else new_preset_name
        )
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
      "🚀 Start Training",
      type="primary",
      use_container_width=True,
      key=f"{mode}_start_btn",
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
            name=exp_name if exp_name.strip() else "exp",
            **current_settings,
        )
      st.success("Training Completed Successfully!")
      st.write(results)
    except Exception as e:
      st.error(f"Error during execution: {e}")
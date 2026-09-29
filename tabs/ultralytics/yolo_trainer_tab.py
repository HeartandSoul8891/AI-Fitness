import json
import os
import streamlit as st
from scripts.settings_script import load_settings
from scripts.ultralytics.yolo_trainer_script import run_yolo_training


def get_profile_paths():
  """Returns target locations for custom presets (root and user home directory)."""
  root_dir = os.path.join(os.getcwd(), "presets")
  user_dir = os.path.expanduser("~/.config/yolo_trainer")

  os.makedirs(root_dir, exist_ok=True)
  os.makedirs(user_dir, exist_ok=True)

  return root_dir, user_dir


def list_saved_profiles():
  """Scans both root/presets and ~/.config/yolo_trainer for saved json profiles."""
  root_dir, user_dir = get_profile_paths()
  profiles = {}

  for location_name, path in [("Root", root_dir), ("User Home", user_dir)]:
    if os.path.exists(path):
      for f in os.listdir(path):
        if f.endswith(".json"):
          profile_name = f"{f[:-5]} ({location_name})"
          profiles[profile_name] = os.path.join(path, f)

  return profiles


def render_yolo_trainer_ui(mode="bbox"):
  is_segm = mode.lower() == "segm"
  task_label = "Segmentation (Segm)" if is_segm else "Bounding Box (BBox)"

  st.title(f"YOLO Trainer — {task_label}")
  st.write(
      "Configure hyperparameters and train Ultralytics YOLO models for"
      f" {task_label.lower()} tasks."
  )

  settings = load_settings()

  # Look under 'training' folder by default (or fallback if settings exist)
  datasets_root = settings.get("datasets_folder") or "training"
  output_root = settings.get("output_folder") or "output"

  if is_segm:
    model_folder = settings.get("ultralytics_segm_folder") or "ultralytics/segm"
    default_models = [
        "yolo26n-seg.pt",
        "yolo26s-seg.pt",
        "yolo26m-seg.pt",
        "yolo26l-seg.pt",
        "yolo26x-seg.pt",
    ]
    project_folder = os.path.join(output_root, "ultralytics", "segm")
  else:
    model_folder = settings.get("ultralytics_bbox_folder") or "ultralytics/bbox"
    default_models = [
        "yolo26n.pt",
        "yolo26s.pt",
        "yolo26m.pt",
        "yolo26l.pt",
        "yolo26x.pt",
    ]
    project_folder = os.path.join(output_root, "ultralytics", "bbox")

  os.makedirs(model_folder, exist_ok=True)
  os.makedirs(project_folder, exist_ok=True)

  st.markdown("---")

  # Preset Manager Section
  st.subheader("💾 Preset & Profile Settings")
  saved_profiles = list_saved_profiles()

  preset_col1, preset_col2 = st.columns([2, 1])

  with preset_col1:
    save_location = st.radio(
        "Save Target Location",
        options=["Root Directory (presets/)", "User Directory (~/.config/)"],
        horizontal=True,
        key=f"preset_loc_{mode}",
    )
    custom_profile_name = st.text_input(
        "Custom Preset Name",
        placeholder="e.g. fast_test_50epochs",
        key=f"preset_name_{mode}",
    )

  with preset_col2:
    if saved_profiles:
      selected_profile = st.selectbox(
          "Load Saved Preset",
          options=["-- Default --"] + list(saved_profiles.keys()),
          key=f"load_preset_select_{mode}",
      )
      if (
          selected_profile != "-- Default --"
          and st.button("Load Preset", key=f"load_preset_btn_{mode}")
      ):
        try:
          with open(saved_profiles[selected_profile], "r") as pf:
            loaded_data = json.load(pf)
            for k, v in loaded_data.items():
              st.session_state[f"{k}_{mode}"] = v
          st.success(f"Loaded preset: {selected_profile}")
          st.rerun()
        except Exception as err:
          st.error(f"Failed to load preset: {err}")

  st.markdown("---")

  # 1. Dataset Selection
  st.subheader("1. Dataset Selection")
  available_yamls = []

  # Recursively find all data.yaml files under training/ and subfolders
  if os.path.exists(datasets_root):
    for root, _, files in os.walk(datasets_root):
      if "data.yaml" in files:
        yaml_full_path = os.path.normpath(os.path.join(root, "data.yaml"))
        available_yamls.append(yaml_full_path)

  if not available_yamls:
    st.warning(
        f"No `data.yaml` files found under `{datasets_root}` or its subfolders."
        " Please build or place your dataset inside the `training/` folder."
    )
    selected_yaml = st.text_input(
        "Or manually specify path to data.yaml",
        value="",
        key=f"manual_yaml_{mode}",
    )
  else:
    selected_yaml = st.selectbox(
        "Select Dataset (`data.yaml`)",
        options=available_yamls,
        format_func=lambda x: os.path.relpath(x, start=os.getcwd())
        if os.path.isabs(x)
        else x,
        key=f"select_yaml_{mode}",
    )

  st.markdown("---")

  # 2. Model Architecture & Setup
  st.subheader("2. Model Architecture & Setup")

  local_models = []
  if os.path.exists(model_folder):
    local_models = [f for f in os.listdir(model_folder) if f.endswith(".pt")]

  all_model_options = local_models + [
      m for m in default_models if m not in local_models
  ]

  col1, col2 = st.columns(2)
  with col1:
    selected_model_option = st.selectbox(
        "Model Weights / Architecture",
        options=all_model_options if all_model_options else default_models,
        help=f"Loaded from model folder: `{model_folder}`",
        key=f"model_select_{mode}",
    )
    epochs = st.number_input(
        "Epochs", min_value=1, max_value=1000, value=50, key=f"epochs_{mode}"
    )
    imgsz = st.selectbox(
        "Image Size",
        options=[640, 960, 1024, 1280],
        index=0,
        key=f"imgsz_{mode}",
    )
    batch = st.selectbox(
        "Batch Size", options=[2, 4, 8, 16, 32], index=3, key=f"batch_{mode}"
    )

  with col2:
    device = st.text_input(
        "Device ID",
        value="0",
        help="GPU index (e.g. 0) or 'cpu'",
        key=f"device_{mode}",
    )
    project = st.text_input(
        "Output Directory", value=project_folder, key=f"project_{mode}"
    )
    run_name = st.text_input(
        "Run Name",
        value="",
        placeholder="e.g. custom_run_name",
        key=f"run_name_{mode}",
    )
    single_cls = st.checkbox(
        "Single Class Mode",
        value=False,
        help="Treat all classes as single identity.",
        key=f"single_cls_{mode}",
    )

  st.markdown("---")

  # 3. Advanced Settings & Confidence Controls
  st.subheader("3. Advanced Settings & Validation")
  col3, col4, col5, col6 = st.columns(4)

  with col3:
    freeze = st.number_input(
        "Freeze Layers",
        min_value=0,
        max_value=25,
        value=0,
        key=f"freeze_{mode}",
    )
  with col4:
    weight_decay = st.number_input(
        "Weight Decay",
        min_value=0.0,
        max_value=0.1,
        value=0.0005,
        format="%.4f",
        key=f"weight_decay_{mode}",
    )
  with col5:
    patience = st.number_input(
        "Patience",
        min_value=0,
        max_value=100,
        value=15,
        key=f"patience_{mode}",
    )
  with col6:
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.35,
        step=0.05,
        key=f"conf_{mode}",
        help="Minimum confidence threshold used for validation and inference.",
    )

  with st.expander("🎨 Data Augmentation Settings (On-the-Fly Pipeline)"):
    aug_col1, aug_col2, aug_col3 = st.columns(3)

    with aug_col1:
      st.markdown("**Color (HSV)**")
      hsv_h = st.slider("HSV-Hue", 0.0, 1.0, 0.015, step=0.005, key=f"hsv_h_{mode}")
      hsv_s = st.slider(
          "HSV-Saturation", 0.0, 1.0, 0.7, step=0.05, key=f"hsv_s_{mode}"
      )
      hsv_v = st.slider(
          "HSV-Value", 0.0, 1.0, 0.4, step=0.05, key=f"hsv_v_{mode}"
      )

    with aug_col2:
      st.markdown("**Geometric & Scale**")
      degrees = st.slider(
          "Rotation (°)", 0.0, 180.0, 0.0, step=5.0, key=f"degrees_{mode}"
      )
      translate = st.slider(
          "Translate", 0.0, 1.0, 0.1, step=0.05, key=f"translate_{mode}"
      )
      scale = st.slider(
          "Scale Gain", 0.0, 1.0, 0.5, step=0.05, key=f"scale_{mode}"
      )
      fliplr = st.slider(
          "Horizontal Flip Prob", 0.0, 1.0, 0.5, step=0.05, key=f"fliplr_{mode}"
      )
      flipud = st.slider(
          "Vertical Flip Prob", 0.0, 1.0, 0.0, step=0.05, key=f"flipud_{mode}"
      )

    with aug_col3:
      st.markdown("**Composition & Erasure**")
      mosaic = st.slider(
          "Mosaic", 0.0, 1.0, 1.0, step=0.1, key=f"mosaic_{mode}"
      )
      mixup = st.slider("MixUp", 0.0, 1.0, 0.0, step=0.1, key=f"mixup_{mode}")
      erasing = st.slider(
          "Random Erasing", 0.0, 1.0, 0.4, step=0.05, key=f"erasing_{mode}"
      )
      close_mosaic = st.number_input(
          "Close Mosaic (Epochs)",
          min_value=0,
          max_value=50,
          value=10,
          key=f"close_mosaic_{mode}",
      )

  # Save Preset Button Action
  if st.button("💾 Save Current Hyperparameters as Preset", key=f"save_preset_btn_{mode}"):
    if not custom_profile_name.strip():
      st.error("Please enter a valid preset name before saving.")
    else:
      root_dir, user_dir = get_profile_paths()
      target_dir = (
          root_dir
          if "Root" in save_location
          else user_dir
      )
      clean_filename = f"{custom_profile_name.strip().replace(' ', '_')}.json"
      save_path = os.path.join(target_dir, clean_filename)

      preset_data = {
          "epochs": epochs,
          "imgsz": imgsz,
          "batch": batch,
          "device": device,
          "single_cls": single_cls,
          "freeze": freeze,
          "weight_decay": weight_decay,
          "patience": patience,
          "conf": conf_threshold,
          "hsv_h": hsv_h,
          "hsv_s": hsv_s,
          "hsv_v": hsv_v,
          "degrees": degrees,
          "translate": translate,
          "scale": scale,
          "fliplr": fliplr,
          "flipud": flipud,
          "mosaic": mosaic,
          "mixup": mixup,
          "erasing": erasing,
          "close_mosaic": close_mosaic,
      }

      try:
        with open(save_path, "w") as pf:
          json.dump(preset_data, pf, indent=4)
        st.success(f"Preset successfully saved to `{save_path}`!")
      except Exception as err:
        st.error(f"Failed to save preset: {err}")

  st.markdown("---")

  # 4. Execute Training
  st.subheader("4. Execute Training")

  parsed_device = int(device) if device.isdigit() else device
  final_model_path = os.path.join(model_folder, selected_model_option)
  final_run_name = run_name.strip() if run_name and run_name.strip() else "exp"

  if st.button("Start Training", type="primary", key=f"start_training_btn_{mode}"):
    if not selected_yaml or not os.path.exists(selected_yaml):
      st.error(
          "Please specify a valid path to a `data.yaml` configuration file."
      )
    else:
      with st.spinner("Training model in progress..."):
        try:
          run_yolo_training(
              data_path=selected_yaml,
              model_name=final_model_path,
              epochs=epochs,
              imgsz=imgsz,
              batch=batch,
              device=parsed_device,
              single_cls=single_cls,
              freeze=freeze,
              weight_decay=weight_decay,
              patience=patience,
              project=project,
              name=final_run_name,
              model_folder=model_folder,
              conf=conf_threshold,
              # Augmentations & Fine-tuning
              hsv_h=hsv_h,
              hsv_s=hsv_s,
              hsv_v=hsv_v,
              degrees=degrees,
              translate=translate,
              scale=scale,
              fliplr=fliplr,
              flipud=flipud,
              mosaic=mosaic,
              mixup=mixup,
              erasing=erasing,
              close_mosaic=close_mosaic,
          )
          st.success(
              "Training completed! Results saved to:"
              f" `{project}/{final_run_name}`"
          )
        except Exception as e:
          st.error(f"Error during execution: {e}")
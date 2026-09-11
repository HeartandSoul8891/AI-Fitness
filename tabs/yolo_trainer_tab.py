import os
import streamlit as st
from scripts.settings_script import load_settings
from scripts.yolo_trainer_script import run_yolo_training

def yolo_trainer_tab():
    st.title("Model Trainer")
    st.write("Configure hyperparameters and execute training runs using Ultralytics YOLO.")

    # Load settings
    settings = load_settings()
    datasets_root = settings.get("custom_datasets_folder") or "datasets"
    
    # Determine app root and training root (pointing to \Yolov-Trainer\training)
    app_root = os.path.dirname(datasets_root) if os.path.dirname(datasets_root) else "."
    training_root = os.path.join(app_root, "training")
    
    model_folder = settings.get("custom_model_folder") or "models"
    training_folder = settings.get("custom_training_folder") or "runs/train"

    # Ensure model folder exists
    os.makedirs(model_folder, exist_ok=True)

    st.markdown("---")

    # 1. Dataset Selection
    st.subheader("1. Dataset Selection")
    
    available_yamls = []
    if os.path.exists(training_root):
        for root, dirs, files in os.walk(training_root):
            if "data.yaml" in files:
                available_yamls.append(os.path.join(root, "data.yaml"))

    if not available_yamls:
        st.warning(f"No `data.yaml` files found under training root (`{training_root}`). Please prepare a dataset first in the Dataset Preparation tab.")
        selected_yaml = st.text_input("Or manually specify path to data.yaml", value="")
    else:
        selected_yaml = st.selectbox("Select Prepared Dataset (`data.yaml`)", available_yamls)

    st.markdown("---")

    # 2. Model & Basic Settings
    st.subheader("2. Model Architecture & Basics")
    
    # Gather local .pt files from the model folder
    local_models = []
    if os.path.exists(model_folder):
        local_models = [f for f in os.listdir(model_folder) if f.endswith(".pt")]
    
    # Standard Ultralytics pretrained baseline options
    standard_models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
    
    # Merge options (local files first, followed by standard baselines)
    all_model_options = local_models + [m for m in standard_models if m not in local_models]

    col1, col2 = st.columns(2)
    
    with col1:
        selected_model_option = st.selectbox(
            "Model Weights / Architecture", 
            options=all_model_options if all_model_options else standard_models,
            help=f"Loaded from model folder: `{model_folder}` or standard Ultralytics presets."
        )
        epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=50)
        imgsz = st.selectbox("Image Size", options=[640, 960, 1024, 1280], index=2)
        batch = st.selectbox("Batch Size", options=[2, 4, 8, 16, 32], index=3)

    with col2:
        device = st.text_input("Device ID", value="0", help="GPU device index (e.g., 0) or 'cpu'")
        project = st.text_input("Output Project Folder", value=training_folder)
        run_name = st.text_input("Run Name", value="", placeholder="e.g. custom_run_name")
        single_cls = st.checkbox(
            "Single Class Mode", 
            value=False, 
            help="Single Class Mode — use only when class identity/name doesn't matter (replaces class names with 'item'). Leave OFF if your dataset already defines its own classes."
        )

    st.markdown("---")
    

    # 3. Advanced Hyperparameters
    st.subheader("3. Advanced Hyperparameters")
    col3, col4, col5 = st.columns(3)

    with col3:
        freeze = st.number_input("Freeze Layers", min_value=0, max_value=25, value=0, help="Number of backbone layers to freeze")
    with col4:
        weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=0.1, value=0.0005, format="%.4f")
    with col5:
        patience = st.number_input("Early Stopping Patience", min_value=0, max_value=100, value=15, help="Epochs to wait with no observable improvement")

    st.markdown("---")

    # 4. Launch Training
    st.subheader("4. Execute Training")
    
    parsed_device = int(device) if device.isdigit() else device

    # Direct model target path to the user's selected model folder
    final_model_path = os.path.join(model_folder, selected_model_option)
    
    # Fallback to 'exp' if run name is left blank
    final_run_name = run_name.strip() if run_name and run_name.strip() else "exp"

    if st.button("Start Training", type="primary"):
        if not selected_yaml or not os.path.exists(selected_yaml):
            st.error("Please provide a valid path to a `data.yaml` configuration file.")
        else:
            with st.spinner("Training model in progress... Check your terminal/console for live Ultralytics logs."):
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
                        model_folder=model_folder
                    )
                    st.success(f"Training completed successfully! Check the output directory: `{project}/{final_run_name}`")
                except Exception as e:
                    st.error(f"Error during training execution: {e}")
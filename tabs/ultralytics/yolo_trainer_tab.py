import os
import streamlit as st
from scripts.settings_script import load_settings
from scripts.ultralytics.yolo_trainer_script import run_yolo_training

def render_yolo_trainer_ui(mode="bbox"):
    """
    Renders the training UI for either 'bbox' or 'segm' modes.
    """
    is_segm = (mode.lower() == "segm")
    task_label = "Segmentation (Segm)" if is_segm else "Bounding Box (BBox)"
    
    st.title(f"YOLO Trainer — {task_label}")
    st.write(f"Configure hyperparameters and train Ultralytics YOLO models for {task_label.lower()} tasks.")

    settings = load_settings()
    
    # 1. Resolve configured folders from settings.json with clear fallbacks
    datasets_root = settings.get("datasets_folder") or "datasets"
    output_root = settings.get("output_folder") or "output"
    
    if is_segm:
        model_folder = settings.get("ultralytics_segm_folder") or "ultralytics/segm"
        default_models = ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt"]
        project_folder = os.path.join(output_root, "ultralytics", "segm")
    else:
        model_folder = settings.get("ultralytics_bbox_folder") or "ultralytics/bbox"
        default_models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
        project_folder = os.path.join(output_root, "ultralytics", "bbox")

    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(project_folder, exist_ok=True)

    st.markdown("---")

    # 2. Dataset Selection
    st.subheader("1. Dataset Selection")
    
    available_yamls = []
    if os.path.exists(datasets_root):
        for root, _, files in os.walk(datasets_root):
            if "data.yaml" in files:
                available_yamls.append(os.path.join(root, "data.yaml"))

    if not available_yamls:
        st.warning(f"No `data.yaml` files found under `{datasets_root}`. Please construct your dataset in the Dataset Preparation tab.")
        selected_yaml = st.text_input("Or manually specify path to data.yaml", value="", key=f"manual_yaml_{mode}")
    else:
        selected_yaml = st.selectbox("Select Dataset (`data.yaml`)", available_yamls, key=f"select_yaml_{mode}")

    st.markdown("---")

    # 3. Model Architecture & Parameters
    st.subheader("2. Model Architecture & Setup")
    
    local_models = []
    if os.path.exists(model_folder):
        local_models = [f for f in os.listdir(model_folder) if f.endswith(".pt")]
    
    all_model_options = local_models + [m for m in default_models if m not in local_models]

    col1, col2 = st.columns(2)
    with col1:
        selected_model_option = st.selectbox(
            "Model Weights / Architecture", 
            options=all_model_options if all_model_options else default_models,
            help=f"Loaded from model folder: `{model_folder}`",
            key=f"model_select_{mode}"
        )
        epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=50, key=f"epochs_{mode}")
        imgsz = st.selectbox("Image Size", options=[640, 960, 1024, 1280], index=0, key=f"imgsz_{mode}")
        batch = st.selectbox("Batch Size", options=[2, 4, 8, 16, 32], index=3, key=f"batch_{mode}")

    with col2:
        device = st.text_input("Device ID", value="0", help="GPU index (e.g. 0) or 'cpu'", key=f"device_{mode}")
        project = st.text_input("Output Directory", value=project_folder, key=f"project_{mode}")
        run_name = st.text_input("Run Name", value="", placeholder="e.g. custom_run_name", key=f"run_name_{mode}")
        single_cls = st.checkbox(
            "Single Class Mode", 
            value=False, 
            help="Treat all classes as single identity.",
            key=f"single_cls_{mode}"
        )

    st.markdown("---")

    # 4. Advanced Hyperparameters
    st.subheader("3. Advanced Settings")
    col3, col4, col5 = st.columns(3)

    with col3:
        freeze = st.number_input("Freeze Layers", min_value=0, max_value=25, value=0, key=f"freeze_{mode}")
    with col4:
        weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=0.1, value=0.0005, format="%.4f", key=f"weight_decay_{mode}")
    with col5:
        patience = st.number_input("Patience", min_value=0, max_value=100, value=15, key=f"patience_{mode}")

    st.markdown("---")

    # 5. Execute Training
    st.subheader("4. Execute Training")
    
    parsed_device = int(device) if device.isdigit() else device
    final_model_path = os.path.join(model_folder, selected_model_option)
    final_run_name = run_name.strip() if run_name and run_name.strip() else "exp"

    if st.button("Start Training", type="primary", key=f"start_training_btn_{mode}"):
        if not selected_yaml or not os.path.exists(selected_yaml):
            st.error("Please specify a valid path to a `data.yaml` configuration file.")
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
                        model_folder=model_folder
                    )
                    st.success(f"Training completed! Results saved to: `{project}/{final_run_name}`")
                except Exception as e:
                    st.error(f"Error during execution: {e}")
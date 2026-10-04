import json
import os
import sys
from pathlib import Path
import streamlit as st

# FIX: Changed Path(file) to Path(__file__)
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent if FILE_PATH.parent.name == "tabs" else FILE_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def load_settings_clean():
    raw = {}
    try:
        from scripts.settings.settings_script import load_settings as _load
        raw = _load()
    except ImportError:
        settings_file = PROJECT_ROOT / "user" / "settings.json"
        if settings_file.exists():
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            except Exception:
                pass
    
    # Strip accidental trailing spaces from keys and values
    return {str(k).strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}

try:
    from scripts.trainer.yolo_complete_script import start_yolo_training, resolve_model_save_dir, TASK_BASE_MODELS
except ImportError:
    TASK_BASE_MODELS = {
        "bbox": "yolo11n.pt", "segm": "yolo11n-seg.pt", "obb": "yolo11n-obb.pt",
        "pose": "yolo11n-pose.pt", "cls": "yolo11n-cls.pt", "dept": "yolo11n.pt"
    }
    def resolve_model_save_dir(task_type):
        return PROJECT_ROOT / "output" / "models" / task_type
    def start_yolo_training(*args, **kwargs):
        return False, "Trainer script module not found.", None

ULTRALYTICS_FOLDER_KEYS = [
    "ultralytics_bbox_folder", "ultralytics_cls_folder", "ultralytics_segm_folder",
    "ultralytics_obb_folder", "ultralytics_pose_folder", "ultralytics_dept_folder",
]

def trainer_tab():
    st.title("🚀 Multi-Task YOLO Model Trainer")
    st.write("Train and fine-tune Ultralytics YOLO models with fully customizable training parameters.")
    
    settings = load_settings_clean()
    training_root = settings.get("training_folder") or str(PROJECT_ROOT / "training")

    # 1. Task & Device Selection
    st.subheader("1. Task & Compute Device Setup")
    col_task, col_device = st.columns(2)
    with col_task:
        task_type = st.selectbox(
            "YOLO Task Type",
            options=["bbox", "segm", "obb", "pose", "cls", "dept"],
            format_func=lambda x: {
                "bbox": "Bounding Box (Detection)", "segm": "Instance Segmentation",
                "obb": "Oriented Bounding Box (OBB)", "pose": "Pose / Keypoint Estimation",
                "cls": "Image Classification", "dept": "Depth / Distance Estimation"
            }[x],
            key="trn_task_type"
        )
    with col_device:
        device_opt = st.selectbox(
            "Compute Hardware Device",
            options=["auto", "gpu", "cpu"],
            format_func=lambda x: {
                "auto": "Auto-Detect (GPU / ROCm / CUDA)",
                "gpu": "GPU (CUDA / ROCm Index 0)",
                "cpu": "CPU Only"
            }[x],
            key="trn_device_opt"
        )

    st.markdown("---")

    # 2. Dataset & Base Weights
    st.subheader("2. Dataset & Base Weights")
    dset_col, weight_col = st.columns(2)
    
    with dset_col:
        prepared_datasets = []
        if os.path.exists(training_root):
            prepared_datasets = [d for d in os.listdir(training_root) if os.path.isdir(os.path.join(training_root, d))]
        if not prepared_datasets:
            prepared_datasets = ["."]
            
        selected_ds_folder = st.selectbox("Select Prepared Training Dataset", prepared_datasets, key="trn_dataset_select")
        dataset_full_path = os.path.join(training_root, selected_ds_folder) if selected_ds_folder != "." else training_root
        
        st.caption(f"Target path: `{dataset_full_path}`")
        
        # --- DATASET STRUCTURE VALIDATOR --- (unchanged)
        if os.path.exists(dataset_full_path) and selected_ds_folder != ".":
            checks = []
            if os.path.exists(os.path.join(dataset_full_path, "data.yaml")):
                checks.append("✅ `data.yaml`")
            else:
                checks.append("❌ `data.yaml`")
                
            if os.path.exists(os.path.join(dataset_full_path, "train", "images")):
                checks.append("✅ `train/images`")
            else:
                checks.append("❌ `train/images`")
                
            if os.path.exists(os.path.join(dataset_full_path, "val", "images")):
                checks.append("✅ `val/images`")
            else:
                checks.append("❌ `val/images`")
                
            st.markdown("**Dataset Structure:** " + " | ".join(checks))
            
            yaml_path = os.path.join(dataset_full_path, "data.yaml")
            if os.path.exists(yaml_path):
                with st.expander("📄 View `data.yaml` content"):
                    with open(yaml_path, "r", encoding="utf-8") as f:
                        st.code(f.read(), language="yaml")
        # -----------------------------------

    with weight_col:
        default_weights = TASK_BASE_MODELS.get(task_type, "yolo11n.pt")
        available_weights = [default_weights]
        model_paths_map = {default_weights: default_weights}

        for folder_key in ULTRALYTICS_FOLDER_KEYS:
            folder_path = settings.get(folder_key, "")
            if folder_path and os.path.exists(folder_path):
                for file_name in sorted(os.listdir(folder_path)):
                    if file_name.endswith((".pt", ".yaml", ".onnx", ".engine")):
                        display_name = file_name
                        if display_name in model_paths_map and model_paths_map[display_name] != os.path.join(folder_path, file_name):
                            subfolder_tag = os.path.basename(folder_path)
                            display_name = f"{file_name} ({subfolder_tag})"
                        if display_name not in available_weights:
                            available_weights.append(display_name)
                        model_paths_map[display_name] = os.path.join(folder_path, file_name)

        selected_weight_name = st.selectbox(
            "Pretrained Model Weights (.pt)",
            options=available_weights,
            key="trn_weights_select"
        )
        weights_path = model_paths_map.get(selected_weight_name, selected_weight_name)
        st.caption(f"Selected model path: `{weights_path}`")

    st.markdown("---")

    # 3. Training Hyperparameters (unchanged)
    st.subheader("3. Training Hyperparameters")
    hp_col1, hp_col2, hp_col3, hp_col4 = st.columns(4)
    with hp_col1:
        epochs = st.number_input("Epochs", min_value=1, max_value=5000, value=50, step=5, key="trn_epochs")
        optimizer = st.selectbox("Optimizer", ["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"], key="trn_optimizer")
    with hp_col2:
        batch_size = st.number_input("Batch Size (-1 for Auto)", min_value=-1, max_value=512, value=16, step=2, key="trn_batch")
        workers = st.number_input("Dataloader Workers", min_value=0, max_value=64, value=4, step=1, key="trn_workers")
    with hp_col3:
        imgsz = st.number_input("Image Size (px)", min_value=128, max_value=2048, value=640, step=32, key="trn_imgsz")
        patience = st.number_input("Patience (Early Stop)", min_value=0, max_value=500, value=50, step=5, key="trn_patience")
    with hp_col4:
        lr = st.number_input("Initial Learning Rate (lr0)", min_value=0.00001, max_value=0.1, value=0.01, step=0.001, format="%.5f", key="trn_lr")
        seed = st.number_input("Random Seed", min_value=0, max_value=999999, value=0, key="trn_seed")

    st.markdown("##### Additional Controls")
    opt_col1, opt_col2, opt_col3 = st.columns(3)
    with opt_col1:
        cos_lr = st.checkbox("Cosine Learning Rate Scheduler", value=False, key="trn_cos_lr")
    with opt_col2:
        augment = st.checkbox("Enable Data Augmentation", value=True, key="trn_augment")
    with opt_col3:
        val_eval = st.checkbox("Run Validation Each Epoch", value=True, key="trn_val_eval")

    st.markdown("---")

    # 4. Project & Run Naming (unchanged)
    st.subheader("4. Experiment Logging")
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        project_name = st.text_input("Project / Experiment Folder", value=f"yolo_{task_type}_project", key="trn_proj_name")
    with e_col2:
        run_name = st.text_input("Run Name", value="train_run_01", key="trn_run_name")

    st.markdown("---")

    # 5. Execution (unchanged)
    if st.button("🔥 Start YOLO Training", type="primary", use_container_width=True, key="trn_start_btn"):
        if not os.path.exists(dataset_full_path):
            st.error(f"Selected dataset folder does not exist: `{dataset_full_path}`")
            return
            
        with st.spinner(f"Training {task_type.upper()} model... Check terminal logs for epoch updates."):
            success, message, output_path = start_yolo_training(
                task_type=task_type,
                dataset_path=dataset_full_path,
                model_weights=weights_path,
                epochs=epochs,
                batch_size=batch_size,
                imgsz=imgsz,
                device=device_opt,
                learning_rate=lr,
                patience=patience,
                optimizer=optimizer,
                workers=workers,
                seed=seed,
                cos_lr=cos_lr,
                augment=augment,
                val_evaluation=val_eval,
                project_name=project_name,
                run_name=run_name
            )
            
            if success:
                st.success(message)
                if output_path and os.path.exists(output_path):
                    st.markdown("### 📊 Training Artifacts")
                    results_png = os.path.join(output_path, "results.png")
                    confusion_png = os.path.join(output_path, "confusion_matrix.png")
                    col_a, col_b = st.columns(2)
                    if os.path.exists(results_png):
                        col_a.image(results_png, caption="Results Metrics Plot", use_container_width=True)
                    if os.path.exists(confusion_png):
                        col_b.image(confusion_png, caption="Confusion Matrix", use_container_width=True)
            else:
                st.error(message)

# Module Entry Point Aliases
render_tab = trainer_tab
render_trainer_tab = trainer_tab

if __name__ == "__main__":
    st.set_page_config(page_title="YOLO Multi-Task Trainer", page_icon="🚀", layout="wide")
    trainer_tab()
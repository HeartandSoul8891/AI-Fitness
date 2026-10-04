import json
import os
import sys
import glob
from pathlib import Path
import streamlit as st

# ==========================================
# PATH RESOLUTION & IMPORTS
# ==========================================
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent if FILE_PATH.parent.name == "UI" else FILE_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.yolo_trainer_script import start_yolo_training, TASK_BASE_MODELS, load_settings
except ImportError:
    st.error("Could not import `yolo_trainer_script.py`. Ensure it is in `scripts/`.")
    # Fallbacks to prevent UI crash if backend is missing
    TASK_BASE_MODELS = {"bbox": "yolo11n.pt", "segm": "yolo11n-seg.pt", "obb": "yolo11n-obb.pt", "pose": "yolo11n-pose.pt", "cls": "yolo11n-cls.pt"}
    def load_settings(): return {}
    def start_yolo_training(*args, **kwargs): return False, "Script missing", None

PRESETS_DIR = PROJECT_ROOT / "user" / "presets" / "ultralytics"

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_yaml_files(settings: dict):
    """Scans both datasets and training folders for data.yaml files."""
    yaml_files = []
    search_dirs = [
        settings.get("datasets_folder", "datasets"),
        settings.get("training_folder", "training")
    ]
    for d in search_dirs:
        if d and os.path.exists(d):
            yaml_files.extend(glob.glob(os.path.join(d, "**", "*.yaml"), recursive=True))
    return sorted(list(set(yaml_files)))

def get_model_files(task_type: str, settings: dict):
    """Scans the unified ultralytics models folder for .pt weights."""
    target_dir = settings.get("ultralytics_models_folder", settings.get("models_folder", "models"))
    pt_files = []
    if target_dir and os.path.exists(target_dir):
        pt_files = glob.glob(os.path.join(target_dir, "**", "*.pt"), recursive=True)
    
    default_model = TASK_BASE_MODELS.get(task_type, "yolo11n.pt")
    unique_files = sorted(list(set(pt_files)))
    
    # Ensure the default model is always an option at the top
    if default_model not in unique_files:
        unique_files.insert(0, default_model)
    return unique_files

# ==========================================
# MAIN GUI RENDER
# ==========================================
def render_yolo_tab():
    st.title("🚀 Unified Multi-Task YOLO Trainer")
    st.write("Train, fine-tune, and augment any Ultralytics YOLO model from a single interface.")
    
    settings = load_settings()
    
    # --- PRESET LOADING LOGIC (Must happen before widgets render) ---
    if "load_preset_flag" in st.session_state:
        preset_data = st.session_state.pop("load_preset_flag")
        for k, v in preset_data.items():
            st.session_state[f"trn_{k}"] = v

    # ==========================================
    # 1. TASK & DEVICE SETUP
    # ==========================================
    st.subheader("1. Task & Compute Device")
    col_task, col_device = st.columns(2)
    with col_task:
        task_type = st.selectbox("YOLO Task Type", options=list(TASK_BASE_MODELS.keys()), 
                                 format_func=lambda x: {"bbox":"Detection (BBox)","segm":"Segmentation","obb":"Oriented BB","pose":"Pose","cls":"Classification","dept":"Depth"}[x], 
                                 key="trn_task_type")
    with col_device:
        device_opt = st.selectbox("Compute Device", ["auto", "gpu", "cpu"], key="trn_device_opt")

    st.markdown("---")

    # ==========================================
    # 2. DATASET & WEIGHTS
    # ==========================================
    st.subheader("2. Dataset & Base Weights")
    dset_col, weight_col = st.columns(2)
    
    with dset_col:
        yaml_files = get_yaml_files(settings)
        if yaml_files:
            data_path = st.selectbox("Select Dataset YAML", yaml_files, key="trn_data_path")
        else:
            st.warning(f"No YAML files found in `datasets/` or `training/`. Did you run Dataset Prep?")
            data_path = st.text_input("Or enter path manually", os.path.join(settings.get("datasets_folder", "datasets"), "data.yaml"), key="trn_data_path_manual")
            data_path = data_path if data_path else os.path.join(settings.get("datasets_folder", "datasets"), "data.yaml")

        # Dataset Validator
        if os.path.exists(data_path):
            checks = []
            if task_type != "cls":
                checks.append("✅ `data.yaml`" if os.path.exists(data_path) else "❌ `data.yaml`")
                base_dir = os.path.dirname(data_path)
                checks.append("✅ `train/images`" if os.path.exists(os.path.join(base_dir, "train", "images")) else "❌ `train/images`")
                checks.append("✅ `val/images`" if os.path.exists(os.path.join(base_dir, "val", "images")) else "❌ `val/images`")
            else:
                checks.append("✅ `data.yaml`")
            st.markdown("**Dataset Check:** " + " | ".join(checks))
            
    with weight_col:
        model_options = get_model_files(task_type, settings)
        model_weights = st.selectbox("Pretrained Weights (.pt)", model_options, key="trn_model_weights")

    st.markdown("---")

    # ==========================================
    # 3. HYPERPARAMETERS & AUGMENTATIONS
    # ==========================================
    st.subheader("3. Hyperparameters & Augmentations")
    tab_basic, tab_aug = st.tabs(["⚙️ Basic Hyperparameters", "🎨 Augmentations"])
    
    with tab_basic:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            epochs = st.number_input("Epochs", 1, 5000, 50, key="trn_epochs")
            batch = st.number_input("Batch Size (-1=Auto)", -1, 512, 16, key="trn_batch")
        with c2:
            imgsz = st.number_input("Image Size (px)", 128, 2048, 640, step=32, key="trn_imgsz")
            patience = st.number_input("Patience (Early Stop)", 0, 500, 50, key="trn_patience")
        with c3:
            lr = st.number_input("Initial LR (lr0)", 0.00001, 0.1, 0.01, step=0.001, format="%.5f", key="trn_lr")
            workers = st.number_input("Dataloader Workers", 0, 64, 4, key="trn_workers")
        with c4:
            optimizer = st.selectbox("Optimizer", ["auto", "SGD", "Adam", "AdamW", "RMSProp"], key="trn_optimizer")
            seed = st.number_input("Random Seed", 0, 999999, 0, key="trn_seed")
        
        c5, c6, c7 = st.columns(3)
        with c5:
            freeze = st.number_input("Freeze Layers (e.g., 10)", 0, 50, 0, key="trn_freeze")
            conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, key="trn_conf")
        with c6:
            weight_decay = st.number_input("Weight Decay", 0.0, 0.1, 0.0005, format="%.4f", key="trn_weight_decay")
            single_cls = st.checkbox("Single Class Mode", key="trn_single_cls")
        with c7:
            cos_lr = st.checkbox("Cosine LR Scheduler", key="trn_cos_lr")
            val_eval = st.checkbox("Run Validation", value=True, key="trn_val_eval")

    with tab_aug:
        a1, a2, a3 = st.columns(3)
        with a1:
            hsv_h = st.slider("HSV-Hue", 0.0, 1.0, 0.015, key="trn_hsv_h")
            hsv_s = st.slider("HSV-Saturation", 0.0, 1.0, 0.7, key="trn_hsv_s")
            hsv_v = st.slider("HSV-Value", 0.0, 1.0, 0.4, key="trn_hsv_v")
            degrees = st.slider("Rotation Degrees", 0.0, 180.0, 0.0, key="trn_degrees")
        with a2:
            translate = st.slider("Translate", 0.0, 1.0, 0.1, key="trn_translate")
            scale = st.slider("Scale", 0.0, 1.0, 0.5, key="trn_scale")
            shear = st.slider("Shear", 0.0, 180.0, 0.0, key="trn_shear")
            perspective = st.slider("Perspective", 0.0, 0.001, 0.0, format="%.4f", key="trn_perspective")
        with a3:
            flipud = st.slider("Flip Up-Down", 0.0, 1.0, 0.0, key="trn_flipud")
            fliplr = st.slider("Flip Left-Right", 0.0, 1.0, 0.5, key="trn_fliplr")
            mosaic = st.slider("Mosaic", 0.0, 1.0, 1.0, key="trn_mosaic")
            mixup = st.slider("Mixup", 0.0, 1.0, 0.0, key="trn_mixup")
        
        c_aug2, _ = st.columns([1, 2])
        with c_aug2:
            erasing = st.slider("Random Erasing", 0.0, 1.0, 0.4, key="trn_erasing")
            close_mosaic = st.number_input("Close Mosaic (Last N Epochs)", 0, 50, 10, key="trn_close_mosaic")

    st.markdown("---")

    # ==========================================
    # 4. PRESETS MANAGEMENT
    # ==========================================
    st.subheader("4. Presets")
    os.makedirs(PRESETS_DIR, exist_ok=True)
    preset_files = [f for f in os.listdir(PRESETS_DIR) if f.endswith(".json")]
    
    p_col1, p_col2, p_col3, p_col4 = st.columns([2, 2, 1, 1])
    
    # Gather current UI state for saving
    current_settings = {k.replace("trn_", ""): v for k, v in st.session_state.items() if k.startswith("trn_") and k not in ["trn_task_type", "trn_device_opt", "trn_data_path", "trn_data_path_manual", "trn_model_weights"]}
    
    with p_col1:
        selected_preset = st.selectbox("Load Preset", ["None"] + preset_files, key="trn_preset_select")
    with p_col2:
        new_preset_name = st.text_input("New Preset Name", placeholder="my_augmented_run", key="trn_preset_name")
    with p_col3:
        st.write(" ")
        if st.button("💾 Save", use_container_width=True):
            if new_preset_name:
                filename = f"{new_preset_name}.json" if not new_preset_name.endswith(".json") else new_preset_name
                with open(PRESETS_DIR / filename, "w") as f:
                    json.dump(current_settings, f, indent=4)
                st.success(f"Saved {filename}")
                st.rerun()
    with p_col4:
        st.write(" ")
        if st.button("📂 Load", use_container_width=True):
            if selected_preset != "None":
                with open(PRESETS_DIR / selected_preset, "r") as f:
                    st.session_state["load_preset_flag"] = json.load(f)
                st.rerun()

    st.markdown("---")

    # ==========================================
    # 5. EXECUTION
    # ==========================================
    st.subheader("5. Experiment Logging")
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        project_name = st.text_input("Project Folder", value=f"yolo_{task_type}_project", key="trn_proj_name")
    with e_col2:
        run_name = st.text_input("Run Name", value="train_run_01", key="trn_run_name")

    if st.button("🔥 Start Unified Training", type="primary", use_container_width=True):
        if not os.path.exists(data_path):
            st.error(f"Dataset path not found: `{data_path}`")
            return
        
        with st.spinner(f"Training {task_type.upper()} model... Check terminal for epoch logs."):
            success, message, output_path = start_yolo_training(
                task_type=task_type,
                dataset_path=data_path,
                model_weights=model_weights,
                project_name=project_name,
                run_name=run_name,
                device=device_opt,
                **current_settings # Passes all hyperparameters and augmentations cleanly
            )
            
            if success:
                st.success(message)
                if output_path and os.path.exists(output_path):
                    st.markdown("### 📊 Training Artifacts")
                    results_png = os.path.join(output_path, "results.png")
                    confusion_png = os.path.join(output_path, "confusion_matrix.png")
                    col_a, col_b = st.columns(2)
                    if os.path.exists(results_png): col_a.image(results_png, caption="Results Metrics", use_container_width=True)
                    if os.path.exists(confusion_png): col_b.image(confusion_png, caption="Confusion Matrix", use_container_width=True)
            else:
                st.error(message)

# ==========================================
# ENTRY POINT ALIASES
# ==========================================
render_trainer_tab = render_yolo_tab

if __name__ == "__main__":
    st.set_page_config(page_title="Unified YOLO Trainer", page_icon="🚀", layout="wide")
    render_yolo_tab()
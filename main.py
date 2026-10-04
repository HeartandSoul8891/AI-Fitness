import json
import os
import sys
import streamlit as st
from pathlib import Path

# Ensure project root is in system path
current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))

st.set_page_config(
    page_title="AI-Fitness",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 1. SETTINGS TAB LOGIC
# ==============================================================================
PROJECT_ROOT = current_dir
DEFAULT_DATASETS_PATH = os.path.join(PROJECT_ROOT, "datasets")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output")
DEFAULT_TRAINING_PATH = os.path.join(PROJECT_ROOT, "training")
DEFAULT_WD14_TAGGER_PATH = os.path.join(PROJECT_ROOT, "datasets")

USER_DIR = PROJECT_ROOT / "user"
USER_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE = USER_DIR / "settings.json"

COMFY_FOLDERS = {
    "blip_folder": "blip",
    "checkpoint_folder": "checkpoints",
    "clip_folder": "clip",
    "clip_vision_folder": "clip_vision",
    "controlnet_folder": "controlnet",
    "diffusion_models_folder": "diffusion_models",
    "embeddings_folder": "embeddings",
    "hypernetworks_folder": "hypernetworks",
    "loras_folder": "loras",
    "text_encoders_folder": "text_encoders",
    "unet_folder": "unet",
    "upscale_models_folder": "upscale_models",
    "vae_folder": "vae",
    "ultralytics_bbox_folder": "ultralytics/bbox",
    "ultralytics_cls_folder": "ultralytics/cls",
    "ultralytics_segm_folder": "ultralytics/segm",
    "ultralytics_obb_folder": "ultralytics/obb",
    "ultralytics_pose_folder": "ultralytics/pose",
    "ultralytics_dept_folder": "ultralytics/dept",
}

def load_settings():
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Fout bij het lezen van settings.json: {e}")
    return {}

def sync_paths_from_root():
    root = st.session_state.get("root_folder", "").strip()
    if not root:
        return
    for key, subfolder in COMFY_FOLDERS.items():
        session_key = f"settings_{key}"
        st.session_state[session_key] = os.path.normpath(os.path.join(root, subfolder))

def save_settings(settings_dict):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings_dict, f, indent=4)
        st.toast("Applicatie-instellingen bijgewerkt.", icon="⚙️")
        st.success("Instellingen succesvol opgeslagen!")
    except Exception as e:
        st.error(f"Fout bij het opslaan van instellingen: {e}")

def render_settings_tab():
    st.title("⚙ Settings & Configuration")
    saved_settings = load_settings()
    
    if "root_folder" not in st.session_state:
        st.session_state["root_folder"] = saved_settings.get("root_folder", "")
    if "settings_datasets_folder" not in st.session_state:
        st.session_state["settings_datasets_folder"] = saved_settings.get("datasets_folder", DEFAULT_DATASETS_PATH)
    if "settings_wd14_tagger_folder" not in st.session_state:
        st.session_state["settings_wd14_tagger_folder"] = saved_settings.get("wd14_tagger_folder", DEFAULT_WD14_TAGGER_PATH)
    if "settings_output_folder" not in st.session_state:
        st.session_state["settings_output_folder"] = saved_settings.get("output_folder", DEFAULT_OUTPUT_PATH)
    if "settings_training_folder" not in st.session_state:
        st.session_state["settings_training_folder"] = saved_settings.get("training_folder", DEFAULT_TRAINING_PATH)
        
    for key in COMFY_FOLDERS:
        session_key = f"settings_{key}"
        if session_key not in st.session_state:
            saved_val = saved_settings.get(key, "")
            if not saved_val and st.session_state["root_folder"]:
                saved_val = os.path.normpath(os.path.join(st.session_state["root_folder"], COMFY_FOLDERS[key]))
            st.session_state[session_key] = saved_val

    st.text_input("Root Folder", key="root_folder", on_change=sync_paths_from_root)

    if st.button("Save Settings", type="primary", use_container_width=True):
        if st.session_state.get("root_folder"):
            sync_paths_from_root()
        settings_dict = {
            "root_folder": st.session_state.get("root_folder", ""),
            "datasets_folder": st.session_state.get("settings_datasets_folder", DEFAULT_DATASETS_PATH),
            "wd14_tagger_folder": st.session_state.get("settings_wd14_tagger_folder", DEFAULT_WD14_TAGGER_PATH),
            "output_folder": st.session_state.get("settings_output_folder", DEFAULT_OUTPUT_PATH),
            "training_folder": st.session_state.get("settings_training_folder", DEFAULT_TRAINING_PATH),
        }
        for key in COMFY_FOLDERS:
            settings_dict[key] = st.session_state.get(f"settings_{key}", "")
        save_settings(settings_dict)

    st.divider()
    st.subheader("App Working Directories")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.text_input("Datasets Folder Path", key="settings_datasets_folder", help="Standaard pad voor dataset opslag")
    with col2:
        st.text_input("WD14 Tagger Folder Path", key="settings_wd14_tagger_folder", help="Standaard dataset pad gebruikt door de WD14 Tagger")
    with col3:
        st.text_input("Output Folder Path", key="settings_output_folder", help="Standaard pad: AI-Fitness/output")
    with col4:
        st.text_input("Training Folder Path", key="settings_training_folder", help="Standaard pad: AI-Fitness/training")

    st.divider()
    st.subheader("Model Directory Paths")
    for key in COMFY_FOLDERS:
        label = key.replace("_", " ").title()
        st.text_input(label, key=f"settings_{key}")


# ==============================================================================
# 2. DATASETS TAB LOGIC
# ==============================================================================
def render_datasets_tab():
    st.title("📦 Multi-Task Dataset Preparation")
    st.write("Prepare, structure, and convert tagged datasets into task-ready YOLO formats (BBox, Segmentation, OBB, Pose, Classification).")
    
    try:
        from scripts.datasets.dataset_preparation_script import (
            get_dataset_folders,
            get_unique_labels_from_source,
            create_dataset_structure,
            move_and_split_files,
            convert_dataset_jsons,
            generate_yaml,
            cleanup_empty_jsons_and_media
        )
        try:
            from scripts.settings.settings_script import load_settings as load_script_settings
            settings = load_script_settings()
        except ImportError:
            settings = load_settings()
            
    except ImportError as e:
        st.error(f"⚠️ Failed to import dataset preparation scripts. Please ensure `scripts/datasets/dataset_preparation_script.py` exists.\nDetails: {e}")
        st.stop()

    datasets_root = settings.get("datasets_folder") or "datasets"
    training_root = settings.get("training_folder") or "training"
    os.makedirs(datasets_root, exist_ok=True)
    os.makedirs(training_root, exist_ok=True)
    
    st.info(f"**Tagged Datasets Source Root:** `{datasets_root}`\n\n**Training Output Root:** `{training_root}`")
    st.markdown("---")
    
    st.subheader("1. Select Task & Source Dataset")
    task_type = st.selectbox(
        "YOLO Model Task Type",
        options=["bbox", "segm", "obb", "pose", "cls"],
        format_func=lambda x: {
            "bbox": "Bounding Box (Detection)",
            "segm": "Instance Segmentation",
            "obb": "Oriented Bounding Box (OBB)",
            "pose": "Pose / Keypoint Estimation",
            "cls": "Image Classification"
        }[x],
        key="ds_task_type_select"
    )
    
    tagged_folders = get_dataset_folders(datasets_root) or ["."]
    selected_subfolder = st.selectbox("Select Tagged Dataset Folder", tagged_folders, key="ds_folder_select") or "."
    source_dir = datasets_root if selected_subfolder == "." else os.path.join(datasets_root, selected_subfolder)
    
    output_dataset_name = st.text_input(
        "Output Training Dataset Folder Name", 
        value=f"{selected_subfolder}_{task_type}" if selected_subfolder != "." else f"prepared_{task_type}_dataset",
        key="ds_output_name_input"
    )
    dataset_path = os.path.join(training_root, output_dataset_name)
    
    st.markdown("### 🧹 Source Dataset Hygiene")
    st.caption("Scan source folder to purge empty JSONs and matching unused images/visualizers.")
    if st.button("Purge Empty JSONs & Dead Media", type="secondary", key="ds_purge_btn"):
        success, msg = cleanup_empty_jsons_and_media(source_dir)
        if success:
            st.success(msg)
        else:
            st.error(msg)
            
    st.markdown("---")
    st.subheader("2. Create Training Folder Structure")
    st.write(f"Creates YOLO dataset tree inside `{dataset_path}`.")
    if st.button("Create Folder Structure", type="primary", key="ds_create_struct_btn"):
        try:
            path = create_dataset_structure(training_root, output_dataset_name, task_type)
            st.success(f"Folder structure created successfully at: `{path}`")
        except Exception as e:
            st.error(f"Error creating folder structure: {e}")
            
    st.markdown("---")
    st.subheader("3. Move & Split Dataset")
    st.write(f"Distributes images and `.json` annotations from `{source_dir}` into train/val subsets.")
    split_ratio = st.slider("Train Split Ratio", min_value=0.5, max_value=0.95, value=0.8, step=0.05, key="ds_split_slider")
    if st.button("Move & Split Dataset", key="ds_move_split_btn"):
        success, msg = move_and_split_files(source_dir, dataset_path, split_ratio)
        if success:
            st.success(msg)
        else:
            st.error(msg)
            
    st.markdown("---")
    st.subheader("4. Convert Annotations to YOLO Format")
    st.write(f"Discovers unique labels and converts annotations into `{task_type}` compliant files.")
    discovered_labels = get_unique_labels_from_source(source_dir)
    default_mapping_str = json.dumps(discovered_labels if discovered_labels else {"DEFAULT": 0}, indent=4)
    class_mapping_input = st.text_area("Class Mapping (JSON Dictionary)", value=default_mapping_str, height=130, key="ds_class_map_input")
    
    if st.button("Convert Dataset to YOLO Format", key="ds_convert_btn"):
        try:
            class_mapping = json.loads(class_mapping_input)
            success, msg = convert_dataset_jsons(dataset_path, class_mapping, task_type)
            if success:
                st.success(msg)
            else:
                st.error(msg)
        except json.JSONDecodeError:
            st.error("Invalid JSON format in Class Mapping box.")
        except Exception as e:
            st.error(f"Error during conversion: {e}")
            
    st.markdown("---")
    st.subheader("5. Generate data.yaml")
    st.write("Generates configuration required to start Ultralytics training.")
    if st.button("Generate data.yaml", type="primary", key="ds_yaml_btn"):
        try:
            class_mapping = json.loads(class_mapping_input)
            success, msg = generate_yaml(dataset_path, class_mapping, task_type)
            if success:
                st.success(msg)
            else:
                st.error(msg)
        except json.JSONDecodeError:
            st.error("Invalid JSON format in Class Mapping.")
        except Exception as e:
            st.error(f"Error generating YAML: {e}")


# ==============================================================================
# 3. TAGGER TAB LOGIC
# ==============================================================================
def render_tagger_tab():
    st.header("🏷️ Dataset Tagger & Annotator")
    (
        tab_auto_bbox, tab_auto_segm, tab_cls, tab_sem, tab_obb, tab_pose, tab_dept,
    ) = st.tabs([
        "🎯 Auto BBox", "🎯 Auto Segm", "🏷️ Classification", "✂️ Semantic / Segm",
        "🔄 OBB Tagger", "🦴 Pose Tagger", "📏 Depth Tagger",
    ])
    
    modules = [
        ("tab_auto_bbox", "bbox_tab", "Auto BBox"),
        ("tab_auto_segm", "segm_tab", "Auto Segm"),
        ("tab_cls", "cls_tab", "Classification"),
        ("tab_sem", "sem_tab", "Semantic"),
        ("tab_obb", "obb_tab", "OBB"),
        ("tab_pose", "pose_tab", "Pose"),
        ("tab_dept", "dept_tab", "Depth"),
    ]
    
    for tab_container, module_name, display_name in modules:
        with locals()[tab_container]:
            try:
                module = __import__(f"UI.tagger.{module_name}", fromlist=[module_name])
                getattr(module, f"{module_name}")()
            except ImportError as e:
                st.error(f"⚠️ Failed to load {display_name} tagger module. Check if `UI/tagger/{module_name}.py` exists.\nDetails: {e}")
            except Exception as e:
                st.error(f"Error in {display_name} tagger: {e}")


# ==============================================================================
# 4. CAPTIONS TAB LOGIC
# ==============================================================================

def render_captions_tab():
    st.subheader("📝 Image Tagging & Text Captions")
    sub_wd14, sub_clip_blip, sub_manual = st.tabs([
        "🏷️ WD14 Tagger",
        "💬 BLIP Captioneer | 🔍 CLIP Interrogator",
        "✍️ Manual Caption Editor"
    ])
    
    modules = [
        (sub_wd14, "wd14", "wd14_tab", "WD14"),
        (sub_clip_blip, "blip_clip_tab", "blip_clip_tab", "BLIP & CLIP"),
        (sub_manual, "florence2", "florence2_tab", "Manual Caption Editor"),
    ]
    
    for tab_container, module_name, func_name, display_name in modules:
        with tab_container:
            try:
                module = __import__(f"UI.captions.{module_name}", fromlist=[module_name])
                getattr(module, func_name)()
            except ImportError as e:
                st.error(f"⚠️ Failed to load {display_name} module. Check if `UI/captions/{module_name}.py` exists.\nDetails: {e}")
            except Exception as e:
                st.error(f"Error in {display_name} module: {e}")


# ==============================================================================
# 5. BENCHMARK TAB LOGIC
# ==============================================================================
def render_benchmark_tab():
    st.header("📊 Model & Hardware Benchmarks")
    tab_bbox, tab_hardware = st.tabs([
        "🎯 BBox & Inference Benchmark",
        "💻 System & Hardware Benchmark"
    ])
    
    with tab_bbox:
        st.subheader("Object Detection Benchmark Suite")
        try:
            from UI.benchmark import bbox_gui
            bbox_gui.render_bbox_benchmark_ui()
        except ImportError as e:
            st.error(f"⚠️ Failed to load BBox benchmark GUI. Check if `UI/benchmark/bbox_gui.py` exists.\nDetails: {e}")
        except Exception as e:
            st.error(f"Error rendering BBox test GUI: {e}")
            
    with tab_hardware:
        st.subheader("Hardware Throughput & VRAM Profiler")
        st.info("Profile inference FPS, memory usage, and execution latency across CPU, CUDA, and ROCm backends.")


# ==============================================================================
# 6. TRAINER TAB LOGIC (Updated with your sub-tabs)
# ==============================================================================
def render_trainer_tab():
    st.header("🏋️ Model Trainer Workspace")

    tab_bbox, tab_segm, tab_lora, tab_embed, tab_train = st.tabs([
        "🎯 BBox Trainer",
        "✂️ Segm Trainer",
        "🎨 LoRA Trainer",
        "🧠 Textual Inversion & Hypernetworks",
        "🏋️ Train"
    ])

    with tab_bbox:
        try:
            from UI.trainer import bbox_trainer_tab
            bbox_trainer_tab.render_bbox_trainer_ui()
        except Exception as e:
            st.error(f"Failed to load BBox Trainer: {e}")

    with tab_segm:
        try:
            from UI.trainer import segm_trainer_tab
            segm_trainer_tab.render_segm_trainer_ui()
        except Exception as e:
            st.error(f"Failed to load Segm Trainer: {e}")

    with tab_lora:
        try:
            from UI.trainer import lora_tab
            lora_tab.render_lora_trainer_ui()
        except Exception as e:
            st.error(f"Failed to load LoRA Trainer: {e}")

    with tab_embed:
        col1, col2 = st.columns(2)
        with col1:
            try:
                from UI.trainer import textual_inversion
                textual_inversion.render_textual_inversion_ui()
            except Exception as e:
                st.error(f"Failed to load Textual Inversion: {e}")
        with col2:
            try:
                from UI.trainer import hypernetwork_tab
                hypernetwork_tab.render_hypernetwork_ui()
            except Exception as e:
                st.error(f"Failed to load Hypernetwork: {e}")

    with tab_train:
        try:
            from UI.trainer import yolo_complete_tab
            yolo_complete_tab.trainer_tab()
        except Exception as e:
            st.error(f"Failed to load YOLO Trainer: {e}")


# ==============================================================================
# MAIN APP ROUTER
# ==============================================================================
def main():
    st.title("🏋️ AI-Fitness")

    tab_datasets, tab_tagger, tab_captions, tab_trainer, tab_benchmark, tab_settings = st.tabs([
        "📁 Datasets", 
        "🏷️ Tagger",
        "📝 Captions",
        "🏋️ Trainer", 
        "📊 Benchmark",
        "⚙️ Settings"
    ])

    with tab_datasets:
        render_datasets_tab()

    with tab_tagger:
        render_tagger_tab()

    with tab_captions:
        render_captions_tab()

    with tab_trainer:
        render_trainer_tab()

    with tab_benchmark:
        render_benchmark_tab()

    with tab_settings:
        render_settings_tab()

if __name__ == "__main__":
    main()
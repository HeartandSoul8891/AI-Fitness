import streamlit as st
import os
from pathlib import Path

# Import backend functions
# Adjust import path if this file is inside a 'tabs' or 'scripts' folder

from scripts.settings_script import load_settings, save_settings, sync_paths_from_root

def render_settings_tab():
    st.title("⚙️ Settings & Configuration")
    st.write("Configure the base directories for your datasets, training outputs, and models.")
    
    # Load current settings into session state
    if "app_settings" not in st.session_state:
        st.session_state.app_settings = load_settings()
        
    settings = st.session_state.app_settings
    
    # ---------------------------------------------------------
    # 1. ROOT WORKSPACE
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📂 Root Workspace Directory")
    st.caption("This is the main folder for your project. All other directories will be created inside it by default.")
    
    root_folder = st.text_input(
        "Root Folder Path",
        value=settings.get("root_folder", ""),
        key="root_folder_input",
        help="Select the main directory for your YOLO/Vision project."
    )
    
    if st.button("🔄 Auto-Generate Paths from Root", use_container_width=True):
        if root_folder and os.path.exists(root_folder):
            new_paths = sync_paths_from_root(root_folder)
            st.session_state.app_settings.update(new_paths)
            st.toast("Paths updated from root!", icon="🔄")
            st.rerun()
        else:
            st.warning("Please enter a valid, existing root folder path.")

    # ---------------------------------------------------------
    # 2. WORKSPACE DIRECTORIES
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🗂️ Workspace Directories")
    
    col1, col2 = st.columns(2)
    with col1:
        datasets_folder = st.text_input(
            "Datasets Folder",
            value=settings.get("datasets_folder", ""),
            key="datasets_folder_input",
            help="Where raw images, annotations, and tagged datasets are stored."
        )
        st.session_state.app_settings["datasets_folder"] = datasets_folder
        
        training_folder = st.text_input(
            "Training Folder",
            value=settings.get("training_folder", ""),
            key="training_folder_input",
            help="Where YOLO training runs, logs, and weights are saved."
        )
        st.session_state.app_settings["training_folder"] = training_folder

    with col2:
        output_folder = st.text_input(
            "Output Folder",
            value=settings.get("output_folder", ""),
            key="output_folder_input",
            help="General output folder for exports, predictions, etc."
        )
        st.session_state.app_settings["output_folder"] = output_folder

    # ---------------------------------------------------------
    # 3. MODEL DIRECTORIES
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🧠 Model Directories")
    st.caption("Directories for storing downloaded AI models (Ultralytics YOLO, Vision-Language models, etc.).")
    
    col3, col4 = st.columns(2)
    with col3:
        models_folder = st.text_input(
            "General Models Folder",
            value=settings.get("models_folder", ""),
            key="models_folder_input"
        )
        st.session_state.app_settings["models_folder"] = models_folder
        
        ultralytics_folder = st.text_input(
            "Ultralytics (YOLO) Models",
            value=settings.get("ultralytics_models_folder", ""),
            key="ultralytics_folder_input",
            help="Folder for YOLO .pt weights."
        )
        st.session_state.app_settings["ultralytics_models_folder"] = ultralytics_folder

    with col4:
        vision_folder = st.text_input(
            "Vision Models (BLIP, CLIP, Florence, WD14)",
            value=settings.get("vision_models_folder", ""),
            key="vision_folder_input",
            help="Folder for HuggingFace vision models and WD14 ONNX files."
        )
        st.session_state.app_settings["vision_models_folder"] = vision_folder

    # ---------------------------------------------------------
    # ACTIONS
    # ---------------------------------------------------------
    st.markdown("---")
    
    col_save, col_create = st.columns([1, 1])
    with col_save:
        if st.button("💾 Save Settings", type="primary", use_container_width=True):
            success, msg = save_settings(st.session_state.app_settings)
            if success:
                st.success(msg)
                st.toast("Settings saved to JSON!", icon="💾")
            else:
                st.error(msg)
                
    with col_create:
        if st.button("📁 Create Missing Folders", use_container_width=True):
            created = []
            for key, path in st.session_state.app_settings.items():
                if key != "root_folder" and path and not os.path.exists(path):
                    try:
                        os.makedirs(path, exist_ok=True)
                        created.append(os.path.basename(path))
                    except Exception as e:
                        st.error(f"Failed to create {path}: {e}")
            if created:
                st.success(f"Created folders: {', '.join(created)}")
            else:
                st.info("All folders already exist.")

if __name__ == "__main__":
    render_settings_tab()
import os
import json
import streamlit as st
from pathlib import Path

# ==========================================
# BACKEND IMPORTS
# ==========================================
# Adjust this import path based on your actual folder structure
try:
    from scripts.dataset_script import (
        get_dataset_folders,
        get_unique_labels_from_source,
        create_dataset_structure,
        cleanup_empty_jsons_and_media,
        move_and_split_files,
        convert_dataset_jsons,
        generate_yaml
    )
except ImportError:
    # Fallback if running directly in the same folder
    from scripts.dataset_script import (
        get_dataset_folders,
        get_unique_labels_from_source,
        create_dataset_structure,
        cleanup_empty_jsons_and_media,
        move_and_split_files,
        convert_dataset_jsons,
        generate_yaml
    )

def dataset_preparation_gui():
    st.title("📂 Dataset Preparation for YOLO")
    st.write("Prepare, structure, and convert annotated datasets into YOLO-ready training sets.")
    st.markdown("---")

    # Initialize session state
    if "class_mapping" not in st.session_state:
        st.session_state.class_mapping = {}
    if "dataset_path" not in st.session_state:
        st.session_state.dataset_path = None

    # ------------------------------------------------------------------
    # 1. Select Source & Auto-Detect Classes
    # ------------------------------------------------------------------
    st.subheader("1. Source Dataset & Class Mapping")
    
    col_src1, col_src2 = st.columns([2, 1])
    with col_src1:
        source_dir = st.text_input(
            "Source Tagged Dataset Folder (Absolute Path)",
            value="./datasets/raw_annotations",
            key="dataset_source_path"
        )
    with col_src2:
        task_type = st.selectbox(
            "YOLO Task Type",
            options=["bbox", "segm", "obb", "pose", "cls"],
            help="Select the annotation format. 'bbox' is standard bounding boxes.",
            key="dataset_task_type"
        )

    col_scan, col_clean = st.columns(2)
    with col_scan:
        if st.button("🔍 Scan & Auto-Detect Classes", use_container_width=True):
            if not os.path.exists(source_dir):
                st.error("Source directory does not exist.")
            else:
                with st.spinner("Scanning JSON files for unique labels..."):
                    detected_labels = get_unique_labels_from_source(source_dir)
                    if detected_labels:
                        st.session_state.class_mapping = detected_labels
                        st.success(f"Found {len(detected_labels)} unique classes!")
                    else:
                        st.warning("No labels found. Ensure your JSONs contain 'shapes', 'detections', or 'keypoints'.")
                        
    with col_clean:
        if st.button("🧹 Cleanup Empty/Invalid JSONs", use_container_width=True):
            if not os.path.exists(source_dir):
                st.error("Source directory does not exist.")
            else:
                with st.spinner("Purging empty annotations..."):
                    success, msg = cleanup_empty_jsons_and_media(source_dir)
                    if success: st.success(msg)
                    else: st.warning(msg)

    # Editable Class Mapping
    st.markdown("**Detected Class Mapping (Editable JSON):**")
    mapping_json_str = st.text_area(
        "Class Mapping",
        value=json.dumps(st.session_state.class_mapping, indent=2),
        height=150,
        key="class_mapping_editor",
        help="You can manually edit the class IDs or add new classes here."
    )
    
    if st.button("💾 Update Mapping from Text", key="update_mapping_btn"):
        try:
            st.session_state.class_mapping = json.loads(mapping_json_str)
            st.toast("Class mapping updated!", icon="💾")
        except json.JSONDecodeError:
            st.error("Invalid JSON format. Please check your syntax.")

    st.markdown("---")

    # ------------------------------------------------------------------
    # 2. Create Structure & Split
    # ------------------------------------------------------------------
    st.subheader("2. Structure, Split & Convert")
    
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        training_root = st.text_input("Training Root Directory", value="./datasets/training", key="train_root")
    with col_cfg2:
        dataset_name = st.text_input("Output Dataset Name", value="yolo_project_v1", key="dataset_name")
    with col_cfg3:
        split_ratio = st.slider("Train Split Ratio", min_value=0.50, max_value=0.95, value=0.80, step=0.05, key="split_ratio")

    # Pipeline Execution Buttons
    st.markdown("#### ⚙️ Execution Pipeline")
    
    # Step A: Create Structure
    if st.button("1️⃣ Create Folder Structure", use_container_width=True, type="primary"):
        if not st.session_state.class_mapping:
            st.error("Please detect or define a class mapping first.")
        else:
            with st.spinner("Creating YOLO directory structure..."):
                st.session_state.dataset_path = create_dataset_structure(training_root, dataset_name, task_type)
                st.success(f"Structure created at: `{st.session_state.dataset_path}`")

    # Step B: Move & Split
    if st.button("2️⃣ Move & Split Dataset (Train/Val)", use_container_width=True):
        if not st.session_state.dataset_path:
            st.error("Please create the folder structure first.")
        else:
            with st.spinner("Copying and splitting files..."):
                success, msg = move_and_split_files(source_dir, st.session_state.dataset_path, split_ratio)
                if success: st.success(msg)
                else: st.error(msg)

    # Step C: Convert JSON to TXT
    if st.button("3️⃣ Convert JSONs to YOLO .txt Labels", use_container_width=True):
        if not st.session_state.dataset_path:
            st.error("Please create the folder structure first.")
        else:
            with st.spinner("Converting annotations to YOLO format..."):
                success, msg = convert_dataset_jsons(st.session_state.dataset_path, st.session_state.class_mapping, task_type)
                if success: st.success(msg)
                else: st.error(msg)

    st.markdown("---")

    # ------------------------------------------------------------------
    # 3. Generate YAML & Finalize
    # ------------------------------------------------------------------
    st.subheader("3. Finalize Configuration")
    
    kpt_shape = None
    if task_type == "pose":
        kpt_shape = st.text_input("Keypoint Shape (e.g., [17, 3])", value="[17, 3]", help="Format: [num_kpts, dims]")

    if st.button("✨ Generate data.yaml", use_container_width=True, type="primary"):
        if not st.session_state.dataset_path:
            st.error("Please complete the previous steps first.")
        else:
            with st.spinner("Generating data.yaml..."):
                success, msg = generate_yaml(
                    st.session_state.dataset_path, 
                    st.session_state.class_mapping, 
                    task_type, 
                    kpt_shape=kpt_shape
                )
                if success: 
                    st.success(msg)
                    st.balloons()
                    st.info(f"Your dataset is ready! You can now point Ultralytics YOLO to: `{os.path.join(st.session_state.dataset_path, 'data.yaml')}`")
                else: 
                    st.error(msg)

if __name__ == "__main__":
    dataset_preparation_gui()
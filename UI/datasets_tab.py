import json
import os
import streamlit as st
from pathlib import Path
from scripts.settings.settings_script import load_settings
from scripts.datasets.dataset_preparation_script import (
    get_dataset_folders,
    get_unique_labels_from_source,
    create_dataset_structure,
    move_and_split_files,
    convert_dataset_jsons,
    generate_yaml,
    cleanup_empty_jsons_and_media
)

def datasets_tab():
    st.title("📦 Multi-Task Dataset Preparation")
    st.write("Prepare, structure, and convert tagged datasets into task-ready YOLO formats (BBox, Segmentation, OBB, Pose, Classification).")

    settings = load_settings()
    datasets_root = settings.get("datasets_folder") or "datasets"
    training_root = settings.get("training_folder") or "training"
    
    os.makedirs(datasets_root, exist_ok=True)
    os.makedirs(training_root, exist_ok=True)

    st.info(f"**Tagged Datasets Source Root:** `{datasets_root}`\n\n**Training Output Root:** `{training_root}`")
    st.markdown("---")

    # Task & Target Selector
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

    # Source Dataset Cleanup
    st.markdown("### 🧹 Source Dataset Hygiene")
    st.caption("Scan source folder to purge empty JSONs and matching unused images/visualizers.")
    if st.button("Purge Empty JSONs & Dead Media", type="secondary", key="ds_purge_btn"):
        success, msg = cleanup_empty_jsons_and_media(source_dir)
        if success:
            st.success(msg)
        else:
            st.error(msg)

    st.markdown("---")
    
    # Step 2: Create Folder Structure
    st.subheader("2. Create Training Folder Structure")
    st.write(f"Creates YOLO dataset tree inside `{dataset_path}`.")
    if st.button("Create Folder Structure", type="primary", key="ds_create_struct_btn"):
        try:
            path = create_dataset_structure(training_root, output_dataset_name, task_type)
            st.success(f"Folder structure created successfully at: `{path}`")
        except Exception as e:
            st.error(f"Error creating folder structure: {e}")

    st.markdown("---")
    
    # Step 3: Move & Split Dataset
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
    
    # Step 4: Class Mapping & Label Conversion
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
    
    # Step 5: Generate data.yaml
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

# Module Entry Point Aliases for main app router
def main():
    datasets_tab()

def app():
    datasets_tab()

def show():
    datasets_tab()

def render():
    datasets_tab()

def dataset_preparation():
    datasets_tab()

render_tab = datasets_tab

if __name__ == "__main__":
    datasets_tab()
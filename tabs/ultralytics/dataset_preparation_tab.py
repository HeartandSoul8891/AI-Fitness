import os
import json
import streamlit as st
from scripts.settings_script import load_settings
from scripts.ultralytics.dataset_preparation_script import (
    get_dataset_folders,
    get_unique_labels_from_source,
    create_dataset_structure,
    move_and_split_files,
    convert_dataset_jsons,
    generate_yaml,
    cleanup_empty_jsons_and_media  # <--- Make sure this is imported here
)

def dataset_preparation():
    st.title("Dataset Preparation")
    st.write("Prepare, structure, and convert tagged datasets into YOLO-ready training sets.")

    # Load settings with explicit fallbacks
    settings = load_settings()
    datasets_root = settings.get("datasets_folder") or "datasets"
    
    # Ensure datasets_root directory exists so get_dataset_folders doesn't break
    os.makedirs(datasets_root, exist_ok=True)
    
    # Determine base app root
    app_root = os.path.dirname(datasets_root) if os.path.dirname(datasets_root) else "."
    training_root = os.path.join(app_root, "training")

    st.info(f"**Tagged Datasets Source Root:** `{datasets_root}`\n\n**Training Output Root:** `{training_root}`")

    st.markdown("---")

    # 1. Select the tagged dataset folder
    st.subheader("1. Select Source Tagged Dataset")
    tagged_folders = get_dataset_folders(datasets_root) or ["."]
    selected_subfolder = st.selectbox("Select Tagged Dataset Folder", tagged_folders) or "."
    
    source_dir = datasets_root if selected_subfolder == "." else os.path.join(datasets_root, selected_subfolder)

    # Output training dataset folder name
    output_dataset_name = st.text_input("Output Training Dataset Folder Name", value=selected_subfolder if selected_subfolder != "." else "prepared_dataset")
    dataset_path = os.path.join(training_root, output_dataset_name)

    # --- ADDED: Source Dataset Hygiene Check Button ---
    st.markdown("### 🧹 Source Dataset Hygiene Check")
    st.write("Scan the selected source folder and completely purge empty JSONs *alongside* their matching original images and `_yolo_edit` visualizer backups.")
    if st.button("Purge Empty JSONs & Dead Image Pairs", type="secondary"):
        success, msg = cleanup_empty_jsons_and_media(source_dir)
        if success:
            st.success(msg)
        else:
            st.error(msg)
    # --------------------------------------------------

    st.markdown("---")
    
    # Step 2: Create Folder Structure
    st.subheader("2. Create Training Folder Structure")
    st.write("Sets up the required YOLO directory tree (`train/images`, `train/json`, `train/labels`, `val/images`, `val/json`, `val/labels`) inside the app's `training` folder.")
    if st.button("Create Folder Structure", type="primary"):
        try:
            path = create_dataset_structure(training_root, output_dataset_name)
            st.success(f"Folder structure created successfully at: `{path}`")
        except Exception as e:
            st.error(f"Error creating folder structure: {e}")

    st.markdown("---")
    
    # Step 3: Move & Split Dataset
    st.subheader("3. Move & Split Dataset")
    st.write(f"Copies images and `.json` annotations from `{source_dir}` and splits them into train/val sets (excluding `_yolo_edit` files).")
    
    split_ratio = st.slider("Train Split Ratio", min_value=0.5, max_value=0.95, value=0.8, step=0.05)

    if st.button("Move & Split Dataset"):
        success, msg = move_and_split_files(source_dir, dataset_path, split_ratio)
        if success:
            st.success(msg)
        else:
            st.error(msg)

    st.markdown("---")
    
    # Step 4: Convert JSONs to YOLO TXT Labels
    st.subheader("4. Convert JSON to .txt Labels")
    st.write("Automatically discovered classes from your `.json` files are populated below. Convert annotation bounding boxes into normalized YOLO `.txt` label files.")

    # Auto-discover labels from the source folder to prefill the text box accurately
    discovered_labels = get_unique_labels_from_source(source_dir)
    default_mapping_str = json.dumps(discovered_labels if discovered_labels else {"DEFAULT": 0}, indent=4)

    class_mapping_input = st.text_area("Class Mapping (JSON Dictionary)", value=default_mapping_str, height=130)

    if st.button("Convert JSONs to YOLO .txt"):
        try:
            class_mapping = json.loads(class_mapping_input)
            success, msg = convert_dataset_jsons(dataset_path, class_mapping)
            if success:
                st.success(msg)
            else:
                st.error(msg)
        except json.JSONDecodeError:
            st.error("Invalid JSON format in Class Mapping box. Please check your braces and quotes.")
        except Exception as e:
            st.error(f"Error during conversion: {e}")

    st.markdown("---")
    
    # Step 5: Generate data.yaml
    st.subheader("5. Generate data.yaml")
    st.write("Generates the final `data.yaml` configuration file required to initialize training in Ultralytics.")

    if st.button("Generate data.yaml", type="primary"):
        try:
            class_mapping = json.loads(class_mapping_input)
            success, msg = generate_yaml(dataset_path, class_mapping)
            if success:
                st.success(msg)
            else:
                st.error(msg)
        except json.JSONDecodeError:
            st.error("Invalid JSON format in Class Mapping.")
        except Exception as e:
            st.error(f"Error generating YAML: {e}")
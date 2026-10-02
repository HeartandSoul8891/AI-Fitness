import os
import streamlit as st

from scripts.tagger.bbox_tagger_script import (
    run_auto_tagger,
    clean_existing_dataset_json,
)

from scripts.settings.settings_script import load_settings

def bbox_tab():
    st.title("Auto Tagger & Bounding Box Visualizer")
    st.write(
        "Detect objects using a YOLO model, map model classes to your own custom"
        " tag names, filter by size, and run safety checks."
    )

    saved_settings = load_settings()
    global_datasets_dir = st.session_state.get(
    "settings_datasets_folder", 
    saved_settings.get("datasets_folder", "./datasets")
    )

    current_settings = load_settings()

    default_model_dir = "models"
    default_datasets_dir = "datasets"

    # Get values from session state only if they are valid, non-empty strings
    ss_model_dir = st.session_state.get("settings_ultralytics_bbox_folder")
    ss_dataset_dir = st.session_state.get("settings_datasets_folder")

    model_dir = (
        (ss_model_dir if ss_model_dir and ss_model_dir.strip() else None)
        or current_settings.get("ultralytics_bbox_folder")
        or current_settings.get("settings_ultralytics_bbox_folder")
        or current_settings.get("custom_model_folder")
        or current_settings.get("model_folder")
        or default_model_dir
    )

    datasets_dir = (
        (ss_dataset_dir if ss_dataset_dir and ss_dataset_dir.strip() else None)
        or current_settings.get("datasets_folder")
        or current_settings.get("settings_datasets_folder")
        or current_settings.get("custom_datasets_folder")
        or default_datasets_dir
    )

    # 1. Select Dataset Folder
    st.subheader("1. Select Dataset Folder")
    dataset_options = []
    if os.path.exists(datasets_dir):
        dataset_options = [datasets_dir] + [
            os.path.normpath(os.path.join(datasets_dir, d))
            for d in os.listdir(datasets_dir)
            if os.path.isdir(os.path.join(datasets_dir, d))
        ]
    else:
        dataset_options = [datasets_dir]

    selected_dataset = st.selectbox(
        "Choose Dataset Directory", dataset_options, key="bbox_tagger_dataset_dir"
    )

    # 2. Select YOLO Model
    st.subheader("2. Select YOLO Model")
    model_options = []
    if os.path.exists(model_dir):
        for root, _, files in os.walk(model_dir):
            for f in files:
                if f.lower().endswith((".pt", ".pth", ".onnx", ".engine", ".safetensors")):
                    model_options.append(os.path.normpath(os.path.join(root, f)))

    if model_options:
        selected_model = st.selectbox(
            f"Choose YOLO Model File (Directory: {model_dir})",
            options=model_options,
            key="bbox_tagger_model_file",
        )
    else:
        st.warning(f"No YOLO model files found in: `{model_dir}`")
        selected_model = st.text_input(
            "Manually enter model file path",
            value="",
            key="bbox_tagger_manual_model_file",
        )

    # 3. Model Tag Mapping Option
    st.subheader("3. YOLO Model Tag Mapping")
    col1, col2 = st.columns(2)
    with col1:
        target_model_class = st.text_input(
            "Model Tag (What model knows)",
            value="",
            placeholder="e.g., flowers",
            key="bbox_tagger_target_class",
        )
    with col2:
        new_tag = st.text_input(
            "New Tag (What to save as)",
            value="",
            placeholder="e.g., rose",
            key="bbox_tagger_new_tag",
        )
    st.caption(
        "Example: If the model detects 'flowers', it will map it and trigger the"
        " tag 'rose'. Leave 'Model Tag' blank to capture all model classes."
    )

    # 4. Additional Global Default Tag
    st.subheader("4. Additional Global Default Tag")
    default_tag = st.text_input(
        "General Default Tag",
        value="",
        placeholder="e.g., project_name",
        key="bbox_tagger_default_tag",
    )
    st.caption(
        "This tag will be added to every processed image regardless of detections."
    )

    # 5. Detection, Sizing Filters & Benchmark Settings
    st.subheader("5. Detection, Bounding Box Sizing & Benchmark Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        key="bbox_tagger_conf_slider",
    )

    col_size1, col_size2 = st.columns(2)
    with col_size1:
        max_box_dimension = st.slider(
            "Max Bounding Box Size Ratio",
            min_value=0.5,
            max_value=1.0,
            value=0.95,
            step=0.01,
            help="Filters out oversized or full-screen bounding boxes exceeding this threshold ratio.",
            key="bbox_tagger_max_box_slider",
        )
    with col_size2:
        min_box_dimension = st.slider(
            "Min Bounding Box Size Ratio",
            min_value=0.0,
            max_value=0.2,
            value=0.01,
            step=0.01,
            help="Filters out micro-noise boxes below this threshold ratio.",
            key="bbox_tagger_min_box_slider",
        )

    is_benchmark = st.checkbox(
        "Enable Benchmark Mode (Preview without saving permanent files)",
        key="bbox_tagger_is_benchmark_cb",
    )
    benchmark_limit = 5
    if is_benchmark:
        benchmark_limit = st.number_input(
            "Number of images to preview in benchmark",
            min_value=1,
            max_value=48,
            value=12,
            key="bbox_tagger_benchmark_limit_num",
        )

    # 6. Trigger Action Buttons
    st.markdown("---")
    col_run, col_clean = st.columns(2)

    with col_run:
        run_btn = st.button(
            "Run Auto-Tagger & Sanity Check Outputs",
            type="primary",
            key="bbox_tagger_run_btn",
        )
    with col_clean:
        clean_btn = st.button(
            "Clean-up Existing JSON Duplicates",
            key="bbox_tagger_clean_btn",
        )

    if run_btn:
        if not selected_dataset:
            st.error("Please select a valid dataset folder.")
        elif not selected_model:
            st.error("Please select or specify a valid YOLO model.")
        else:
            action_label = (
                "Running Benchmark Preview with Sanity Check..."
                if is_benchmark
                else "Running YOLO detection, size filtering, and sanity check..."
            )
            with st.spinner(action_label):
                result = run_auto_tagger(
                    dataset_path=selected_dataset,
                    model_path=selected_model,
                    default_tag=default_tag,
                    target_model_class=target_model_class,
                    new_tag=new_tag,
                    conf_threshold=conf_threshold,
                    max_box_dimension=max_box_dimension,
                    min_box_dimension=min_box_dimension,
                    is_benchmark=is_benchmark,
                    benchmark_limit=int(benchmark_limit),
                )
                if result["success"]:
                    st.success(result["message"])

                    if is_benchmark and "preview_items" in result:
                        st.markdown("### 🔍 Benchmark Visual Preview (Sanitized)")
                        st.info(
                            f"Displaying {len(result['preview_items'])} test images with"
                            " double-checked clean bounding boxes. (No files were written"
                            " to disk)."
                        )

                        for item in result["preview_items"]:
                            st.markdown(
                                f"**Image:** `{os.path.basename(item['image_path'])}`"
                            )
                            st.json(item["detections_summary"])
                            st.image(
                                item["img_rgb"], channels="RGB", use_container_width=True
                            )
                            st.markdown("---")
                else:
                    st.error(result["message"])

    if clean_btn:
        if not selected_dataset:
            st.error("Please select a valid dataset folder.")
        else:
            with st.spinner(
                "Scanning and cleaning duplicate tags/boxes in existing JSONs..."
            ):
                clean_result = clean_existing_dataset_json(selected_dataset)
                if clean_result["success"]:
                    st.success(clean_result["message"])
                else:
                    st.error(clean_result["message"])
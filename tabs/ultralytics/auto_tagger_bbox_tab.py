import os
import streamlit as str_lit
from scripts.settings_script import load_settings
from scripts.ultralytics.auto_tagger_script import (
    run_auto_tagger,
    clean_existing_dataset_json,
)


def auto_tagger_tab():
    str_lit.title("Auto Tagger & Bounding Box Visualizer")
    str_lit.write(
        "Detect objects using a YOLO model, map model classes to your own custom"
        " tag names, filter by size, and run safety checks."
    )

    current_settings = load_settings()

    default_model_dir = "models"
    default_datasets_dir = "datasets"

    model_dir = (
        current_settings.get("custom_model_folder")
        or current_settings.get("ultralytics_bbox_folder")
        or current_settings.get("model_folder")
        or default_model_dir
    )

    datasets_dir = (
        current_settings.get("custom_datasets_folder")
        or current_settings.get("datasets_folder")
        or default_datasets_dir
    )

    # 1. Select Dataset Folder
    str_lit.subheader("1. Select Dataset Folder")
    dataset_options = []
    if os.path.exists(datasets_dir):
        dataset_options = [datasets_dir] + [
            os.path.join(datasets_dir, d)
            for d in os.listdir(datasets_dir)
            if os.path.isdir(os.path.join(datasets_dir, d))
        ]
    else:
        dataset_options = [datasets_dir]

    selected_dataset = str_lit.selectbox(
        "Choose Dataset Directory", dataset_options, key="bbox_tagger_dataset_dir"
    )

    # 2. Select YOLO Model
    str_lit.subheader("2. Select YOLO Model")
    model_options = []
    if os.path.exists(model_dir):
        model_options = [
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir)
            if f.endswith((".pt", ".pth"))
        ]

    if model_options:
        selected_model = str_lit.selectbox(
            "Choose YOLO Model File", model_options, key="bbox_tagger_model_file"
        )
    else:
        str_lit.warning(
            f"No YOLO model files (.pt/.pth) found in '{model_dir}'. Please add"
            " weights or check your settings."
        )
        selected_model = str_lit.text_input(
            "Or enter model path manually", "", key="bbox_tagger_manual_model"
        )

    # 3. Model Tag Mapping Option
    str_lit.subheader("3. YOLO Model Tag Mapping")
    col1, col2 = str_lit.columns(2)
    with col1:
        target_model_class = str_lit.text_input(
            "Model Tag (What model knows)",
            value="",
            placeholder="e.g., flowers",
            key="bbox_tagger_target_class",
        )
    with col2:
        new_tag = str_lit.text_input(
            "New Tag (What to save as)",
            value="",
            placeholder="e.g., rose",
            key="bbox_tagger_new_tag",
        )
    str_lit.caption(
        "Example: If the model detects 'flowers', it will map it and trigger the"
        " tag 'rose'. Leave 'Model Tag' blank to capture all model classes."
    )

    # 4. Additional Global Default Tag
    str_lit.subheader("4. Additional Global Default Tag")
    default_tag = str_lit.text_input(
        "General Default Tag",
        value="",
        placeholder="e.g., project_name",
        key="bbox_tagger_default_tag",
    )
    str_lit.caption(
        "This tag will be added to every processed image regardless of detections."
    )

    # 5. Detection, Sizing Filters & Benchmark Settings
    str_lit.subheader("5. Detection, Bounding Box Sizing & Benchmark Settings")
    conf_threshold = str_lit.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        key="bbox_tagger_conf_slider",
    )

    col_size1, col_size2 = str_lit.columns(2)
    with col_size1:
        max_box_dimension = str_lit.slider(
            "Max Bounding Box Size Ratio",
            min_value=0.5,
            max_value=1.0,
            value=0.95,
            step=0.01,
            help="Filters out oversized or full-screen bounding boxes exceeding this threshold ratio.",
            key="bbox_tagger_max_box_slider",
        )
    with col_size2:
        min_box_dimension = str_lit.slider(
            "Min Bounding Box Size Ratio",
            min_value=0.0,
            max_value=0.2,
            value=0.01,
            step=0.01,
            help="Filters out micro-noise boxes below this threshold ratio.",
            key="bbox_tagger_min_box_slider",
        )

    is_benchmark = str_lit.checkbox(
        "Enable Benchmark Mode (Preview without saving permanent files)",
        key="bbox_tagger_is_benchmark_cb",
    )
    benchmark_limit = 5
    if is_benchmark:
        benchmark_limit = str_lit.number_input(
            "Number of images to preview in benchmark",
            min_value=1,
            max_value=48,
            value=12,
            key="bbox_tagger_benchmark_limit_num",
        )

    # 6. Trigger Action Buttons
    str_lit.markdown("---")
    col_run, col_clean = str_lit.columns(2)

    with col_run:
        run_btn = str_lit.button(
            "Run Auto-Tagger & Sanity Check Outputs",
            type="primary",
            key="bbox_tagger_run_btn",
        )
    with col_clean:
        clean_btn = str_lit.button(
            "Clean-up Existing JSON Duplicates",
            key="bbox_tagger_clean_btn",
        )

    if run_btn:
        if not selected_dataset:
            str_lit.error("Please select a valid dataset folder.")
        elif not selected_model:
            str_lit.error("Please select or specify a valid YOLO model.")
        else:
            action_label = (
                "Running Benchmark Preview with Sanity Check..."
                if is_benchmark
                else "Running YOLO detection, size filtering, and sanity check..."
            )
            with str_lit.spinner(action_label):
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
                    str_lit.success(result["message"])

                    if is_benchmark and "preview_items" in result:
                        str_lit.markdown("### 🔍 Benchmark Visual Preview (Sanitized)")
                        str_lit.info(
                            f"Displaying {len(result['preview_items'])} test images with"
                            " double-checked clean bounding boxes. (No files were written"
                            " to disk)."
                        )

                        for item in result["preview_items"]:
                            str_lit.markdown(
                                f"**Image:** `{os.path.basename(item['image_path'])}`"
                            )
                            str_lit.json(item["detections_summary"])
                            str_lit.image(
                                item["img_rgb"], channels="RGB", use_container_width=True
                            )
                            str_lit.markdown("---")
                else:
                    str_lit.error(result["message"])

    if clean_btn:
        if not selected_dataset:
            str_lit.error("Please select a valid dataset folder.")
        else:
            with str_lit.spinner(
                "Scanning and cleaning duplicate tags/boxes in existing JSONs..."
            ):
                clean_result = clean_existing_dataset_json(selected_dataset)
                if clean_result["success"]:
                    str_lit.success(clean_result["message"])
                else:
                    str_lit.error(clean_result["message"])
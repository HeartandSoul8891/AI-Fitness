import os
import streamlit as str_lit
from scripts.settings.settings_script import load_settings
from scripts.tagger.bbox_tagger_script import (
    run_auto_tagger,
    clean_existing_dataset_json,
)


def segm_tab():
    str_lit.title("Auto-Tagger: BBox → Pseudo-Segmentation")
    str_lit.write(
        "Detect objects using a YOLO model and automatically convert detections into "
        "pseudo-segmentation polygons/masks for downstream segmentation re-tagging."
    )

    current_settings = load_settings()

    default_model_dir = "models"
    default_datasets_dir = "datasets"

    # Get values from session state only if they are valid, non-empty strings
    ss_model_dir = str_lit.session_state.get("settings_ultralytics_bbox_folder")
    ss_dataset_dir = str_lit.session_state.get("settings_datasets_folder")

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
    str_lit.subheader("1. Select Dataset Folder")
    dataset_options = []
    if os.path.exists(datasets_dir):
        dataset_options = [datasets_dir] + [
            os.path.normpath(os.path.join(datasets_dir, d))
            for d in os.listdir(datasets_dir)
            if os.path.isdir(os.path.join(datasets_dir, d))
        ]
    else:
        dataset_options = [datasets_dir]

    selected_dataset = str_lit.selectbox(
        "Choose Dataset Directory", dataset_options, key="segm_tagger_dataset_dir"
    )

    # 2. Select YOLO Model
    str_lit.subheader("2. Select YOLO Model")
    model_options = []
    if os.path.exists(model_dir):
        for root, _, files in os.walk(model_dir):
            for f in files:
                if f.lower().endswith((".pt", ".pth", ".onnx", ".engine", ".safetensors")):
                    model_options.append(os.path.normpath(os.path.join(root, f)))

    if model_options:
        selected_model = str_lit.selectbox(
            f"Choose YOLO Model File (Directory: {model_dir})",
            options=model_options,
            key="segm_tagger_model_file",
        )
    else:
        str_lit.warning(f"No YOLO model files found in: `{model_dir}`")
        selected_model = str_lit.text_input(
            "Manually enter model file path",
            value="",
            key="segm_tagger_manual_model_file",
        )

    
    # 3. Model Tag Mapping Option
    str_lit.subheader("3. YOLO Model Tag Mapping")
    col1, col2 = str_lit.columns(2)
    with col1:
        target_model_class = str_lit.text_input(
            "Model Tag (What model knows)",
            value="",
            placeholder="e.g., person",
            key="segm_tagger_target_class",
        )
    with col2:
        new_tag = str_lit.text_input(
            "New Tag (What to save as)",
            value="",
            placeholder="e.g., subject",
            key="segm_tagger_new_tag",
        )

    # 4. Global Default Tag
    str_lit.subheader("4. Additional Global Default Tag")
    default_tag = str_lit.text_input(
        "General Default Tag",
        value="",
        placeholder="e.g., project_v1",
        key="segm_tagger_default_tag",
    )

    # 5. Pseudo-Segmentation Settings
    str_lit.subheader("5. Pseudo-Segmentation Mask Controls")
    c_shape, c_pad, c_exp = str_lit.columns(3)
    with c_shape:
        mask_shape = str_lit.selectbox(
            "Pseudo-Mask Geometry",
            ["Fitted Ellipse", "GrabCut Foreground Refinement", "Rectangle Poly"],
            index=0,
            key="segm_tagger_mask_shape",
        )
    with c_pad:
        padding_pct = str_lit.slider(
            "Box Padding / Shrink (%)",
            -20,
            20,
            0,
            1,
            key="segm_tagger_padding_pct",
        )
    with c_exp:
        export_mode = str_lit.selectbox(
            "Output Format",
            ["Polygons (YOLO Seg .txt)", "Inpainting Binary Mask (.png)", "Both"],
            index=0,
            key="segm_tagger_export_mode",
        )

    # 6. Detection, Sizing Filters & Benchmarking
    str_lit.subheader("6. Thresholds & Benchmarking")
    conf_threshold = str_lit.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        key="segm_tagger_conf_slider",
    )

    col_size1, col_size2 = str_lit.columns(2)
    with col_size1:
        max_box_dimension = str_lit.slider(
            "Max Bounding Box Size Ratio",
            min_value=0.5,
            max_value=1.0,
            value=0.95,
            step=0.01,
            key="segm_tagger_max_box_slider",
        )
    with col_size2:
        min_box_dimension = str_lit.slider(
            "Min Bounding Box Size Ratio",
            min_value=0.0,
            max_value=0.2,
            value=0.01,
            step=0.01,
            key="segm_tagger_min_box_slider",
        )

    is_benchmark = str_lit.checkbox(
        "Enable Benchmark Mode (Preview without saving permanent files)",
        key="segm_tagger_is_benchmark_cb",
    )
    benchmark_limit = 5
    if is_benchmark:
        benchmark_limit = str_lit.number_input(
            "Number of images to preview in benchmark",
            min_value=1,
            max_value=48,
            value=12,
            key="segm_tagger_benchmark_limit_num",
        )

    # 7. Trigger Action Buttons
    str_lit.markdown("---")
    col_run, col_clean = str_lit.columns(2)

    with col_run:
        run_btn = str_lit.button(
            "🚀 Run BBox -> Pseudo-SEGM Auto-Tagger",
            type="primary",
            key="segm_tagger_run_btn",
        )
    with col_clean:
        clean_btn = str_lit.button(
            "Clean-up Existing JSON Duplicates",
            key="segm_tagger_clean_btn",
        )

    if run_btn:
        if not selected_dataset:
            str_lit.error("Please select a valid dataset folder.")
        elif not selected_model:
            str_lit.error("Please select or specify a valid YOLO model.")
        else:
            action_label = (
                "Running Benchmark Preview..."
                if is_benchmark
                else "Generating pseudo-segmentation masks and tagging..."
            )
            with str_lit.spinner(action_label):
                kwargs = {
                    "dataset_path": selected_dataset,
                    "model_path": selected_model,
                    "default_tag": default_tag,
                    "target_model_class": target_model_class,
                    "new_tag": new_tag,
                    "conf_threshold": conf_threshold,
                    "max_box_dimension": max_box_dimension,
                    "min_box_dimension": min_box_dimension,
                    "is_benchmark": is_benchmark,
                    "benchmark_limit": int(benchmark_limit),
                }

                import inspect
                sig = inspect.signature(run_auto_tagger)
                params = sig.parameters

                if "mask_shape" in params:
                    kwargs["mask_shape"] = mask_shape
                elif "pseudo_shape" in params:
                    kwargs["pseudo_shape"] = mask_shape

                if "padding_pct" in params:
                    kwargs["padding_pct"] = padding_pct
                elif "padding" in params:
                    kwargs["padding"] = padding_pct

                if "export_mode" in params:
                    kwargs["export_mode"] = export_mode
                elif "export_format" in params:
                    kwargs["export_format"] = export_mode

                result = run_auto_tagger(**kwargs)
                if result.get("success"):
                    str_lit.success(result.get("message", "Auto-tagging completed successfully."))
                else:
                    str_lit.error(result.get("message", "Auto-tagging failed."))

    if clean_btn:
        if not selected_dataset:
            str_lit.error("Please select a valid dataset folder.")
        else:
            with str_lit.spinner("Cleaning up existing annotations..."):
                clean_result = clean_existing_dataset_json(selected_dataset)
                if clean_result.get("success"):
                    str_lit.success(clean_result.get("message", "Cleaned successfully."))
                else:
                    str_lit.error(clean_result.get("message", "Clean failed."))
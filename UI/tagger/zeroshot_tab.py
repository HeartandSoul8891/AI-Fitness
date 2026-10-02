import os
import streamlit as st
from scripts.tagger.zeroshot_script import load_zeroshot_model, run_zeroshot_inference


def zeroshot_tagger_tab():
    st.title("🎯 Zero-Shot Open-Vocabulary Detection")
    st.write(
        "Detect arbitrary objects without re-training using natural language prompts powered by YOLO-World."
    )

    # --- Setup & Configuration Sidebar / Panel ---
    st.subheader("⚙️ Configuration & Prompts")

    col_model, col_source = st.columns(2)
    with col_model:
        model_name = st.selectbox(
            "YOLO-World Checkpoint",
            options=["yolov8s-worldv2.pt", "yolov8m-worldv2.pt", "yolov8l-worldv2.pt"],
            index=0,
            help="Select the model architecture size. Larger models provide higher accuracy.",
        )

    with col_source:
        source_input = st.text_input(
            "Image File or Directory Path",
            value="datasets/test_images",
            help="Provide a local path to an image file or a directory containing images.",
        )

    # Prompt Inputs
    raw_prompts = st.text_area(
        "Target Class Prompts (Comma-Separated)",
        value="person, dog, red car, coffee cup, laptop",
        help="Type class names or descriptive prompts separated by commas.",
    )

    # Sliders
    col_conf, col_iou = st.columns(2)
    with col_conf:
        conf_thresh = st.slider(
            "Confidence Threshold",
            min_value=0.01,
            max_value=1.0,
            value=0.15,
            step=0.01,
            help="Lower confidence threshold helps detect smaller/rare prompt descriptions.",
        )
    with col_iou:
        iou_thresh = st.slider(
            "IoU / NMS Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.45,
            step=0.05,
        )

    # Parse Prompts
    prompt_list = [p.strip() for p in raw_prompts.split(",") if p.strip()]

    # Trigger Inference Button
    if st.button("🚀 Run Zero-Shot Detection", type="primary", use_container_width=True):
        if not os.path.exists(source_input):
            st.error(f"Provided path does not exist: `{source_input}`")
            return

        if not prompt_list:
            st.warning("Please specify at least one text prompt category.")
            return

        with st.spinner("Loading YOLO-World model & running inference..."):
            try:
                # Load model instance
                model = load_zeroshot_model(model_name)

                # Execute detection
                annotated_images, df_results = run_zeroshot_inference(
                    model=model,
                    source_path=source_input,
                    prompts=prompt_list,
                    conf_threshold=conf_thresh,
                    iou_threshold=iou_thresh,
                )

                st.session_state["zeroshot_images"] = annotated_images
                st.session_state["zeroshot_df"] = df_results
                st.success("Zero-shot detection completed successfully!")

            except Exception as e:
                st.error(f"An error occurred during zero-shot execution: {e}")

    # --- Display Results ---
    if "zeroshot_images" in st.session_state:
        annotated_images = st.session_state["zeroshot_images"]
        df_results = st.session_state["zeroshot_df"]

        st.markdown("---")
        st.subheader("📊 Visual Results")

        if not annotated_images:
            st.info("No valid images found or processed.")
        else:
            # Display image selector if multiple images were processed
            img_keys = list(annotated_images.keys())
            selected_img_name = st.selectbox("Select Image Preview", options=img_keys)

            if selected_img_name:
                st.image(
                    annotated_images[selected_img_name],
                    caption=f"Detections for: {selected_img_name}",
                    use_container_width=True,
                )

        st.subheader("🔍 Detection Breakdown Table")
        if not df_results.empty:
            st.dataframe(df_results, use_container_width=True)
            
            # Summary Metrics
            st.markdown("##### Detection Count by Category")
            summary_counts = df_results["label"].value_counts().reset_index()
            summary_counts.columns = ["Class Label", "Count"]
            st.dataframe(summary_counts, use_container_width=True)
        else:
            st.warning("No detections met the current confidence threshold.")


if __name__ == "__main__":
    zeroshot_tagger_tab()
import streamlit as st
from PIL import Image
from scripts.benchmark.florence2_script import (
    FLORENCE2_TASKS,
    run_florence2_task,
)


def florence2_tab():
    st.title("🌸 Florence-2 Vision & Multimodal Suite")
    st.write(
        "Execute unified vision-language tasks including fine-grained captioning, object grounding, dense region detection, and OCR using Microsoft's Florence-2 architecture."
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        uploaded_img = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png", "webp"],
            key="florence2_uploader",
        )
        if uploaded_img:
            image = Image.open(uploaded_img).convert("RGB")
            st.image(image, caption="Input Image", use_container_width=True)

    with col_right:
        st.subheader("⚙️ Configuration")

        model_id = st.selectbox(
            "Florence-2 Model Checkpoint",
            options=[
                "microsoft/Florence-2-base",
                "microsoft/Florence-2-large",
                "microsoft/Florence-2-base-ft",
                "microsoft/Florence-2-large-ft",
            ],
            index=0,
            help="Select base pre-trained or downstream fine-tuned (ft) models.",
        )

        task_choice = st.selectbox(
            "Select Vision Task Prompt",
            options=list(FLORENCE2_TASKS.keys()),
            index=0,
        )

        # Additional text box for grounding tasks
        extra_text_input = ""
        if task_choice == "Caption to Phrase Grounding":
            extra_text_input = st.text_input(
                "Grounding Phrase",
                value="a dog running on the grass",
                help="Phrase to locate within the image.",
            )

        col_beams, col_tokens = st.columns(2)
        with col_beams:
            num_beams = st.slider("Beam Search", min_value=1, max_value=5, value=3)
        with col_tokens:
            max_tokens = st.slider("Max New Tokens", min_value=128, max_value=2048, value=1024)

        if st.button("🚀 Run Florence-2 Execution", type="primary", use_container_width=True):
            if not uploaded_img:
                st.warning("Please upload an image first.")
            else:
                with st.spinner("Processing image through Florence-2..."):
                    try:
                        raw_text, parsed_res, annotated_img = run_florence2_task(
                            image=image,
                            task_name=task_choice,
                            text_input=extra_text_input,
                            model_id=model_id,
                            max_new_tokens=max_tokens,
                            num_beams=num_beams,
                        )

                        st.session_state["f2_raw_text"] = raw_text
                        st.session_state["f2_parsed"] = parsed_res
                        st.session_state["f2_annotated"] = annotated_img
                        st.success("Execution Complete!")

                    except Exception as e:
                        st.error(f"Error executing Florence-2 inference: {e}")

    # ---------------------------------------------------------
    # Display Output Results
    # ---------------------------------------------------------
    if "f2_parsed" in st.session_state:
        st.markdown("---")
        st.subheader("📊 Execution Results")

        # Visual Bounding Box Results
        if st.session_state.get("f2_annotated") is not None:
            st.image(
                st.session_state["f2_annotated"],
                caption="Visual Detection & Grounding Output",
                use_container_width=True,
            )

        # Parsed Output
        st.markdown("**Parsed Structure / Text:**")
        parsed_data = st.session_state["f2_parsed"]

        if isinstance(parsed_data, dict):
            st.json(parsed_data)
        else:
            st.info(f"**{parsed_data}**")

        # Raw Model Output Inspection
        with st.expander("🔍 Raw Model Output Sequence"):
            st.code(st.session_state["f2_raw_text"], language="text")


if __name__ == "__main__":
    florence2_tab()
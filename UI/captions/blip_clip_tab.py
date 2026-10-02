import streamlit as st
from PIL import Image
from scripts.captions.blip_script import generate_blip_caption, answer_blip_vqa
from scripts.captions.clip_script import run_clip_zero_shot


def blip_clip_tab():
    st.title("👁️ Vision & Language Suite (BLIP & CLIP)")
    st.write(
        "Generate automated captions, perform Visual Question Answering (VQA), or execute zero-shot classification using BLIP and CLIP models."
    )

    tab_blip, tab_clip = st.tabs([
        "📝 BLIP: Caption & VQA",
        "🎯 CLIP: Zero-Shot Classification"
    ])

    # ---------------------------------------------------------
    # TAB 1: BLIP (Captioning & VQA)
    # ---------------------------------------------------------
    with tab_blip:
        st.subheader("BLIP (Bootstrapped Language-Image Pre-training)")

        blip_col_left, blip_col_right = st.columns([1, 1])

        with blip_col_left:
            uploaded_blip_img = st.file_uploader(
                "Upload Image for BLIP", type=["jpg", "jpeg", "png", "webp"], key="blip_uploader"
            )
            if uploaded_blip_img:
                image = Image.open(uploaded_blip_img).convert("RGB")
                st.image(image, caption="Target Image", use_container_width=True)

        with blip_col_right:
            task_type = st.radio("Select BLIP Task", ["Image Captioning", "Visual Question Answering (VQA)"])

            if task_type == "Image Captioning":
                cond_prompt = st.text_input("Prefix / Context Prompt (Optional)", value="a photo of")
                cap_model_id = st.selectbox(
                    "BLIP Caption Model",
                    ["Salesforce/blip-image-captioning-base", "Salesforce/blip-image-captioning-large"],
                )
                max_tokens = st.slider("Max New Tokens", 10, 100, 30)

                if st.button("✨ Generate Caption", type="primary", use_container_width=True):
                    if not uploaded_blip_img:
                        st.warning("Please upload an image first.")
                    else:
                        with st.spinner("Generating caption..."):
                            try:
                                caption = generate_blip_caption(
                                    image=image,
                                    model_id=cap_model_id,
                                    conditional_prompt=cond_prompt,
                                    max_new_tokens=max_tokens,
                                )
                                st.success("Caption Generated!")
                                st.subheader("Generated Output:")
                                st.info(f"**{caption}**")
                            except Exception as e:
                                st.error(f"Error executing BLIP captioning: {e}")

            else:  # VQA
                vqa_question = st.text_input("Ask a question about the image", value="What color is the object?")
                vqa_model_id = st.selectbox(
                    "BLIP VQA Model",
                    ["Salesforce/blip-vqa-base", "Salesforce/blip-vqa-capfilt-large"],
                )

                if st.button("❓ Answer Question", type="primary", use_container_width=True):
                    if not uploaded_blip_img:
                        st.warning("Please upload an image first.")
                    elif not vqa_question.strip():
                        st.warning("Please enter a question.")
                    else:
                        with st.spinner("Analyzing image and answering..."):
                            try:
                                answer = answer_blip_vqa(
                                    image=image,
                                    question=vqa_question,
                                    model_id=vqa_model_id,
                                )
                                st.success("Answer Ready!")
                                st.subheader("VQA Response:")
                                st.info(f"**{answer}**")
                            except Exception as e:
                                st.error(f"Error executing BLIP VQA: {e}")

    # ---------------------------------------------------------
    # TAB 2: CLIP (Zero-Shot Classification)
    # ---------------------------------------------------------
    with tab_clip:
        st.subheader("CLIP (Contrastive Language-Image Pre-training)")

        clip_col_left, clip_col_right = st.columns([1, 1])

        with clip_col_left:
            uploaded_clip_img = st.file_uploader(
                "Upload Image for CLIP", type=["jpg", "jpeg", "png", "webp"], key="clip_uploader"
            )
            if uploaded_clip_img:
                clip_image = Image.open(uploaded_clip_img).convert("RGB")
                st.image(clip_image, caption="Target Image", use_container_width=True)

        with clip_col_right:
            clip_model_id = st.selectbox(
                "CLIP Architecture Checkpoint",
                ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"],
            )

            candidate_text = st.text_area(
                "Candidate Labels (Comma-Separated)",
                value="a dog, a cat, a car, a landscape photo, a person working on a laptop",
                help="Enter candidate categories or descriptions separated by commas.",
            )

            if st.button("🚀 Evaluate Zero-Shot Probabilities", type="primary", use_container_width=True):
                if not uploaded_clip_img:
                    st.warning("Please upload an image first.")
                else:
                    labels = [l.strip() for l in candidate_text.split(",") if l.strip()]
                    if not labels:
                        st.warning("Please specify at least one valid candidate label.")
                    else:
                        with st.spinner("Calculating embeddings and similarity scores..."):
                            try:
                                df_res = run_clip_zero_shot(
                                    image=clip_image,
                                    candidate_labels=labels,
                                    model_id=clip_model_id,
                                )
                                st.success("Evaluation complete!")
                                st.subheader("📊 Class Match Predictions")
                                st.dataframe(df_res, use_container_width=True)

                                # Top prediction callout
                                top_pred = df_res.iloc[0]
                                st.metric(
                                    label="Top Prediction",
                                    value=top_pred["Label"],
                                    delta=f"{top_pred['Probability']:.2%} Confidence",
                                )
                            except Exception as e:
                                st.error(f"Error executing CLIP evaluation: {e}")


if __name__ == "__main__":
    blip_clip_tab()
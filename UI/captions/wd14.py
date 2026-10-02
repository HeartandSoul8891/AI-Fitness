import os
import streamlit as st
from PIL import Image
from scripts.captions.wd14_script import (
    AVAILABLE_WD14_MODELS,
    run_wd14_tagger,
    save_caption_file,
)


def wd14_tab():
    st.title("🏷️ WD14 & Manual Dataset Tagger")
    st.write(
        "Auto-tag anime/realism datasets using WD14 ONNX models, or inspect and manually edit existing `.txt` captions side-by-side."
    )

    tab_auto, tab_manual = st.tabs([
        "🤖 Automated WD14 Interrogation",
        "✏️ Manual Tag & Caption Editor"
    ])

    # ---------------------------------------------------------
    # TAB 1: WD14 Auto Tagger
    # ---------------------------------------------------------
    with tab_auto:
        st.subheader("WD14 Interrogator Settings")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            uploaded_img = st.file_uploader(
                "Upload Image to Interrogate",
                type=["jpg", "jpeg", "png", "webp"],
                key="wd14_uploader",
            )
            if uploaded_img:
                img = Image.open(uploaded_img)
                st.image(img, caption="Selected Image", use_container_width=True)

        with col_right:
            model_choice = st.selectbox(
                "WD14 Checkpoint Model",
                options=list(AVAILABLE_WD14_MODELS.keys()),
                index=0,
            )

            gen_thresh = st.slider("General Tag Threshold", 0.05, 0.95, 0.35, step=0.01)
            char_thresh = st.slider("Character Tag Threshold", 0.05, 0.95, 0.75, step=0.01)

            replace_und = st.checkbox("Replace Underscores with Spaces (`1girl` vs `1_girl`)", value=True)

            exclude_str = st.text_input("Exclude Tags (Comma Separated)", value="rating:safe, rating:general")
            exclude_tags = [t.strip() for t in exclude_str.split(",") if t.strip()]

            if st.button("🚀 Interrogate Image", type="primary", use_container_width=True):
                if not uploaded_img:
                    st.warning("Please upload an image first.")
                else:
                    with st.spinner("Running WD14 ONNX Inference..."):
                        try:
                            caption_str, ratings, df_res = run_wd14_tagger(
                                image=img,
                                model_name=model_choice,
                                general_thresh=gen_thresh,
                                character_thresh=char_thresh,
                                replace_underscores=replace_und,
                                exclude_tags=exclude_tags,
                            )

                            st.session_state["wd14_caption"] = caption_str
                            st.session_state["wd14_ratings"] = ratings
                            st.session_state["wd14_df"] = df_res
                            st.success("Interrogation Complete!")

                        except Exception as e:
                            st.error(f"Failed to interrogate image: {e}")

        # Interrogation Results
        if "wd14_caption" in st.session_state:
            st.markdown("---")
            st.subheader("📊 Output Results")

            # Display Ratings
            if "wd14_ratings" in st.session_state:
                ratings = st.session_state["wd14_ratings"]
                st.markdown("**Predicted Content Rating Scores:**")
                r_cols = st.columns(len(ratings))
                for i, (r_name, r_score) in enumerate(ratings.items()):
                    r_cols[i].metric(r_name.capitalize(), f"{r_score:.1%}")

            # Display Generated Tag String
            st.markdown("**Generated Tag Caption String:**")
            tag_text = st.text_area(
                "Copy/Edit Prompt Output",
                value=st.session_state["wd14_caption"],
                height=100,
            )

            # Detailed Breakdown Table
            if not st.session_state["wd14_df"].empty:
                with st.expander("🔍 Detailed Tag Confidence Breakdown"):
                    st.dataframe(st.session_state["wd14_df"], use_container_width=True)

    # ---------------------------------------------------------
    # TAB 2: Manual Tag & Caption Editor
    # ---------------------------------------------------------
    with tab_manual:
        st.subheader("Dataset Directory Caption Editor")

        dataset_dir = st.text_input(
            "Dataset Directory Path",
            value="datasets/training_set",
            help="Path to folder containing images and matching `.txt` caption files.",
        )

        if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
            valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            img_files = [
                f for f in os.listdir(dataset_dir)
                if os.path.splitext(f)[1].lower() in valid_exts
            ]

            if not img_files:
                st.warning("No images found in specified directory.")
            else:
                selected_img_name = st.selectbox("Select Image from Dataset", options=sorted(img_files))
                img_full_path = os.path.join(dataset_dir, selected_img_name)
                txt_full_path = os.path.splitext(img_full_path)[0] + ".txt"

                col_img, col_txt = st.columns([1, 1])

                with col_img:
                    st.image(img_full_path, caption=selected_img_name, use_container_width=True)

                with col_txt:
                    existing_caption = ""
                    if os.path.exists(txt_full_path):
                        with open(txt_full_path, "r", encoding="utf-8") as f:
                            existing_caption = f.read()

                    edited_caption = st.text_area(
                        f"Edit Text File (`{os.path.basename(txt_full_path)}`)",
                        value=existing_caption,
                        height=250,
                    )

                    if st.button("💾 Save Caption File", type="primary", use_container_width=True):
                        try:
                            saved_path = save_caption_file(img_full_path, edited_caption)
                            st.success(f"Successfully saved to `{os.path.basename(saved_path)}`!")
                        except Exception as e:
                            st.error(f"Failed to save caption file: {e}")
        else:
            st.info("Please enter a valid directory path above to edit dataset caption files.")


if __name__ == "__main__":
    wd14_tab()
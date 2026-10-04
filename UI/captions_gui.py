import os
import json
import streamlit as st
from pathlib import Path
from PIL import Image

# ==========================================
# BACKEND IMPORTS
# ==========================================
# Assuming this file is run from the project root and 'scripts' is a folder.
# If this file is INSIDE the 'scripts' folder, change to: from .captions_script import ...
from scripts.captions_script import (
    generate_blip_caption,
    answer_blip_vqa,
    run_clip_zero_shot,
    FLORENCE2_TASKS,
    run_florence2_task,
    run_wd14_tagger,
    AVAILABLE_WD14_MODELS
)

# ==========================================
# WD14 HELPER FUNCTIONS (File & Settings I/O)
# ==========================================
def get_project_root() -> Path:
    """Dynamically finds the project root by looking for the 'user' folder."""
    current = Path(__file__).resolve().parent
    for _ in range(3):
        if (current / "user").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent

def get_app_settings():
    """Loads settings.json dynamically based on project root."""
    root = get_project_root()
    settings_file = root / "user" / "settings.json"
    if settings_file.exists():
        try:
            with open(settings_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading settings: {e}")
    return {}

def scan_images_in_dir(target_dir: Path):
    """Scans for valid image files, filtering out generated edit files."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if not target_dir.exists():
        return []
    return [
        p for p in target_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts and not p.stem.endswith("-edit")
    ]

def get_wd14_models_dir() -> Path:
    settings = get_app_settings()
    wd14_folder = st.session_state.get(
        "settings_wd14_folder",
        settings.get("wd14_folder", "models/wd14")
    )
    return Path(str(wd14_folder)).expanduser()

def scan_wd14_models(models_folder: Path):
    """Scans for WD14 ONNX models (requires .onnx and selected_tags.csv)."""
    if not models_folder.exists():
        return []
    models = []
    for item in models_folder.iterdir():
        if item.is_dir():
            onnx_files = list(item.glob("*.onnx"))
            csv_files = list(item.glob("*.csv"))
            if onnx_files and csv_files:
                models.append(item.name)
        elif item.suffix == ".onnx":
            models.append(item.stem)
    return models

def load_existing_tags(txt_path: Path):
    if txt_path.exists():
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return [t.strip() for t in content.split(",")]
    return []

def save_tags_to_txt(txt_path: Path, tags: list):
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(", ".join(tags))

def run_wd14_local_inference(image_path: Path, model_folder: Path, threshold: float = 0.35):
    """Adapter to bridge the GUI's local file logic with the unified backend."""
    img = Image.open(image_path).convert("RGB")
    model_name = model_folder.name
    model_root = str(model_folder.parent)
    
    all_tags, _, _ = run_wd14_tagger(
        image=img,
        model_name=model_name,
        model_root=model_root,
        general_thresh=threshold,
        character_thresh=threshold,
        replace_underscores=True
    )
    return [t.strip() for t in all_tags.split(",") if t.strip()]


# ==========================================
# MAIN GUI FUNCTION
# ==========================================
def captions_gui():
    st.set_page_config(page_title="Vision & Language Suite", layout="wide")
    st.title("🖼️ Unified Vision & Language Suite")
    st.write("Generate captions, perform VQA, zero-shot classification, object detection, and booru-style tagging using state-of-the-art models.")

    tab_blip, tab_clip, tab_florence, tab_wd14 = st.tabs([
        "📝 BLIP: Caption & VQA",
        "🎯 CLIP: Zero-Shot",
        "🌸 Florence-2: Vision Tasks",
        "🏷️ WD14: Image Tagger"
    ])

    # ---------------------------------------------------------
    # TAB 1: BLIP (Captioning & VQA)
    # ---------------------------------------------------------
    with tab_blip:
        st.subheader("BLIP (Bootstrapped Language-Image Pre-training)")
        blip_col_left, blip_col_right = st.columns([1, 1])
        
        with blip_col_left:
            uploaded_blip_img = st.file_uploader("Upload Image for BLIP", type=["jpg", "jpeg", "png", "webp"], key="blip_uploader")
            if uploaded_blip_img:
                image = Image.open(uploaded_blip_img).convert("RGB")
                st.image(image, caption="Target Image", use_container_width=True)
                
        with blip_col_right:
            task_type = st.radio("Select BLIP Task", ["Image Captioning", "Visual Question Answering (VQA)"])
            
            if task_type == "Image Captioning":
                cond_prompt = st.text_input("Prefix / Context Prompt (Optional)", value="a photo of")
                cap_model_id = st.selectbox("BLIP Caption Model", ["Salesforce/blip-image-captioning-base", "Salesforce/blip-image-captioning-large"])
                max_tokens = st.slider("Max New Tokens", 10, 100, 30)
                
                if st.button("✨ Generate Caption", type="primary", use_container_width=True):
                    if not uploaded_blip_img: st.warning("Please upload an image first.")
                    else:
                        with st.spinner("Generating caption..."):
                            try:
                                caption = generate_blip_caption(image=image, model_id=cap_model_id, conditional_prompt=cond_prompt, max_new_tokens=max_tokens)
                                st.success("Caption Generated!")
                                st.info(f"**{caption}**")
                            except Exception as e: st.error(f"Error: {e}")
            else:  # VQA
                vqa_question = st.text_input("Ask a question about the image", value="What color is the object?")
                vqa_model_id = st.selectbox("BLIP VQA Model", ["Salesforce/blip-vqa-base", "Salesforce/blip-vqa-capfilt-large"])
                
                if st.button("❓ Answer Question", type="primary", use_container_width=True):
                    if not uploaded_blip_img: st.warning("Please upload an image first.")
                    elif not vqa_question.strip(): st.warning("Please enter a question.")
                    else:
                        with st.spinner("Analyzing image and answering..."):
                            try:
                                answer = answer_blip_vqa(image=image, question=vqa_question, model_id=vqa_model_id)
                                st.success("Answer Ready!")
                                st.info(f"**{answer}**")
                            except Exception as e: st.error(f"Error: {e}")

    # ---------------------------------------------------------
    # TAB 2: CLIP (Zero-Shot Classification)
    # ---------------------------------------------------------
    with tab_clip:
        st.subheader("CLIP (Contrastive Language-Image Pre-training)")
        clip_col_left, clip_col_right = st.columns([1, 1])
        
        with clip_col_left:
            uploaded_clip_img = st.file_uploader("Upload Image for CLIP", type=["jpg", "jpeg", "png", "webp"], key="clip_uploader")
            if uploaded_clip_img:
                clip_image = Image.open(uploaded_clip_img).convert("RGB")
                st.image(clip_image, caption="Target Image", use_container_width=True)
                
        with clip_col_right:
            clip_model_id = st.selectbox("CLIP Architecture Checkpoint", ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"])
            candidate_text = st.text_area("Candidate Labels (Comma-Separated)", value="a dog, a cat, a car, a landscape photo, a person working on a laptop")
            
            if st.button("🚀 Evaluate Zero-Shot Probabilities", type="primary", use_container_width=True):
                if not uploaded_clip_img: st.warning("Please upload an image first.")
                else:
                    labels = [l.strip() for l in candidate_text.split(",") if l.strip()]
                    if not labels: st.warning("Please specify at least one valid candidate label.")
                    else:
                        with st.spinner("Calculating embeddings and similarity scores..."):
                            try:
                                df_res = run_clip_zero_shot(image=clip_image, candidate_labels=labels, model_id=clip_model_id)
                                st.success("Evaluation complete!")
                                st.dataframe(df_res, use_container_width=True)
                                top_pred = df_res.iloc[0]
                                st.metric(label="Top Prediction", value=top_pred["Label"], delta=f"{top_pred['Probability']:.2%} Confidence")
                            except Exception as e: st.error(f"Error: {e}")

    # ---------------------------------------------------------
    # TAB 3: FLORENCE-2 (Vision & Multimodal Tasks)
    # ---------------------------------------------------------
    with tab_florence:
        st.subheader("Florence-2 Vision & Multimodal Suite")
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            uploaded_f2_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"], key="florence2_uploader")
            if uploaded_f2_img:
                f2_image = Image.open(uploaded_f2_img).convert("RGB")
                st.image(f2_image, caption="Input Image", use_container_width=True)
                
        with col_right:
            st.subheader("⚙️ Configuration")
            f2_model_id = st.selectbox("Florence-2 Model Checkpoint", [
                "microsoft/Florence-2-base", "microsoft/Florence-2-large", 
                "microsoft/Florence-2-base-ft", "microsoft/Florence-2-large-ft"
            ], index=0)
            
            task_choice = st.selectbox("Select Vision Task Prompt", options=list(FLORENCE2_TASKS.keys()), index=0)
            
            extra_text_input = ""
            if task_choice == "Caption to Phrase Grounding":
                extra_text_input = st.text_input("Grounding Phrase", value="a dog running on the grass")
                
            col_beams, col_tokens = st.columns(2)
            with col_beams: num_beams = st.slider("Beam Search", 1, 5, 3)
            with col_tokens: max_tokens = st.slider("Max New Tokens", 128, 2048, 1024)
            
            if st.button("🚀 Run Florence-2 Execution", type="primary", use_container_width=True):
                if not uploaded_f2_img: st.warning("Please upload an image first.")
                else:
                    with st.spinner("Processing image through Florence-2..."):
                        try:
                            raw_text, parsed_res, annotated_img = run_florence2_task(
                                image=f2_image, task_name=task_choice, text_input=extra_text_input,
                                model_id=f2_model_id, max_new_tokens=max_tokens, num_beams=num_beams
                            )
                            st.session_state["f2_raw_text"] = raw_text
                            st.session_state["f2_parsed"] = parsed_res
                            st.session_state["f2_annotated"] = annotated_img
                            st.success("Execution Complete!")
                        except Exception as e: st.error(f"Error: {e}")
                        
        if "f2_parsed" in st.session_state:
            st.markdown("---")
            st.subheader("📊 Execution Results")
            if st.session_state.get("f2_annotated") is not None:
                st.image(st.session_state["f2_annotated"], caption="Visual Detection & Grounding Output", use_container_width=True)
            st.markdown("**Parsed Structure / Text:**")
            parsed_data = st.session_state["f2_parsed"]
            if isinstance(parsed_data, dict): st.json(parsed_data)
            else: st.info(f"**{parsed_data}**")
            with st.expander("🔍 Raw Model Output Sequence"):
                st.code(st.session_state["f2_raw_text"], language="text")

    # ---------------------------------------------------------
    # TAB 4: WD14 (Image Tagger)
    # ---------------------------------------------------------
    with tab_wd14:
        st.subheader("WD14 Image Tagger")
        settings = get_app_settings()
        global_datasets_dir = settings.get("datasets_folder", "./datasets")
        base_datasets_path = Path(global_datasets_dir).expanduser()
        models_dir = get_wd14_models_dir()
        available_models = scan_wd14_models(models_dir)

        st.markdown("### ⚙️ Dataset & Model Settings")
        cfg_col1, cfg_col2, cfg_col3 = st.columns([2, 2, 1])
        
        with cfg_col1:
            if not base_datasets_path.exists():
                st.error(f"Base datasets folder does not exist: `{base_datasets_path}`")
            else:
                available_folders = [d.name for d in base_datasets_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
                available_folders.sort()
                folder_options = ["(Root Datasets Directory)"] + available_folders
                selected_subfolder = st.selectbox("Select Dataset Directory", options=folder_options, index=0, key="wd14_subfolder_select")
                datasets_folder = base_datasets_path if selected_subfolder == "(Root Datasets Directory)" else base_datasets_path / selected_subfolder
                
        with cfg_col2:
            st.info(f"📁 **WD14 Models Folder:** `{models_dir}`")
            model_options = available_models if available_models else ["No models found"]
            if not available_models: st.warning("No valid WD14 models found.")
            selected_model = st.selectbox("WD14 Model", options=model_options, key="wd14_model_select")
            
        with cfg_col3:
            threshold = st.slider("Confidence Threshold", 0.05, 0.95, 0.35, 0.05, key="wd14_threshold")

        if "wd14_dataset_folder" not in st.session_state or st.session_state.wd14_dataset_folder != str(datasets_folder):
            st.session_state.wd14_dataset_folder = str(datasets_folder)
            st.session_state.wd14_image_index = 0
            st.session_state.wd14_tags_cache = {}

        image_files = scan_images_in_dir(datasets_folder) if base_datasets_path.exists() else []

        # Batch Processing
        st.markdown("---")
        st.subheader("🚀 Batch Processing")
        if st.button("⚡ Run WD14 on All Untagged Images", use_container_width=True, type="primary"):
            if not image_files: st.warning("No images found.")
            elif selected_model == "No models found": st.error("Please select a valid WD14 model.")
            else:
                model_path = models_dir / selected_model
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i, img_file in enumerate(image_files):
                    txt_file = datasets_folder / f"{img_file.stem}.txt"
                    if not txt_file.exists():
                        status_text.text(f"Tagging {img_file.name}...")
                        tags = run_wd14_local_inference(img_file, model_path, threshold)
                        save_tags_to_txt(txt_file, tags)
                        st.session_state.wd14_tags_cache[img_file.name] = tags
                    progress_bar.progress((i + 1) / len(image_files))
                status_text.text("Batch processing complete!")
                st.success(f"Processed {len(image_files)} images.")
                st.rerun()

        # Single Image Editor
        st.markdown("---")
        if not image_files:
            st.warning(f"No valid image files found in `{datasets_folder}`.")
        else:
            total_imgs = len(image_files)
            if st.session_state.get("wd14_image_index", 0) >= total_imgs: st.session_state.wd14_image_index = total_imgs - 1
            elif st.session_state.get("wd14_image_index", 0) < 0: st.session_state.wd14_image_index = 0
            
            idx = st.session_state.wd14_image_index
            current_file = image_files[idx]
            filename = current_file.name
            txt_file = datasets_folder / f"{current_file.stem}.txt"

            col1, col2, col3 = st.columns(3)
            tagged_count = len([f for f in image_files if (datasets_folder / f"{f.stem}.txt").exists()])
            col1.metric("Total Images", total_imgs)
            col2.metric("Tagged Images", tagged_count)
            col3.metric("Progress", f"{idx + 1} / {total_imgs}")
            st.progress((idx + 1) / total_imgs)

            img_col, tag_col = st.columns([2, 1])
            with img_col:
                st.subheader(f"🖼️ `{filename}`")
                try: st.image(str(current_file), use_container_width=True)
                except Exception as e: st.error(f"Error rendering image: {e}")
                
            with tag_col:
                st.subheader("🏷️ Tags")
                if filename in st.session_state.wd14_tags_cache:
                    current_tags = st.session_state.wd14_tags_cache[filename]
                else:
                    current_tags = load_existing_tags(txt_file)
                    st.session_state.wd14_tags_cache[filename] = current_tags

                tags_text = st.text_area("Edit Tags (comma-separated)", value=", ".join(current_tags), height=250, key=f"wd14_tag_editor_{idx}")
                col_save, col_gen = st.columns(2)
                
                with col_save:
                    if st.button("💾 Save Tags", use_container_width=True, key="save_wd14_tags"):
                        new_tags = [t.strip() for t in tags_text.split(",") if t.strip()]
                        save_tags_to_txt(txt_file, new_tags)
                        st.session_state.wd14_tags_cache[filename] = new_tags
                        st.toast("Tags saved!", icon="💾")
                        
                with col_gen:
                    if st.button("✨ Generate Tags", use_container_width=True, key="gen_wd14_tags"):
                        if selected_model == "No models found": st.error("No model selected.")
                        else:
                            with st.spinner("Running WD14 inference..."):
                                model_path = models_dir / selected_model
                                new_tags = run_wd14_local_inference(current_file, model_path, threshold)
                                st.session_state.wd14_tags_cache[filename] = new_tags
                                save_tags_to_txt(txt_file, new_tags)
                                st.rerun()

                st.markdown("---")
                nav_prev, nav_next = st.columns(2)
                with nav_prev:
                    if st.button("⬅ Previous", use_container_width=True, key="wd14_prev_button"):
                        if st.session_state.wd14_image_index > 0:
                            st.session_state.wd14_image_index -= 1
                            st.rerun()
                with nav_next:
                    if st.button("➡️ Next", use_container_width=True, key="wd14_next_button"):
                        if st.session_state.wd14_image_index < total_imgs - 1:
                            st.session_state.wd14_image_index += 1
                            st.rerun()

if __name__ == "__main__":
    captions_gui()
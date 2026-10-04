import json
import os
import sys
import math
from pathlib import Path
import streamlit as st
import numpy as np

# statement to indicate that this module is being imported
print("wd14.py (WD14 Tagger) is being imported")

#================================================
# Safe resolution of project root into sys.path
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent if FILE_PATH.parent.name == "tabs" else FILE_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load settings from settings.json
def get_app_settings():
    settings_file = PROJECT_ROOT / "user" / "settings.json"
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
    images = [
        p for p in target_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts and not p.stem.endswith("-edit")
    ]
    return images

def get_wd14_models_dir() -> Path:
    settings = get_app_settings()
    # Look for a specific WD14 folder, fallback to a general models folder
    wd14_folder = st.session_state.get(
        "settings_wd14_folder",
        settings.get("wd14_folder", "models/wd14")
    )
    return Path(str(wd14_folder)).expanduser()

def scan_wd14_models(models_folder: Path):
    """Scans for WD14 ONNX models (requires .onnx and selected_tags.csv)"""
    if not models_folder.exists():
        return []
    # A valid WD14 model folder usually contains an .onnx and a .csv
    models = []
    for item in models_folder.iterdir():
        if item.is_dir():
            onnx_files = list(item.glob("*.onnx"))
            csv_files = list(item.glob("*.csv"))
            if onnx_files and csv_files:
                models.append(item.name)
        elif item.suffix == ".onnx":
            # If they are just loose files in the root
            models.append(item.stem)
    return models

def preprocess_image_for_wd14(img_path: Path, target_size=448):
    """Prepares an image for WD14 inference (resize to 448x448 with padding)."""
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    
    # Calculate new dimensions to maintain aspect ratio
    old_size = img.size
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    
    # Resize
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Create a new black image and paste the resized image in the center
    new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    new_img.paste(img, ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2))
    
    # Convert to numpy array and normalize
    img_array = np.asarray(new_img, dtype=np.float32)
    img_array = img_array / 255.0
    
    # Transpose to CHW and add batch dimension (N, C, H, W)
    img_array = img_array.transpose(2, 0, 1)[np.newaxis, ...]
    return img_array

def run_wd14_inference(image_path: Path, model_folder: Path, threshold: float = 0.35):
    """Runs WD14 ONNX inference and returns a list of tags."""
    try:
        import onnxruntime as ort
        import pandas as pd
    except ImportError:
        st.error("Missing dependencies! Please install `onnxruntime` and `pandas`.")
        return []

    # Find the actual .onnx and .csv files
    onnx_path = None
    csv_path = None
    
    if model_folder.is_dir():
        onnx_files = list(model_folder.glob("*.onnx"))
        csv_files = list(model_folder.glob("*.csv"))
        if onnx_files and csv_files:
            onnx_path = onnx_files[0]
            csv_path = csv_files[0]
    else:
        onnx_path = model_folder.with_suffix(".onnx")
        csv_path = model_folder.with_suffix(".csv")

    if not onnx_path or not onnx_path.exists() or not csv_path or not csv_path.exists():
        st.error(f"Could not find .onnx and .csv files in `{model_folder}`")
        return []

    # 1. Load Tags CSV
    tags_df = pd.read_csv(csv_path)
    tag_names = tags_df['name'].tolist()

    # 2. Preprocess Image
    img_array = preprocess_image_for_wd14(image_path)

    # 3. Run ONNX Inference
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    result = sess.run([label_name], {input_name: img_array})[0]
    
    # 4. Apply Sigmoid (if not already applied by the model) and filter by threshold
    # Note: Some WD14 models output probabilities directly, some need sigmoid. 
    # We'll assume standard raw logits and apply sigmoid just in case, or clamp if already 0-1.
    if np.max(result) > 1.0:
        result = 1 / (1 + np.exp(-result)) # Sigmoid
        
    result = result[0] # Remove batch dimension
    
    # Filter tags
    tags = []
    for i, score in enumerate(result):
        if score >= threshold and i < len(tag_names):
            tag = tag_names[i].replace("_", " ") # WD14 uses underscores, but spaces are often preferred
            tags.append((tag, float(score)))
            
    # Sort by confidence
    tags.sort(key=lambda x: x[1], reverse=True)
    return [t[0] for t in tags]

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

# ==========================================
# MAIN UI TAB
# ==========================================
def wd14_tab():
    st.title("🏷️ WD14 Image Tagger")
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
            return
        available_folders = [d.name for d in base_datasets_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        available_folders.sort()
        folder_options = ["(Root Datasets Directory)"] + available_folders
        selected_subfolder = st.selectbox("Select Dataset Directory", options=folder_options, index=0, key="wd14_subfolder_select")
        datasets_folder = base_datasets_path if selected_subfolder == "(Root Datasets Directory)" else base_datasets_path / selected_subfolder
        
    with cfg_col2:
        st.info(f"📁 **WD14 Models Folder:** `{models_dir}`")
        if available_models:
            model_options = available_models
        else:
            st.warning("No valid WD14 models found. Ensure folders contain `.onnx` and `.csv` files.")
            model_options = ["No models found"]
        selected_model = st.selectbox("WD14 Model", options=model_options, key="wd14_model_select")
        
    with cfg_col3:
        threshold = st.slider("Confidence Threshold", min_value=0.05, max_value=0.95, value=0.35, step=0.05, key="wd14_threshold")

    # Initialize Session State for this dataset
    if "wd14_dataset_folder" not in st.session_state or st.session_state.wd14_dataset_folder != str(datasets_folder):
        st.session_state.wd14_dataset_folder = str(datasets_folder)
        st.session_state.wd14_image_index = 0
        st.session_state.wd14_tags_cache = {} # Cache tags in memory
        
    st.markdown("---")
    
    image_files = scan_images_in_dir(datasets_folder)
    
    # Batch Processing Tool
    st.subheader("🚀 Batch Processing")
    if st.button("⚡ Run WD14 on All Untagged Images", use_container_width=True, type="primary"):
        if not image_files:
            st.warning("No images found.")
        elif selected_model == "No models found":
            st.error("Please select a valid WD14 model.")
        else:
            model_path = models_dir / selected_model
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, img_file in enumerate(image_files):
                txt_file = datasets_folder / f"{img_file.stem}.txt"
                if not txt_file.exists(): # Only process untagged
                    status_text.text(f"Tagging {img_file.name}...")
                    tags = run_wd14_inference(img_file, model_path, threshold)
                    save_tags_to_txt(txt_file, tags)
                    st.session_state.wd14_tags_cache[img_file.name] = tags
                progress_bar.progress((i + 1) / len(image_files))
                
            status_text.text("Batch processing complete!")
            st.success(f"Processed {len(image_files)} images.")
            st.rerun()

    st.markdown("---")

    if not image_files:
        st.warning(f"No valid image files found in `{datasets_folder}`.")
        return
        
    total_imgs = len(image_files)
    
    # Navigation bounds
    if st.session_state.get("wd14_image_index", 0) >= total_imgs:
        st.session_state.wd14_image_index = total_imgs - 1
    elif st.session_state.get("wd14_image_index", 0) < 0:
        st.session_state.wd14_image_index = 0
        
    idx = st.session_state.wd14_image_index
    current_file = image_files[idx]
    filename = current_file.name
    txt_file = datasets_folder / f"{current_file.stem}.txt"
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    tagged_count = len([f for f in image_files if (datasets_folder / f"{f.stem}.txt").exists()])
    col1.metric("Total Images", total_imgs)
    col2.metric("Tagged Images", tagged_count)
    col3.metric("Progress", f"{idx + 1} / {total_imgs}")
    st.progress((idx + 1) / total_imgs)
    
    # Main Layout
    img_col, tag_col = st.columns([2, 1])
    
    with img_col:
        st.subheader(f"🖼️ `{filename}`")
        try:
            st.image(str(current_file), use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering image: {e}")
            
    with tag_col:
        st.subheader("🏷️ Tags")
        
        # Load tags (from cache or disk)
        if filename in st.session_state.wd14_tags_cache:
            current_tags = st.session_state.wd14_tags_cache[filename]
        else:
            current_tags = load_existing_tags(txt_file)
            st.session_state.wd14_tags_cache[filename] = current_tags
            
        # Tag Editor
        tags_text = st.text_area(
            "Edit Tags (comma-separated)", 
            value=", ".join(current_tags), 
            height=250, 
            key=f"wd14_tag_editor_{idx}"
        )
        
        col_save, col_gen = st.columns(2)
        with col_save:
            if st.button("💾 Save Tags", use_container_width=True, key="save_wd14_tags"):
                new_tags = [t.strip() for t in tags_text.split(",") if t.strip()]
                save_tags_to_txt(txt_file, new_tags)
                st.session_state.wd14_tags_cache[filename] = new_tags
                st.toast("Tags saved!", icon="💾")
                
        with col_gen:
            if st.button("✨ Generate Tags", use_container_width=True, key="gen_wd14_tags"):
                if selected_model == "No models found":
                    st.error("No model selected.")
                else:
                    with st.spinner("Running WD14 inference..."):
                        model_path = models_dir / selected_model
                        new_tags = run_wd14_inference(current_file, model_path, threshold)
                        st.session_state.wd14_tags_cache[filename] = new_tags
                        save_tags_to_txt(txt_file, new_tags)
                        st.rerun()

        # Navigation
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

# ==========================================
# Universal Entry Points & Aliases
# ==========================================
def main(): wd14_tab()
def app(): wd14_tab()
def show(): wd14_tab()
def render(): wd14_tab()

# Map all expected entry points to the actual UI function
render_tab = wd14_tab
wd14_tagger_tab = wd14_tab

if __name__ == "__main__":
    wd14_tab()
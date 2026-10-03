import json
import os
import shutil
from pathlib import Path
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from scripts.settings.settings_script import load_settings


def scan_images_in_dir(target_dir: Path, recursive: bool = False):
    """Scans for valid images excluding generated preview artifacts."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    def is_valid(p: Path):
        return p.is_file() and p.suffix.lower() in valid_exts and not p.stem.endswith("-edit")

    if recursive:
        return [p for p in target_dir.rglob("*") if is_valid(p)]
    return [p for p in target_dir.iterdir() if is_valid(p)]


def get_cls_models_dir() -> Path:
    """Resolves classification models path from settings."""
    settings = load_settings()
    cls_folder = st.session_state.get(
        "settings_ultralytics_cls_folder",
        settings.get("ultralytics_cls_folder", "models/ultralytics/cls")
    )
    return Path(cls_folder).expanduser()


def scan_cls_models(models_folder: Path):
    """Scans models directory for .pt classification model files."""
    if not models_folder.exists():
        return []
    return list(models_folder.glob("*.pt"))


def save_dataset_manifest(dataset_folder: Path, classes: list):
    """Generates json manifest and yolo data.yaml metadata files."""
    manifest_path = dataset_folder / "dataset_manifest.json"
    yaml_path = dataset_folder / "data.yaml"
    
    annotations = {}
    for cls in classes:
        cls_dir = dataset_folder / cls
        if cls_dir.exists():
            for img in scan_images_in_dir(cls_dir, recursive=False):
                annotations[img.name] = cls

    manifest_data = {
        "classes": classes,
        "total_images": len(annotations),
        "annotations": annotations
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4)

    yaml_content = f"path: {dataset_folder.resolve()}\nnames:\n"
    for i, cls in enumerate(classes):
        yaml_content += f"  {i}: {cls}\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)


def save_rendered_cls_preview(img_path: Path, class_name: str, confidence: float = None) -> Path:
    """Saves annotated classification image artifact to filename-edit.jpeg in dataset folder."""
    out_path = img_path.parent / f"{img_path.stem}-edit.jpeg"
    try:
        pil_img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(pil_img)
        text = f"Class: {class_name}" + (f" ({confidence*100:.1f}%)" if confidence is not None else "")
        draw.rectangle([10, 10, min(350, pil_img.width - 10), 50], fill=(0, 0, 0, 180))
        draw.text((20, 20), text, fill=(0, 255, 128))
        pil_img.save(out_path, format="JPEG", quality=90)
    except Exception as e:
        st.warning(f"Could not save preview artifact: {e}")
    return out_path


def run_cls_test(model_path: str, image_files: list, conf_thresh: float = 0.25):
    """Backend test function running inference and generating annotated preview files."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    test_results = []

    for img_p in image_files[:6]:
        results = model.predict(source=str(img_p), conf=conf_thresh, verbose=False)
        top_cls = "Unknown"
        conf_val = 0.0

        if results and results[0].probs is not None:
            top_id = int(results[0].probs.top1)
            conf_val = float(results[0].probs.top1conf)
            top_cls = results[0].names.get(top_id, str(top_id))

        out_edited_path = save_rendered_cls_preview(img_p, top_cls, conf_val)
        test_results.append((img_p.name, top_cls, conf_val, out_edited_path))

    return test_results


def cls_tab():
    st.title("🏷️️ Classification Tagger & Auto-Labeller (YOLO26)")

    # Settings & Paths
    saved_settings = load_settings()
    global_datasets_dir = st.session_state.get(
        "settings_datasets_folder", 
        saved_settings.get("datasets_folder", "./datasets")
    )
    base_datasets_path = Path(global_datasets_dir).expanduser()

    models_dir = get_cls_models_dir()
    available_models = scan_cls_models(models_dir)

    # Configuration Banner
    st.markdown("### ⚙️ Dataset & Backend Model Settings")
    cfg_col1, cfg_col2 = st.columns(2)

    with cfg_col1:
        if not base_datasets_path.exists():
            st.error(f"Base datasets folder does not exist: `{base_datasets_path}`")
            return
        
        available_folders = [d.name for d in base_datasets_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        available_folders.sort()
        folder_options = ["(Root Datasets Directory)"] + available_folders
        selected_subfolder = st.selectbox("Select Dataset Directory", options=folder_options, index=0, key="cls_subfolder_select")
        datasets_folder = base_datasets_path if selected_subfolder == "(Root Datasets Directory)" else base_datasets_path / selected_subfolder

    with cfg_col2:
        st.info(f"📁 **Classification Models Folder:** `{models_dir}`")
        if available_models:
            model_opts = [str(p) for p in available_models]
        else:
            model_opts = ["yolo26n-cls.pt", "yolo11n-cls.pt", "yolov8n-cls.pt"]
        selected_model = st.selectbox("Backend YOLO Model", options=model_opts, key="cls_model_select")

    # Class Discovery
    class_dirs = [d.name for d in datasets_folder.iterdir() if d.is_dir() and not d.name.startswith(".")]
    class_dirs.sort()

    if "cls_classes" not in st.session_state or st.session_state.get("last_cls_dataset") != str(datasets_folder):
        st.session_state.cls_classes = class_dirs if class_dirs else ["class_a", "class_b"]
        st.session_state.last_cls_dataset = str(datasets_folder)
        st.session_state.cls_image_index = 0

    st.markdown("---")

    # Class Management & Backend Test Toolbar
    st.subheader("📁 Class Management & Backend Test")
    col_cls1, col_cls2, col_test = st.columns([2, 1, 1])

    with col_cls1:
        new_class = st.text_input("New Class Name", key="new_cls_input", placeholder="e.g. cat, dog")
    with col_cls2:
        st.write("")
        st.write("")
        if st.button("➕ Create Folder", use_container_width=True) and new_class:
            cls_clean = new_class.strip().replace(" ", "_")
            if cls_clean:
                (datasets_folder / cls_clean).mkdir(parents=True, exist_ok=True)
                if cls_clean not in st.session_state.cls_classes:
                    st.session_state.cls_classes.append(cls_clean)
                    st.session_state.cls_classes.sort()
                save_dataset_manifest(datasets_folder, st.session_state.cls_classes)
                st.success(f"Created class directory: `{cls_clean}`")
                st.rerun()

    view_mode = st.radio(
        "View Source Images From:",
        options=["Unassigned (Root Directory Only)", "All Images (Include Subfolders)"],
        horizontal=True,
        key="cls_view_mode"
    )

    recursive = (view_mode == "All Images (Include Subfolders)")
    image_files = scan_images_in_dir(datasets_folder, recursive=recursive)

    with col_test:
        st.write("")
        st.write("")
        if st.button("🔍 Test Model Tagging", type="secondary", use_container_width=True):
            if not image_files:
                st.warning("No images available to run test.")
            else:
                with st.spinner("Running model inference and rendering preview files..."):
                    test_results = run_cls_test(selected_model, image_files)
                st.success("Test inference completed! Saved rendered samples as `filename-edit.jpeg`.")
                st.markdown("#### Test Previews")
                t_cols = st.columns(min(len(test_results), 3))
                for i, (fname, top_cls, conf, saved_path) in enumerate(test_results):
                    with t_cols[i % 3]:
                        st.image(str(saved_path), caption=f"{fname}\nPred: {top_cls} ({conf*100:.1f}%)", use_container_width=True)

    if not image_files:
        st.warning(f"No unassigned images found in `{datasets_folder}`.")
        return

    total_imgs = len(image_files)
    if st.session_state.get("cls_image_index", 0) >= total_imgs:
        st.session_state.cls_image_index = total_imgs - 1
    elif st.session_state.get("cls_image_index", 0) < 0:
        st.session_state.cls_image_index = 0

    idx = st.session_state.cls_image_index
    current_file = image_files[idx]
    current_class = current_file.parent.name if current_file.parent != datasets_folder else "Unassigned"

    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("Total Images", total_imgs)
    col_stat2.metric("Current Class", current_class)
    col_stat3.metric("Progress", f"{idx + 1} / {total_imgs}")

    st.progress((idx + 1) / total_imgs)

    img_col, tag_col = st.columns([2, 1])

    with img_col:
        st.subheader(f"🖼️ `{current_file.name}`")
        try:
            image = Image.open(current_file)
            st.image(image, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")

    with tag_col:
        st.subheader("Assign Class & Save Artifact")

        for i, cls_name in enumerate(st.session_state.cls_classes):
            shortcut = f"[{i+1}] " if i < 9 else ""
            btn_type = "primary" if cls_name == current_class else "secondary"

            if st.button(f"{shortcut}📁 {cls_name}", key=f"cls_btn_{cls_name}", type=btn_type, use_container_width=True):
                target_dir = datasets_folder / cls_name
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / current_file.name

                if target_path != current_file:
                    if target_path.exists():
                        base, ext = os.path.splitext(current_file.name)
                        target_path = target_dir / f"{base}_{int(os.path.getmtime(current_file))}{ext}"
                    
                    shutil.move(str(current_file), str(target_path))
                    save_rendered_cls_preview(target_path, cls_name)
                    save_dataset_manifest(datasets_folder, st.session_state.cls_classes)
                    st.toast(f"Saved & Moved to `{cls_name}`", icon="💾")

                if st.session_state.cls_image_index < total_imgs - 1:
                    st.session_state.cls_image_index += 1
                st.rerun()

        st.markdown("---")
        nav_prev, nav_next = st.columns(2)
        with nav_prev:
            if st.button("⬅️ Previous", use_container_width=True):
                if st.session_state.cls_image_index > 0:
                    st.session_state.cls_image_index -= 1
                    st.rerun()
                    
        with nav_next:
            if st.button("Next ➡️️", use_container_width=True):
                if st.session_state.cls_image_index < total_imgs - 1:
                    st.session_state.cls_image_index += 1
                    st.rerun()


render_cls_tab = cls_tab
classification_tab = cls_tab
render_classification_tab = cls_tab

if __name__ == "__main__":
    cls_tab()
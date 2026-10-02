import json
import os
import shutil
from pathlib import Path
import streamlit as st
from PIL import Image

from scripts.settings.settings_script import load_settings


def scan_images_in_dir(target_dir: Path, recursive: bool = False):
    """Scans for images. If recursive is False, only scans target_dir root."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if recursive:
        return [p for p in target_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts]
    return [p for p in target_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]


def save_dataset_manifest(dataset_folder: Path, classes: list):
    """Generates optional json and yolo data.yaml metadata files."""
    manifest_path = dataset_folder / "dataset_manifest.json"
    yaml_path = dataset_folder / "data.yaml"
    
    # Generate JSON summary
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

    # Generate YOLO standard classification yaml
    yaml_content = f"path: {dataset_folder.resolve()}\nnames:\n"
    for i, cls in enumerate(classes):
        yaml_content += f"  {i}: {cls}\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)


def cls_tab():
    st.title("🏷️ Classification Tagger (YOLO Format)")

    # Load settings and resolve paths
    saved_settings = load_settings()
    global_datasets_dir = st.session_state.get(
        "settings_datasets_folder", 
        saved_settings.get("datasets_folder", "./datasets")
    )

    base_datasets_path = Path(global_datasets_dir).expanduser()

    if not base_datasets_path.exists():
        st.error(f"Base datasets folder does not exist:\n\n`{base_datasets_path}`")
        return

    # Select subfolder under datasets
    available_folders = [d.name for d in base_datasets_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    available_folders.sort()
    
    folder_options = ["(Root Datasets Directory)"] + available_folders
    selected_subfolder = st.selectbox("Select Dataset Directory", options=folder_options, index=0)

    datasets_folder = base_datasets_path if selected_subfolder == "(Root Datasets Directory)" else base_datasets_path / selected_subfolder

    source_dir_str = st.text_input("Dataset Directory Path", value=str(datasets_folder), key="cls_source_dir_input")
    datasets_folder = Path(source_dir_str).expanduser()

    if not datasets_folder.exists():
        st.error(f"Selected dataset folder does not exist:\n\n`{datasets_folder}`")
        return

    # Discover existing class subfolders
    class_dirs = [d.name for d in datasets_folder.iterdir() if d.is_dir() and not d.name.startswith(".")]
    class_dirs.sort()

    if "cls_classes" not in st.session_state or st.session_state.get("last_dataset_folder") != str(datasets_folder):
        st.session_state.cls_classes = class_dirs if class_dirs else ["class_a", "class_b"]
        st.session_state.last_dataset_folder = str(datasets_folder)
        st.session_state.cls_image_index = 0

    st.markdown("---")

    # Class management
    st.subheader("📁 Class Directories")
    col_add1, col_add2 = st.columns([3, 1])
    with col_add1:
        new_class = st.text_input("New Class Name", key="new_cls_input", placeholder="e.g. cat, dog")
    with col_add2:
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

    # View Mode Selector
    view_mode = st.radio(
        "View Source Images From:",
        options=["Unassigned (Root Directory Only)", "All Images (Include Subfolders)"],
        horizontal=True
    )

    recursive = (view_mode == "All Images (Include Subfolders)")
    image_files = scan_images_in_dir(datasets_folder, recursive=recursive)

    if not image_files:
        st.warning(f"No valid images found in `{datasets_folder}` for mode: **{view_mode}**.")
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
    col_stat1.metric("Total Images in View", total_imgs)
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
        st.subheader("Move to Class Folder")

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
                    save_dataset_manifest(datasets_folder, st.session_state.cls_classes)
                    st.toast(f"Moved to `{cls_name}`")

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
            if st.button("Next ➡️", use_container_width=True):
                if st.session_state.cls_image_index < total_imgs - 1:
                    st.session_state.cls_image_index += 1
                    st.rerun()

        target_idx = st.number_input("Jump to Image #", min_value=1, max_value=total_imgs, value=idx + 1)
        if target_idx - 1 != idx:
            st.session_state.cls_image_index = target_idx - 1
            st.rerun()


render_cls_tab = cls_tab
classification_tab = cls_tab
render_classification_tab = cls_tab

if __name__ == "__main__":
    cls_tab()
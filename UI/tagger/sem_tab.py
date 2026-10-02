import os
import json
import numpy as np
from pathlib import Path
import streamlit as st
from PIL import Image, ImageDraw

from scripts.settings.settings_script import load_settings


def scan_images_in_dir(target_dir: Path):
    """Scans for valid image files inside the target directory."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if not target_dir.exists():
        return []
    return [
        p for p in target_dir.iterdir() 
        if p.is_file() and p.suffix.lower() in valid_exts and not p.name.endswith("_mask.png")
    ]


def save_semantic_png_mask(dataset_folder: Path, filename_stem: str, image_size: tuple, polygons: list, classes: list):
    """Generates a dense PNG mask image where pixel values correspond to class IDs (YOLO26-sem format)."""
    w, h = image_size
    mask_array = np.zeros((h, w), dtype=np.uint8)

    for poly in polygons:
        cls_name = poly["class"]
        if cls_name not in classes:
            continue
        cls_id = classes.index(cls_name) + 1  # 0 reserved for background

        pts = [(int(p["x"]), int(p["y"])) for p in poly["points"]]
        if len(pts) >= 3:
            mask_img = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask_img)
            draw.polygon(pts, fill=cls_id)
            mask_array = np.maximum(mask_array, np.array(mask_img))

    mask_path = dataset_folder / f"{filename_stem}_mask.png"
    Image.fromarray(mask_array).save(mask_path)


def save_dataset_manifest(dataset_folder: Path, classes: list, annotations: dict):
    """Saves sem_manifest.json and generates data.yaml for YOLO26 semantic segmentation."""
    manifest_path = dataset_folder / "sem_manifest.json"
    yaml_path = dataset_folder / "data.yaml"

    manifest_data = {
        "classes": classes,
        "annotations": annotations
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4)

    yaml_content = f"path: {dataset_folder.resolve()}\ntrain: .\nval: .\n\nnames:\n"
    for i, cls in enumerate(classes):
        yaml_content += f"  {i}: {cls}\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)


def sem_tab():
    st.title("✂️ YOLO26 Semantic Segmentation Tagger (`-sem`)")

    # Load system settings cleanly
    try:
        saved_settings = load_settings()
    except Exception:
        saved_settings = {}

    global_datasets_dir = st.session_state.get(
        "settings_datasets_folder", 
        saved_settings.get("datasets_folder", "./datasets")
    )

    base_datasets_path = Path(global_datasets_dir).expanduser()

    if not base_datasets_path.exists():
        st.error(f"Base datasets folder does not exist:\n\n`{base_datasets_path}`")
        return

    # Subfolder Selection
    available_folders = [d.name for d in base_datasets_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    available_folders.sort()
    
    folder_options = ["(Root Datasets Directory)"] + available_folders
    selected_subfolder = st.selectbox("Select Dataset Directory", options=folder_options, index=0)

    datasets_folder = base_datasets_path if selected_subfolder == "(Root Datasets Directory)" else base_datasets_path / selected_subfolder

    source_dir_str = st.text_input("Dataset Directory Path", value=str(datasets_folder), key="sem_source_dir_input")
    datasets_folder = Path(source_dir_str).expanduser()

    if not datasets_folder.exists():
        st.error(f"Selected dataset directory does not exist:\n\n`{datasets_folder}`")
        return

    manifest_path = datasets_folder / "sem_manifest.json"

    # Persistent Session State Setup
    if "sem_dataset_folder" not in st.session_state or st.session_state.sem_dataset_folder != str(datasets_folder):
        st.session_state.sem_dataset_folder = str(datasets_folder)
        
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                st.session_state.sem_classes = manifest.get("classes", ["object", "background"])
                st.session_state.sem_annotations = manifest.get("annotations", {})
            except Exception:
                st.session_state.sem_classes = ["object", "background"]
                st.session_state.sem_annotations = {}
        else:
            st.session_state.sem_classes = ["object", "background"]
            st.session_state.sem_annotations = {}

        st.session_state.sem_image_index = 0

    st.markdown("---")

    # Inlined Class Management
    st.subheader("🏷️ Class Management")
    col_add1, col_add2 = st.columns([3, 1])
    with col_add1:
        new_class = st.text_input("New Class Label", key="sem_new_cls_input", placeholder="e.g. road, sky, building")
    with col_add2:
        st.write("")
        st.write("")
        if st.button("➕ Add Class", use_container_width=True) and new_class:
            cls_clean = new_class.strip().lower().replace(" ", "_")
            if cls_clean and cls_clean not in st.session_state.sem_classes:
                st.session_state.sem_classes.append(cls_clean)
                save_dataset_manifest(datasets_folder, st.session_state.sem_classes, st.session_state.sem_annotations)
                st.success(f"Added class: `{cls_clean}`")
                st.rerun()

    if st.session_state.sem_classes:
        st.caption("Active Classes:")
        st.write(" ".join([f"`🏷️ {c}`" for c in st.session_state.sem_classes]))

    st.markdown("---")

    image_files = scan_images_in_dir(datasets_folder)

    if not image_files:
        st.warning(f"No valid image files found directly in `{datasets_folder}`.")
        return

    total_imgs = len(image_files)

    if st.session_state.get("sem_image_index", 0) >= total_imgs:
        st.session_state.sem_image_index = total_imgs - 1
    elif st.session_state.get("sem_image_index", 0) < 0:
        st.session_state.sem_image_index = 0

    idx = st.session_state.sem_image_index
    current_file = image_files[idx]
    filename = current_file.name

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Images", total_imgs)
    col2.metric("Tagged Images", len([k for k, v in st.session_state.sem_annotations.items() if v.get("polygons")]))
    col3.metric("Progress", f"{idx + 1} / {total_imgs}")

    st.progress((idx + 1) / total_imgs)

    img_col, tag_col = st.columns([2, 1])

    if "current_polygon_vertices" not in st.session_state:
        st.session_state.current_polygon_vertices = []

    with img_col:
        st.subheader(f"🖼️ `{filename}`")
        try:
            image = Image.open(current_file).convert("RGBA")
            w, h = image.size

            overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)

            existing_data = st.session_state.sem_annotations.get(filename, {})
            polygons = existing_data.get("polygons", [])

            for poly in polygons:
                pts = [(p["x"], p["y"]) for p in poly["points"]]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=(0, 255, 128, 80), outline=(0, 255, 128, 255))
                    draw.text((pts[0][0] + 5, pts[0][1] - 5), poly["class"], fill="yellow")

            curr_pts = [(p["x"], p["y"]) for p in st.session_state.current_polygon_vertices]
            if len(curr_pts) > 0:
                for pt in curr_pts:
                    draw.ellipse((pt[0]-4, pt[1]-4, pt[0]+4, pt[1]+4), fill="red", outline="white")
                if len(curr_pts) >= 2:
                    draw.line(curr_pts, fill="cyan", width=2)

            annotated_img = Image.alpha_composite(image, overlay)
            st.image(annotated_img, use_container_width=True)
            st.caption(f"Resolution: {w}x{h} px")
        except Exception as e:
            st.error(f"Error rendering image: {e}")

    with tag_col:
        st.subheader("Add Semantic Region")

        selected_cls = st.selectbox("Select Class", options=st.session_state.sem_classes)

        vx = st.number_input("Vertex X (px)", min_value=0, max_value=w if 'w' in locals() else 1920, value=w//2 if 'w' in locals() else 100)
        vy = st.number_input("Vertex Y (px)", min_value=0, max_value=h if 'h' in locals() else 1080, value=h//2 if 'h' in locals() else 100)

        col_add, col_undo = st.columns(2)
        with col_add:
            if st.button("📍 Add Vertex", use_container_width=True):
                st.session_state.current_polygon_vertices.append({"x": int(vx), "y": int(vy)})
                st.rerun()

        with col_undo:
            if st.button("↩️ Undo Point", use_container_width=True):
                if st.session_state.current_polygon_vertices:
                    st.session_state.current_polygon_vertices.pop()
                    st.rerun()

        st.write(f"Active Vertices: **{len(st.session_state.current_polygon_vertices)}** (Min 3 required)")

        if st.button("✅ Complete Region", type="primary", use_container_width=True):
            if len(st.session_state.current_polygon_vertices) < 3:
                st.error("At least 3 vertices are required to form a polygon!")
            else:
                if filename not in st.session_state.sem_annotations:
                    st.session_state.sem_annotations[filename] = {"polygons": []}

                st.session_state.sem_annotations[filename]["polygons"].append({
                    "class": selected_cls,
                    "points": st.session_state.current_polygon_vertices.copy()
                })

                save_semantic_png_mask(
                    datasets_folder, 
                    current_file.stem, 
                    (w, h), 
                    st.session_state.sem_annotations[filename]["polygons"], 
                    st.session_state.sem_classes
                )
                save_dataset_manifest(datasets_folder, st.session_state.sem_classes, st.session_state.sem_annotations)

                st.session_state.current_polygon_vertices = []
                st.success(f"Semantic region added for `{selected_cls}`!")
                st.rerun()

        if filename in st.session_state.sem_annotations and st.session_state.sem_annotations[filename]["polygons"]:
            st.markdown("---")
            st.write("**Regions for Image:**")
            for i, p in enumerate(st.session_state.sem_annotations[filename]["polygons"]):
                st.text(f"#{i+1}: {p['class']} ({len(p['points'])} vertices)")

            if st.button("🗑️ Clear Regions for Image", use_container_width=True):
                st.session_state.sem_annotations[filename]["polygons"] = []
                
                mask_path = datasets_folder / f"{current_file.stem}_mask.png"
                if mask_path.exists():
                    mask_path.unlink()
                
                save_dataset_manifest(datasets_folder, st.session_state.sem_classes, st.session_state.sem_annotations)
                st.rerun()

        st.markdown("---")
        nav_prev, nav_next = st.columns(2)
        with nav_prev:
            if st.button("⬅ Previous", use_container_width=True):
                if st.session_state.sem_image_index > 0:
                    st.session_state.sem_image_index -= 1
                    st.session_state.current_polygon_vertices = []
                    st.rerun()

        with nav_next:
            if st.button("Next ➡️", use_container_width=True):
                if st.session_state.sem_image_index < total_imgs - 1:
                    st.session_state.sem_image_index += 1
                    st.session_state.current_polygon_vertices = []
                    st.rerun()


# Standard alias exports expected by your host app
render_sem_tab = sem_tab
segmentation_tab = sem_tab
render_segmentation_tab = sem_tab

if __name__ == "__main__":
    sem_tab()
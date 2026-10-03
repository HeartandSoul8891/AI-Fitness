import json
import os
import numpy as np
from pathlib import Path
import streamlit as st
from PIL import Image, ImageDraw

from scripts.settings.settings_script import load_settings


def scan_images_in_dir(target_dir: Path):
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if not target_dir.exists():
        return []
    return [
        p for p in target_dir.iterdir() 
        if p.is_file() and p.suffix.lower() in valid_exts and not (p.name.endswith("_mask.png") or p.stem.endswith("-edit"))
    ]


def get_sem_models_dir() -> Path:
    settings = load_settings()
    seg_folder = st.session_state.get(
        "settings_ultralytics_segm_folder",
        settings.get("ultralytics_segm_folder", "models/ultralytics/segm")
    )
    return Path(seg_folder).expanduser()


def scan_sem_models(models_folder: Path):
    if not models_folder.exists():
        return []
    return list(models_folder.glob("*.pt"))


def save_semantic_png_mask(dataset_folder: Path, filename_stem: str, image_size: tuple, polygons: list, classes: list):
    """Generates dense mask PNG for YOLO26 segmentation training."""
    w, h = image_size
    mask_array = np.zeros((h, w), dtype=np.uint8)

    for poly in polygons:
        cls_name = poly["class"]
        if cls_name not in classes:
            continue
        cls_id = classes.index(cls_name) + 1

        pts = [(int(p["x"]), int(p["y"])) for p in poly["points"]]
        if len(pts) >= 3:
            mask_img = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask_img)
            draw.polygon(pts, fill=cls_id)
            mask_array = np.maximum(mask_array, np.array(mask_img))

    mask_path = dataset_folder / f"{filename_stem}_mask.png"
    Image.fromarray(mask_array).save(mask_path)


def draw_sem_overlay(image: Image.Image, polygons: list) -> Image.Image:
    """Renders filled polygon regions over PIL Image."""
    annotated = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for poly in polygons:
        pts = [(p["x"], p["y"]) for p in poly["points"]]
        if len(pts) >= 3:
            draw.polygon(pts, fill=(0, 255, 128, 80), outline=(0, 255, 128, 255))
            draw.text((pts[0][0] + 5, pts[0][1] - 5), poly["class"], fill="yellow")

    return Image.alpha_composite(annotated, overlay).convert("RGB")


def save_rendered_sem_preview(dataset_folder: Path, img_file: Path, polygons: list) -> Path:
    """Saves rendered segmentation image to dataset_folder / filename-edit.jpeg."""
    out_path = dataset_folder / f"{img_file.stem}-edit.jpeg"
    try:
        pil_img = Image.open(img_file)
        annotated = draw_sem_overlay(pil_img, polygons)
        annotated.save(out_path, format="JPEG", quality=90)
    except Exception as e:
        st.warning(f"Could not save segmentation preview artifact: {e}")
    return out_path


def save_dataset_manifest(dataset_folder: Path, classes: list, annotations: dict):
    manifest_path = dataset_folder / "sem_manifest.json"
    yaml_path = dataset_folder / "data.yaml"

    manifest_data = {"classes": classes, "annotations": annotations}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4)

    yaml_content = f"path: {dataset_folder.resolve()}\ntrain: .\nval: .\n\nnames:\n"
    for i, cls in enumerate(classes):
        yaml_content += f"  {i}: {cls}\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)


def run_sem_test_backend(model_path: str, image_files: list, conf_thresh: float = 0.25):
    """Runs backend segmentation model and saves generated masks and edit artifacts."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    test_previews = []

    for img_p in image_files[:6]:
        results = model.predict(source=str(img_p), conf=conf_thresh, verbose=False)
        polygons = []

        if results and results[0].masks is not None:
            masks = results[0].masks.xy
            cls_ids = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
            names = results[0].names

            for poly_pts, cid in zip(masks, cls_ids):
                cls_name = names.get(int(cid), f"class_{int(cid)}")
                pts_list = [{"x": int(pt[0]), "y": int(pt[1])} for pt in poly_pts]
                polygons.append({"class": cls_name, "points": pts_list})

        img_pil = Image.open(img_p)
        save_rendered_sem_preview(img_p.parent, img_p, polygons)
        out_edit = save_rendered_sem_preview(img_p.parent, img_p, polygons)
        test_previews.append((img_p.name, out_edit, len(polygons)))

    return test_previews


def sem_tab():
    st.title("✂️ YOLO26 Semantic Segmentation Tagger (`-sem`)")

    saved_settings = load_settings()
    global_datasets_dir = st.session_state.get(
        "settings_datasets_folder", 
        saved_settings.get("datasets_folder", "./datasets")
    )
    base_datasets_path = Path(global_datasets_dir).expanduser()

    models_dir = get_sem_models_dir()
    available_models = scan_sem_models(models_dir)

    st.markdown("### ⚙️ Dataset & Backend Model Settings")
    cfg_col1, cfg_col2 = st.columns(2)

    with cfg_col1:
        if not base_datasets_path.exists():
            st.error(f"Base datasets folder does not exist: `{base_datasets_path}`")
            return
        available_folders = [d.name for d in base_datasets_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        available_folders.sort()
        folder_options = ["(Root Datasets Directory)"] + available_folders
        selected_subfolder = st.selectbox("Select Dataset Directory", options=folder_options, index=0, key="sem_subfolder_select")
        datasets_folder = base_datasets_path if selected_subfolder == "(Root Datasets Directory)" else base_datasets_path / selected_subfolder

    with cfg_col2:
        st.info(f"📁 **Segmentation Models Directory:** `{models_dir}`")
        if available_models:
            model_options = [str(p) for p in available_models]
        else:
            model_options = ["yolo26n-seg.pt", "yolo11n-seg.pt", "yolov8n-seg.pt"]
        selected_model_path = st.selectbox("Backend Segmentation Model", options=model_options, key="sem_model_select")

    manifest_path = datasets_folder / "sem_manifest.json"

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

    # Class Management & Test Tools
    st.subheader("🏷️ Class Management & Model Test")
    col_add1, col_add2, col_test = st.columns([2, 1, 1])

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

    image_files = scan_images_in_dir(datasets_folder)

    with col_test:
        st.write("")
        st.write("")
        if st.button("🔍 Test Segmentation", use_container_width=True):
            if not image_files:
                st.warning("No images available to test.")
            else:
                with st.spinner("Running backend segmentation and generating edits..."):
                    test_outputs = run_sem_test_backend(selected_model_path, image_files)
                st.success("Test inference complete! Rendered preview saved as `filename-edit.jpeg`.")
                t_cols = st.columns(min(len(test_outputs), 3))
                for idx, (fname, out_img, cnt) in enumerate(test_outputs):
                    with t_cols[idx % 3]:
                        st.image(str(out_img), caption=f"{fname}\nRegions: {cnt}", use_container_width=True)

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
            image = Image.open(current_file)
            w, h = image.size
            polygons = st.session_state.sem_annotations.get(filename, {}).get("polygons", [])
            annotated_img = draw_sem_overlay(image, polygons)

            # Draw current in-progress points
            if st.session_state.current_polygon_vertices:
                draw_curr = ImageDraw.Draw(annotated_img)
                curr_pts = [(p["x"], p["y"]) for p in st.session_state.current_polygon_vertices]
                for pt in curr_pts:
                    draw_curr.ellipse((pt[0]-4, pt[1]-4, pt[0]+4, pt[1]+4), fill="red", outline="white")
                if len(curr_pts) >= 2:
                    draw_curr.line(curr_pts, fill="cyan", width=2)

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

        if st.button("✅ Complete Region & Save Preview", type="primary", use_container_width=True):
            if len(st.session_state.current_polygon_vertices) < 3:
                st.error("At least 3 vertices are required to form a polygon!")
            else:
                if filename not in st.session_state.sem_annotations:
                    st.session_state.sem_annotations[filename] = {"polygons": []}

                st.session_state.sem_annotations[filename]["polygons"].append({
                    "class": selected_cls,
                    "points": st.session_state.current_polygon_vertices.copy()
                })

                polygons = st.session_state.sem_annotations[filename]["polygons"]
                save_semantic_png_mask(datasets_folder, current_file.stem, (w, h), polygons, st.session_state.sem_classes)
                out_edit = save_rendered_sem_preview(datasets_folder, current_file, polygons)
                save_dataset_manifest(datasets_folder, st.session_state.sem_classes, st.session_state.sem_annotations)

                st.session_state.current_polygon_vertices = []
                st.toast(f"Saved tag & generated artifact `{out_edit.name}`", icon="💾")
                st.rerun()

        if filename in st.session_state.sem_annotations and st.session_state.sem_annotations[filename]["polygons"]:
            st.markdown("---")
            if st.button("🗑️ Clear Regions for Image", use_container_width=True):
                st.session_state.sem_annotations[filename]["polygons"] = []
                mask_path = datasets_folder / f"{current_file.stem}_mask.png"
                edit_path = datasets_folder / f"{current_file.stem}-edit.jpeg"
                if mask_path.exists(): mask_path.unlink()
                if edit_path.exists(): edit_path.unlink()
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


render_sem_tab = sem_tab
segmentation_tab = sem_tab
render_segmentation_tab = sem_tab

if __name__ == "__main__":
    sem_tab()
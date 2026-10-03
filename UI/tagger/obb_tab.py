import json
import math
import os
import sys
from pathlib import Path
import streamlit as st

# Safe resolution of project root into sys.path
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent if FILE_PATH.parent.name == "tabs" else FILE_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe fallback for settings loader
def get_app_settings():
    try:
        from scripts.settings.settings_script import load_settings
        return load_settings()
    except Exception:
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


def get_obb_models_dir() -> Path:
    settings = get_app_settings()
    obb_folder = st.session_state.get(
        "settings_ultralytics_obb_folder",
        settings.get("ultralytics_obb_folder", "models/ultralytics/obb")
    )
    return Path(obb_folder).expanduser()


def scan_obb_models(models_folder: Path):
    if not models_folder.exists():
        return []
    return list(models_folder.glob("*.pt"))


def compute_obb_corners(cx: float, cy: float, w: float, h: float, angle_deg: float):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    dx, dy = w / 2.0, h / 2.0
    local_corners = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    corners = []
    for lx, ly in local_corners:
        rx = lx * cos_a - ly * sin_a + cx
        ry = lx * sin_a + ly * cos_a + cy
        corners.append((rx, ry))
    return corners


def save_yolo_obb_txt(dataset_folder: Path, filename_stem: str, image_size: tuple, boxes: list, classes: list):
    img_w, img_h = image_size
    txt_path = dataset_folder / f"{filename_stem}.txt"
    lines = []

    for b in boxes:
        cls_name = b["class"]
        if cls_name not in classes:
            continue
        cls_id = classes.index(cls_name)

        corners = compute_obb_corners(b["cx"], b["cy"], b["w"], b["h"], b["angle"])
        norm_coords = []
        for px, py in corners:
            nx = max(0.0, min(1.0, px / img_w))
            ny = max(0.0, min(1.0, py / img_h))
            norm_coords.extend([f"{nx:.6f}", f"{ny:.6f}"])

        lines.append(f"{cls_id} " + " ".join(norm_coords))

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def draw_obb_overlay(pil_image, boxes: list):
    from PIL import Image, ImageDraw

    annotated = pil_image.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for box in boxes:
        cx, cy, bw, bh, angle, label = box["cx"], box["cy"], box["w"], box["h"], box["angle"], box["class"]
        corners = compute_obb_corners(cx, cy, bw, bh, angle)
        draw.polygon(corners, outline=(0, 255, 128, 255), fill=(0, 255, 128, 60), width=3)
        front_mid = ((corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2)
        draw.line([(cx, cy), front_mid], fill=(255, 0, 0, 255), width=2)
        draw.text((corners[0][0] + 5, corners[0][1] - 10), f"{label} ({angle}°)", fill="yellow")

    return Image.alpha_composite(annotated, overlay).convert("RGB")


def save_rendered_obb_preview(dataset_folder: Path, img_file: Path, boxes: list) -> Path:
    from PIL import Image

    out_path = dataset_folder / f"{img_file.stem}-edit.jpeg"
    try:
        pil_img = Image.open(img_file)
        annotated = draw_obb_overlay(pil_img, boxes)
        annotated.save(out_path, format="JPEG", quality=90)
    except Exception as e:
        st.warning(f"Failed to save rendered OBB artifact: {e}")
    return out_path


def save_dataset_manifest(dataset_folder: Path, classes: list, annotations: dict):
    manifest_path = dataset_folder / "obb_manifest.json"
    yaml_path = dataset_folder / "data.yaml"

    manifest_data = {"classes": classes, "annotations": annotations}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4)

    yaml_content = f"path: {dataset_folder.resolve()}\ntrain: .\nval: .\n\nnames:\n"
    for i, cls in enumerate(classes):
        yaml_content += f"  {i}: {cls}\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)


def run_obb_test_inference(model_path: str, image_files: list, conf_thresh: float = 0.25):
    from ultralytics import YOLO

    model = YOLO(model_path)
    test_previews = []

    for img_p in image_files[:6]:
        results = model.predict(source=str(img_p), conf=conf_thresh, verbose=False)
        boxes = []
        if results and results[0].obb is not None:
            res_obb = results[0].obb
            xywhr = res_obb.xywhr.cpu().numpy() if res_obb.xywhr is not None else []
            cls_ids = res_obb.cls.cpu().numpy() if res_obb.cls is not None else []
            names = results[0].names

            for (cx, cy, w, h, r), cid in zip(xywhr, cls_ids):
                angle_deg = math.degrees(float(r))
                cls_name = names.get(int(cid), f"class_{int(cid)}")
                boxes.append({
                    "class": cls_name,
                    "cx": float(cx),
                    "cy": float(cy),
                    "w": float(w),
                    "h": float(h),
                    "angle": float(angle_deg)
                })

        out_edited = save_rendered_obb_preview(img_p.parent, img_p, boxes)
        test_previews.append((img_p.name, out_edited, len(boxes)))

    return test_previews


def obb_tab():
    from PIL import Image

    st.title("🔄 YOLO Oriented Bounding Box (OBB) Tagger")

    saved_settings = get_app_settings()
    global_datasets_dir = st.session_state.get(
        "settings_datasets_folder", 
        saved_settings.get("datasets_folder", "./datasets")
    )
    base_datasets_path = Path(global_datasets_dir).expanduser()

    models_dir = get_obb_models_dir()
    available_models = scan_obb_models(models_dir)

    st.markdown("### ⚙️ Dataset & Backend Model Settings")
    cfg_col1, cfg_col2 = st.columns(2)

    with cfg_col1:
        if not base_datasets_path.exists():
            st.error(f"Base datasets folder does not exist: `{base_datasets_path}`")
            return
        available_folders = [d.name for d in base_datasets_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        available_folders.sort()
        folder_options = ["(Root Datasets Directory)"] + available_folders
        selected_subfolder = st.selectbox("Select Dataset Directory", options=folder_options, index=0, key="obb_subfolder_select")
        datasets_folder = base_datasets_path if selected_subfolder == "(Root Datasets Directory)" else base_datasets_path / selected_subfolder

    with cfg_col2:
        st.info(f"📁 **OBB Models Folder:** `{models_dir}`")
        if available_models:
            model_options = [str(p) for p in available_models]
        else:
            model_options = ["yolo26n-obb.pt", "yolo11n-obb.pt", "yolov8n-obb.pt"]
        selected_model_path = st.selectbox("Backend OBB Model", options=model_options, key="obb_model_select")

    manifest_path = datasets_folder / "obb_manifest.json"

    if "obb_dataset_folder" not in st.session_state or st.session_state.obb_dataset_folder != str(datasets_folder):
        st.session_state.obb_dataset_folder = str(datasets_folder)
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                st.session_state.obb_classes = manifest.get("classes", ["vehicle", "ship", "building"])
                st.session_state.obb_annotations = manifest.get("annotations", {})
            except Exception:
                st.session_state.obb_classes = ["vehicle", "ship", "building"]
                st.session_state.obb_annotations = {}
        else:
            st.session_state.obb_classes = ["vehicle", "ship", "building"]
            st.session_state.obb_annotations = {}
        st.session_state.obb_image_index = 0

    st.markdown("---")

    # Class Management & Test Tools
    st.subheader("🏷️ Class Management & Model Test")
    col_add1, col_add2, col_test = st.columns([2, 1, 1])

    with col_add1:
        new_class = st.text_input("New Class Label", key="obb_new_cls_input", placeholder="e.g. vehicle, ship, plane")
    with col_add2:
        st.write("")
        st.write("")
        if st.button("➕ Add Class", use_container_width=True) and new_class:
            cls_clean = new_class.strip().lower().replace(" ", "_")
            if cls_clean and cls_clean not in st.session_state.obb_classes:
                st.session_state.obb_classes.append(cls_clean)
                save_dataset_manifest(datasets_folder, st.session_state.obb_classes, st.session_state.obb_annotations)
                st.success(f"Added class: `{cls_clean}`")
                st.rerun()

    image_files = scan_images_in_dir(datasets_folder)

    with col_test:
        st.write("")
        st.write("")
        if st.button("🔍 Test OBB Inference", use_container_width=True):
            if not image_files:
                st.warning("No images found to test.")
            else:
                with st.spinner("Executing OBB backend inference & rendering outputs..."):
                    test_outputs = run_obb_test_inference(selected_model_path, image_files)
                st.success("OBB Test complete! Saved rendered previews as `filename-edit.jpeg`.")
                t_cols = st.columns(min(len(test_outputs), 3))
                for idx, (fname, out_img, cnt) in enumerate(test_outputs):
                    with t_cols[idx % 3]:
                        st.image(str(out_img), caption=f"{fname}\nDetected: {cnt} OBB(s)", use_container_width=True)

    if not image_files:
        st.warning(f"No valid image files found in `{datasets_folder}`.")
        return

    total_imgs = len(image_files)
    if st.session_state.get("obb_image_index", 0) >= total_imgs:
        st.session_state.obb_image_index = total_imgs - 1
    elif st.session_state.get("obb_image_index", 0) < 0:
        st.session_state.obb_image_index = 0

    idx = st.session_state.obb_image_index
    current_file = image_files[idx]
    filename = current_file.name

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Images", total_imgs)
    col2.metric("Tagged Images", len([k for k, v in st.session_state.obb_annotations.items() if v.get("boxes")]))
    col3.metric("Progress", f"{idx + 1} / {total_imgs}")

    st.progress((idx + 1) / total_imgs)

    img_col, tag_col = st.columns([2, 1])

    with img_col:
        st.subheader(f"🖼️ `{filename}`")
        try:
            image = Image.open(current_file)
            w, h = image.size
            existing_boxes = st.session_state.obb_annotations.get(filename, {}).get("boxes", [])
            annotated_img = draw_obb_overlay(image, existing_boxes)
            st.image(annotated_img, use_container_width=True)
            st.caption(f"Resolution: {w}x{h} px")
        except Exception as e:
            st.error(f"Error rendering image: {e}")

    with tag_col:
        st.subheader("Add Rotated Box (OBB)")
        selected_cls = st.selectbox("Select Class", options=st.session_state.obb_classes)

        cx_val = st.number_input("Center X (px)", min_value=0, max_value=w if 'w' in locals() else 1920, value=w//2 if 'w' in locals() else 100)
        cy_val = st.number_input("Center Y (px)", min_value=0, max_value=h if 'h' in locals() else 1080, value=h//2 if 'h' in locals() else 100)
        w_val = st.number_input("Width (px)", min_value=1, max_value=w if 'w' in locals() else 1920, value=100)
        h_val = st.number_input("Height (px)", min_value=1, max_value=h if 'h' in locals() else 1080, value=50)
        angle_val = st.slider("Rotation Angle (°)", min_value=-180, max_value=180, value=0, step=1)

        if st.button("➕ Save OBB & Render Preview", type="primary", use_container_width=True):
            if filename not in st.session_state.obb_annotations:
                st.session_state.obb_annotations[filename] = {"boxes": []}

            st.session_state.obb_annotations[filename]["boxes"].append({
                "class": selected_cls,
                "cx": int(cx_val),
                "cy": int(cy_val),
                "w": int(w_val),
                "h": int(h_val),
                "angle": float(angle_val)
            })

            current_boxes = st.session_state.obb_annotations[filename]["boxes"]

            save_yolo_obb_txt(datasets_folder, current_file.stem, (w, h), current_boxes, st.session_state.obb_classes)
            out_preview = save_rendered_obb_preview(datasets_folder, current_file, current_boxes)
            save_dataset_manifest(datasets_folder, st.session_state.obb_classes, st.session_state.obb_annotations)

            st.toast(f"Saved tags & rendered preview `{out_preview.name}`", icon="💾")
            st.rerun()

        if filename in st.session_state.obb_annotations and st.session_state.obb_annotations[filename]["boxes"]:
            st.markdown("---")
            if st.button("🗑️ Clear Boxes for Image", use_container_width=True):
                st.session_state.obb_annotations[filename]["boxes"] = []
                txt_path = datasets_folder / f"{current_file.stem}.txt"
                edit_path = datasets_folder / f"{current_file.stem}-edit.jpeg"
                if txt_path.exists(): txt_path.unlink()
                if edit_path.exists(): edit_path.unlink()
                save_dataset_manifest(datasets_folder, st.session_state.obb_classes, st.session_state.obb_annotations)
                st.rerun()

        st.markdown("---")
        nav_prev, nav_next = st.columns(2)
        with nav_prev:
            if st.button("⬅ Previous", use_container_width=True):
                if st.session_state.obb_image_index > 0:
                    st.session_state.obb_image_index -= 1
                    st.rerun()

        with nav_next:
            if st.button("Next ➡️", use_container_width=True):
                if st.session_state.obb_image_index < total_imgs - 1:
                    st.session_state.obb_image_index += 1
                    st.rerun()


# Universal Entry Points to handle any Streamlit dynamic tab loader strategy
def main():
    obb_tab()

def app():
    obb_tab()

def show():
    obb_tab()

def render():
    obb_tab()

render_tab = obb_tab
render_obb_tab = obb_tab
obb_tagger_tab = obb_tab

if __name__ == "__main__":
    obb_tab()
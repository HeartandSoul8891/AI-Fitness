import os
import json
import math
from pathlib import Path
import streamlit as st
from PIL import Image, ImageDraw

from scripts.settings.settings_script import load_settings


def scan_images_in_dir(target_dir: Path):
    """Scans for valid image files inside the specified target directory."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if not target_dir.exists():
        return []
    return [p for p in target_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]


def compute_obb_corners(cx: float, cy: float, w: float, h: float, angle_deg: float):
    """Calculates the 4 corner coordinates (x, y) of an oriented bounding box from center and angle."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Unrotated local offsets relative to center
    dx = w / 2.0
    dy = h / 2.0

    local_corners = [
        (-dx, -dy),  # Top-Left
        (dx, -dy),   # Top-Right
        (dx, dy),    # Bottom-Right
        (-dx, dy)    # Bottom-Left
    ]

    corners = []
    for lx, ly in local_corners:
        rx = lx * cos_a - ly * sin_a + cx
        ry = lx * sin_a + ly * cos_a + cy
        corners.append((rx, ry))

    return corners


def save_yolo_obb_txt(dataset_folder: Path, filename_stem: str, image_size: tuple, boxes: list, classes: list):
    """Exports normalized 4-corner polygon points for YOLO OBB training (.txt)."""
    img_w, img_h = image_size
    txt_path = dataset_folder / f"{filename_stem}.txt"
    lines = []

    for b in boxes:
        cls_name = b["class"]
        if cls_name not in classes:
            continue
        cls_id = classes.index(cls_name)

        corners = compute_obb_corners(b["cx"], b["cy"], b["w"], b["h"], b["angle"])

        # Normalize 4 corners to [0.0, 1.0]
        norm_coords = []
        for px, py in corners:
            nx = max(0.0, min(1.0, px / img_w))
            ny = max(0.0, min(1.0, py / img_h))
            norm_coords.extend([f"{nx:.6f}", f"{ny:.6f}"])

        lines.append(f"{cls_id} " + " ".join(norm_coords))

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_dataset_manifest(dataset_folder: Path, classes: list, annotations: dict):
    """Saves obb_manifest.json and updates data.yaml for YOLO OBB dataset format."""
    manifest_path = dataset_folder / "obb_manifest.json"
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


def obb_tab():
    st.title("🔄 YOLO Oriented Bounding Box (OBB) Tagger")

    # Load system settings
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

    # Folder Selector
    available_folders = [d.name for d in base_datasets_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    available_folders.sort()
    
    folder_options = ["(Root Datasets Directory)"] + available_folders
    selected_subfolder = st.selectbox("Select Dataset Directory", options=folder_options, index=0, key="obb_subfolder_select")

    datasets_folder = base_datasets_path if selected_subfolder == "(Root Datasets Directory)" else base_datasets_path / selected_subfolder

    source_dir_str = st.text_input("Dataset Directory Path", value=str(datasets_folder), key="obb_source_dir_input")
    datasets_folder = Path(source_dir_str).expanduser()

    if not datasets_folder.exists():
        st.error(f"Selected dataset directory does not exist:\n\n`{datasets_folder}`")
        return

    manifest_path = datasets_folder / "obb_manifest.json"

    # Persistent Session State Setup
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

    # Inlined Class Management
    st.subheader("🏷️ Class Management")
    col_add1, col_add2 = st.columns([3, 1])
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

    if st.session_state.obb_classes:
        st.caption("Active Classes:")
        st.write(" ".join([f"`🏷️ {c}`" for c in st.session_state.obb_classes]))

    st.markdown("---")

    image_files = scan_images_in_dir(datasets_folder)

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
            image = Image.open(current_file).convert("RGBA")
            w, h = image.size

            overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)

            existing_data = st.session_state.obb_annotations.get(filename, {})
            boxes = existing_data.get("boxes", [])

            for box in boxes:
                cx, cy = box["cx"], box["cy"]
                bw, bh = box["w"], box["h"]
                angle = box["angle"]
                label = box["class"]

                corners = compute_obb_corners(cx, cy, bw, bh, angle)
                draw.polygon(corners, outline=(0, 255, 128, 255), fill=(0, 255, 128, 60), width=3)
                
                # Draw front-facing orientation indicator line
                front_mid = ((corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2)
                draw.line([(cx, cy), front_mid], fill=(255, 0, 0, 255), width=2)
                
                draw.text((corners[0][0] + 5, corners[0][1] - 10), f"{label} ({angle}°)", fill="yellow")

            annotated_img = Image.alpha_composite(image, overlay)
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

        if st.button("➕ Add OBB Box", type="primary", use_container_width=True):
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

            # Save dataset manifest and normalized YOLO .txt file
            save_yolo_obb_txt(
                datasets_folder, 
                current_file.stem, 
                (w, h), 
                st.session_state.obb_annotations[filename]["boxes"], 
                st.session_state.obb_classes
            )
            save_dataset_manifest(datasets_folder, st.session_state.obb_classes, st.session_state.obb_annotations)

            st.success(f"Added `{selected_cls}` box at {angle_val}°")
            st.rerun()

        if filename in st.session_state.obb_annotations and st.session_state.obb_annotations[filename]["boxes"]:
            st.markdown("---")
            st.write("**Current OBBs:**")
            for i, b in enumerate(st.session_state.obb_annotations[filename]["boxes"]):
                st.text(f"#{i+1}: {b['class']} | Pos: ({b['cx']},{b['cy']}) | Size: {b['w']}x{b['h']} | {b['angle']}°")

            if st.button("🗑️ Clear Boxes for Image", use_container_width=True):
                st.session_state.obb_annotations[filename]["boxes"] = []
                
                txt_path = datasets_folder / f"{current_file.stem}.txt"
                if txt_path.exists():
                    txt_path.unlink()

                save_dataset_manifest(datasets_folder, st.session_state.obb_classes, st.session_state.obb_annotations)
                st.rerun()

        st.markdown("---")
        nav_prev, nav_next = st.columns(2)
        with nav_prev:
            if st.button("⬅️️ Previous", use_container_width=True):
                if st.session_state.obb_image_index > 0:
                    st.session_state.obb_image_index -= 1
                    st.rerun()

        with nav_next:
            if st.button("Next ➡️", use_container_width=True):
                if st.session_state.obb_image_index < total_imgs - 1:
                    st.session_state.obb_image_index += 1
                    st.rerun()


# Standard alias exports
render_obb_tab = obb_tab
obb_tagger_tab = obb_tab

if __name__ == "__main__":
    obb_tab()
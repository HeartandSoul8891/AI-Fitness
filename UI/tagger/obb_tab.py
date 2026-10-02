import streamlit as st
from PIL import Image, ImageDraw
from pathlib import Path
from scripts.tagger.obb_script import (
    load_obb_manifest,
    save_obb_manifest,
    scan_obb_dataset,
    get_obb_corners
)

st.set_page_config(
    page_title="YOLO OBB Tagger",
    page_icon="🔄",
    layout="wide"
)

st.title("🔄 YOLO Oriented Bounding Box (OBB) Tagger")

# ==========================================
# CONFIG & CLASS MANAGEMENT (MAIN PAGE)
# ==========================================
with st.expander("⚙️ File Paths & Class Management", expanded=False):
    cfg_col1, cfg_col2 = st.columns(2)
    with cfg_col1:
        st.subheader("📁 File Paths")
        source_dir = st.text_input("Source Directory (Raw Images)", value="./raw_images")
        manifest_path = st.text_input("Manifest JSON Path", value="./obb_manifest.json")

    with cfg_col2:
        st.subheader("🏷️ Class Management")
        new_class = st.text_input("Add Class Name")
        if st.button("➕ Add Class", use_container_width=True) and new_class:
            cls_clean = new_class.strip().lower()
            if cls_clean and cls_clean not in st.session_state.classes:
                st.session_state.classes.append(cls_clean)
                save_obb_manifest(manifest_path, st.session_state.classes, st.session_state.annotations)
                st.success(f"Added: `{cls_clean}`")
                st.rerun()

        if st.session_state.get("classes"):
            cls_to_remove = st.selectbox("Remove Class", options=[""] + st.session_state.classes)
            if st.button("🗑️ Delete Class", use_container_width=True) and cls_to_remove:
                st.session_state.classes.remove(cls_to_remove)
                save_obb_manifest(manifest_path, st.session_state.classes, st.session_state.annotations)
                st.warning(f"Deleted: `{cls_to_remove}`")
                st.rerun()

# Load existing manifest on initial run or path change
if "manifest_path" not in st.session_state or st.session_state.manifest_path != manifest_path:
    manifest = load_obb_manifest(manifest_path)
    st.session_state.manifest_path = manifest_path
    st.session_state.classes = manifest.get("classes", ["vehicle", "ship", "building"])
    st.session_state.annotations = manifest.get("annotations", {})
    st.session_state.image_index = 0

# ==========================================
# MAIN INTERFACE
# ==========================================
image_files = scan_obb_dataset(source_dir)

if not image_files:
    st.warning(f"No valid images found in `{source_dir}`. Check directory path.")
else:
    total_imgs = len(image_files)
    
    if st.session_state.image_index >= total_imgs:
        st.session_state.image_index = total_imgs - 1
    elif st.session_state.image_index < 0:
        st.session_state.image_index = 0

    idx = st.session_state.image_index
    current_file = image_files[idx]
    filename = current_file.name

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Images", total_imgs)
    col2.metric("Tagged Images", len([k for k, v in st.session_state.annotations.items() if v.get("boxes")]))
    col3.metric("Progress", f"{idx + 1} / {total_imgs}")

    st.progress((idx + 1) / total_imgs)

    img_col, tag_col = st.columns([2, 1])

    with img_col:
        st.subheader(f"🖼️ `{filename}`")
        try:
            image = Image.open(current_file).convert("RGB")
            w, h = image.size

            annotated_img = image.copy()
            draw = ImageDraw.Draw(annotated_img)

            existing_data = st.session_state.annotations.get(filename, {})
            boxes = existing_data.get("boxes", [])

            for box in boxes:
                cx, cy = box["cx"], box["cy"]
                bw, bh = box["w"], box["h"]
                angle = box["angle"]
                label = box["class"]

                corners = get_obb_corners(cx, cy, bw, bh, angle)
                polygon = [(px, py) for px, py in corners]

                draw.polygon(polygon, outline="lime", width=3)
                draw.text((corners[0][0] + 5, corners[0][1] - 10), f"{label} ({angle}°)", fill="yellow")

            st.image(annotated_img, use_container_width=True)
            st.caption(f"Resolution: {w}x{h} px")
        except Exception as e:
            st.error(f"Error rendering image: {e}")

    with tag_col:
        st.subheader("Add Rotated Box (OBB)")

        selected_cls = st.selectbox("Select Class", options=st.session_state.classes)

        cx_val = st.number_input("Center X (px)", min_value=0, max_value=w if 'w' in locals() else 1920, value=w//2 if 'w' in locals() else 100)
        cy_val = st.number_input("Center Y (px)", min_value=0, max_value=h if 'h' in locals() else 1080, value=h//2 if 'h' in locals() else 100)
        w_val = st.number_input("Width (px)", min_value=1, max_value=w if 'w' in locals() else 1920, value=100)
        h_val = st.number_input("Height (px)", min_value=1, max_value=h if 'h' in locals() else 1080, value=50)
        angle_val = st.slider("Rotation Angle (°)", min_value=-180, max_value=180, value=0, step=1)

        if st.button("➕ Add OBB Box", use_container_width=True):
            if filename not in st.session_state.annotations:
                st.session_state.annotations[filename] = {"boxes": []}

            st.session_state.annotations[filename]["boxes"].append({
                "class": selected_cls,
                "cx": int(cx_val),
                "cy": int(cy_val),
                "w": int(w_val),
                "h": int(h_val),
                "angle": float(angle_val)
            })

            save_obb_manifest(manifest_path, st.session_state.classes, st.session_state.annotations)
            st.success(f"Added `{selected_cls}` box at {angle_val}°")
            st.rerun()

        if filename in st.session_state.annotations and st.session_state.annotations[filename]["boxes"]:
            st.markdown("---")
            st.write("**Current OBBs:**")
            for i, b in enumerate(st.session_state.annotations[filename]["boxes"]):
                st.text(f"#{i+1}: {b['class']} | Pos: ({b['cx']},{b['cy']}) | Size: {b['w']}x{b['h']} | {b['angle']}°")

            if st.button("🗑️️ Clear Boxes for Image", use_container_width=True):
                st.session_state.annotations[filename]["boxes"] = []
                save_obb_manifest(manifest_path, st.session_state.classes, st.session_state.annotations)
                st.rerun()

        st.markdown("---")
        nav_prev, nav_next = st.columns(2)
        with nav_prev:
            if st.button("⬅️ Previous", use_container_width=True):
                if st.session_state.image_index > 0:
                    st.session_state.image_index -= 1
                    st.rerun()

        with nav_next:
            if st.button("Next ➡️️", use_container_width=True):
                if st.session_state.image_index < total_imgs - 1:
                    st.session_state.image_index += 1
                    st.rerun()
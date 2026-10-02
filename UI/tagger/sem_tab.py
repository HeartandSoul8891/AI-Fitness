import streamlit as st
from PIL import Image, ImageDraw
from pathlib import Path
from scripts.tagger.sem_script import (
    load_sem_manifest,
    save_sem_manifest,
    scan_sem_dataset,
    normalize_polygon
)

st.set_page_config(
    page_title="YOLO Segmentation Tagger",
    page_icon="✂️",
    layout="wide"
)

st.title("✂ YOLO Semantic / Instance Segmentation Tagger")

# ==========================================
# CONFIG & CLASS MANAGEMENT (MAIN PAGE)
# ==========================================
with st.expander("⚙️ Directory & Class Settings", expanded=False):
    cfg_col1, cfg_col2 = st.columns(2)
    with cfg_col1:
        st.subheader("📁 Directory Settings")
        source_dir = st.text_input("Source Directory (Raw Images)", value="./raw_images")
        manifest_path = st.text_input("Manifest JSON Path", value="./sem_manifest.json")

    with cfg_col2:
        st.subheader("🏷️ Class Management")
        new_class = st.text_input("Add Class Name")
        if st.button("➕ Add Class", use_container_width=True) and new_class:
            cls_clean = new_class.strip().lower()
            if cls_clean and cls_clean not in st.session_state.classes:
                st.session_state.classes.append(cls_clean)
                save_sem_manifest(manifest_path, st.session_state.classes, st.session_state.annotations)
                st.success(f"Added: `{cls_clean}`")
                st.rerun()

        if st.session_state.get("classes"):
            cls_to_remove = st.selectbox("Remove Class", options=[""] + st.session_state.classes)
            if st.button("🗑️ Delete Class", use_container_width=True) and cls_to_remove:
                st.session_state.classes.remove(cls_to_remove)
                save_sem_manifest(manifest_path, st.session_state.classes, st.session_state.annotations)
                st.warning(f"Deleted: `{cls_to_remove}`")
                st.rerun()

# Initialize session state
if "manifest_path" not in st.session_state or st.session_state.manifest_path != manifest_path:
    manifest = load_sem_manifest(manifest_path)
    st.session_state.manifest_path = manifest_path
    st.session_state.classes = manifest.get("classes", ["object", "background"])
    st.session_state.annotations = manifest.get("annotations", {})
    st.session_state.image_index = 0

# ==========================================
# MAIN INTERFACE
# ==========================================
image_files = scan_sem_dataset(source_dir)

if not image_files:
    st.warning(f"No valid images found in `{source_dir}`. Please verify directory path.")
else:
    total_imgs = len(image_files)

    # Boundary checks
    if st.session_state.image_index >= total_imgs:
        st.session_state.image_index = total_imgs - 1
    elif st.session_state.image_index < 0:
        st.session_state.image_index = 0

    idx = st.session_state.image_index
    current_file = image_files[idx]
    filename = current_file.name

    # Header Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Images", total_imgs)
    col2.metric("Tagged Images", len([k for k, v in st.session_state.annotations.items() if v.get("polygons")]))
    col3.metric("Progress", f"{idx + 1} / {total_imgs}")

    st.progress((idx + 1) / total_imgs)

    img_col, tag_col = st.columns([2, 1])

    # Temporary vertex builder state
    if "current_polygon_vertices" not in st.session_state:
        st.session_state.current_polygon_vertices = []

    with img_col:
        st.subheader(f"🖼️ `{filename}`")
        try:
            image = Image.open(current_file).convert("RGBA")
            w, h = image.size

            overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)

            existing_data = st.session_state.annotations.get(filename, {})
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
        st.subheader("Add Polygon Segmentation Mask")

        selected_cls = st.selectbox("Select Class", options=st.session_state.classes)

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

        if st.button("✅ Complete Polygon", type="primary", use_container_width=True):
            if len(st.session_state.current_polygon_vertices) < 3:
                st.error("At least 3 vertices are required to form a polygon!")
            else:
                if filename not in st.session_state.annotations:
                    st.session_state.annotations[filename] = {"polygons": []}

                norm_pts = normalize_polygon(st.session_state.current_polygon_vertices, w, h)

                st.session_state.annotations[filename]["polygons"].append({
                    "class": selected_cls,
                    "points": st.session_state.current_polygon_vertices.copy(),
                    "normalized_points": norm_pts
                })

                st.session_state.current_polygon_vertices = []
                save_sem_manifest(manifest_path, st.session_state.classes, st.session_state.annotations)
                st.success(f"Polygon added for `{selected_cls}`!")
                st.rerun()

        if filename in st.session_state.annotations and st.session_state.annotations[filename]["polygons"]:
            st.markdown("---")
            st.write("**Polygons for Image:**")
            for i, p in enumerate(st.session_state.annotations[filename]["polygons"]):
                st.text(f"#{i+1}: {p['class']} ({len(p['points'])} vertices)")

            if st.button("🗑️ Clear Polygons for Image", use_container_width=True):
                st.session_state.annotations[filename]["polygons"] = []
                save_sem_manifest(manifest_path, st.session_state.classes, st.session_state.annotations)
                st.rerun()

        st.markdown("---")
        nav_prev, nav_next = st.columns(2)
        with nav_prev:
            if st.button("⬅️ Previous", use_container_width=True):
                if st.session_state.image_index > 0:
                    st.session_state.image_index -= 1
                    st.session_state.current_polygon_vertices = []
                    st.rerun()

        with nav_next:
            if st.button("Next ➡️", use_container_width=True):
                if st.session_state.image_index < total_imgs - 1:
                    st.session_state.image_index += 1
                    st.session_state.current_polygon_vertices = []
                    st.rerun()
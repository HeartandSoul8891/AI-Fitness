import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from scripts.tagger.dept_script import (
    load_depth_manifest,
    save_depth_manifest,
    scan_depth_dataset,
    export_depth_map_npy
)

st.set_page_config(
    page_title="YOLO26 Depth Tagger",
    page_icon="📏",
    layout="wide"
)

st.title("📏 YOLO26 Depth Estimation Tagger")

# ==========================================
# CONFIG & SETTINGS (MAIN PAGE)
# ==========================================
with st.expander("⚙️ Dataset Configuration & Depth Settings", expanded=False):
    cfg_col1, cfg_col2 = st.columns(2)
    with cfg_col1:
        st.subheader("📁 Dataset Directory Configuration")
        source_dir = st.text_input("RGB Images Source Directory", value="./raw_images")
        depth_out_dir = st.text_input("Depth Maps Output Directory (.npy)", value="./depth_maps")
        manifest_path = st.text_input("Manifest JSON Path", value="./depth_manifest.json")

    with cfg_col2:
        st.subheader("⚙️ Default Depth Settings")
        default_min_depth = st.number_input("Min Depth (meters)", value=0.5, step=0.1)
        default_max_depth = st.number_input("Max Depth (meters)", value=10.0, step=0.5)

# Initialize session state
if "manifest_path" not in st.session_state or st.session_state.manifest_path != manifest_path:
    manifest = load_depth_manifest(manifest_path)
    st.session_state.manifest_path = manifest_path
    st.session_state.annotations = manifest.get("annotations", {})
    st.session_state.image_index = 0

# ==========================================
# MAIN INTERFACE
# ==========================================
image_files = scan_depth_dataset(source_dir)

if not image_files:
    st.warning(f"No valid RGB images found in `{source_dir}`. Please verify the folder path.")
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
    col2.metric("Depth Maps Created", len(st.session_state.annotations))
    col3.metric("Progress", f"{idx + 1} / {total_imgs}")

    st.progress((idx + 1) / total_imgs)

    img_col, tag_col = st.columns([2, 1])

    with img_col:
        st.subheader(f"🖼️ RGB Frame: `{filename}`")
        try:
            image = Image.open(current_file).convert("RGB")
            w, h = image.size

            annotated_img = image.copy()
            draw = ImageDraw.Draw(annotated_img)

            existing_data = st.session_state.annotations.get(filename, {})
            points = existing_data.get("points", [])

            for pt in points:
                px, py, d = pt["x"], pt["y"], pt["depth"]
                radius = 6
                draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill="red", outline="white")
                draw.text((px + 8, py - 8), f"{d}m", fill="yellow")

            st.image(annotated_img, use_container_width=True)
            st.caption(f"Dimensions: {w}x{h} px")
        except Exception as e:
            st.error(f"Error loading image: {e}")

    with tag_col:
        st.subheader("Annotate Distance Points")
        st.write("Add depth anchor points (In Meters):")
        
        x_coord = st.number_input("Point X (px)", min_value=0, max_value=w if 'w' in locals() else 1920, value=w//2 if 'w' in locals() else 0)
        y_coord = st.number_input("Point Y (px)", min_value=0, max_value=h if 'h' in locals() else 1080, value=h//2 if 'h' in locals() else 0)
        depth_val = st.number_input("Distance / Depth (meters)", min_value=0.01, max_value=100.0, value=2.5, step=0.1)

        if st.button("➕ Add Depth Point", use_container_width=True):
            if filename not in st.session_state.annotations:
                st.session_state.annotations[filename] = {"points": []}
            
            st.session_state.annotations[filename]["points"].append({
                "x": int(x_coord),
                "y": int(y_coord),
                "depth": float(depth_val)
            })
            
            npy_path = export_depth_map_npy(
                depth_out_dir, 
                filename, 
                (w, h), 
                st.session_state.annotations[filename]["points"]
            )
            st.session_state.annotations[filename]["npy_file"] = npy_path
            
            save_depth_manifest(manifest_path, st.session_state.annotations)
            st.success(f"Added anchor: ({x_coord}, {y_coord}) = {depth_val}m")
            st.rerun()

        if filename in st.session_state.annotations and st.session_state.annotations[filename]["points"]:
            st.markdown("---")
            st.write("**Current Keypoints:**")
            for i, pt in enumerate(st.session_state.annotations[filename]["points"]):
                st.text(f"#{i+1}: ({pt['x']}, {pt['y']}) -> {pt['depth']}m")

            if st.button("🗑️ Clear All Points for Image", use_container_width=True):
                st.session_state.annotations[filename]["points"] = []
                export_depth_map_npy(depth_out_dir, filename, (w, h), [])
                save_depth_manifest(manifest_path, st.session_state.annotations)
                st.rerun()

        st.markdown("---")
        nav_prev, nav_next = st.columns(2)
        with nav_prev:
            if st.button("⬅️ Previous", use_container_width=True):
                if st.session_state.image_index > 0:
                    st.session_state.image_index -= 1
                    st.rerun()

        with nav_next:
            if st.button("Next ➡️", use_container_width=True):
                if st.session_state.image_index < total_imgs - 1:
                    st.session_state.image_index += 1
                    st.rerun()
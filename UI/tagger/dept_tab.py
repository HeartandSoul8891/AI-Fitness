import json
import os
import sys
from pathlib import Path
import numpy as np
import streamlit as st

# Safe resolution of project root into sys.path
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent if FILE_PATH.parent.name == "tabs" else FILE_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe fallback for settings script
def get_app_settings():
    try:
        from scripts.settings.settings_script import load_settings
        return load_settings()
    except Exception:
        return {}

# Safe fallbacks for depth backend functions
try:
    from scripts.tagger.dept_script import (
        export_depth_map_npy,
        load_depth_manifest,
        save_depth_manifest,
        scan_depth_dataset,
    )
except ImportError:
    def load_depth_manifest(manifest_path):
        p = Path(manifest_path)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"annotations": {}}

    def save_depth_manifest(manifest_path, annotations):
        p = Path(manifest_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"annotations": annotations}, f, indent=4)

    def scan_depth_dataset(source_dir):
        p = Path(source_dir)
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        if not p.exists():
            return []
        return [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in valid_exts and not f.stem.endswith("-edit")]

    def export_depth_map_npy(output_dir, filename, img_size, points):
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        npy_path = out_p / f"{Path(filename).stem}_depth.npy"
        w, h = img_size
        depth_map = np.zeros((h, w), dtype=np.float32)
        for pt in points:
            px, py, d = int(pt["x"]), int(pt["y"]), float(pt["depth"])
            if 0 <= px < w and 0 <= py < h:
                depth_map[py, px] = d
        np.save(npy_path, depth_map)
        return str(npy_path)


def dept_tab():
    from PIL import Image, ImageDraw

    saved_settings = get_app_settings()
    global_datasets_dir = st.session_state.get(
        "settings_datasets_folder", 
        saved_settings.get("datasets_folder", "./datasets")
    )

    # Session state initialization with dedicated dept_ prefix
    if "dept_manifest_path" not in st.session_state:
        st.session_state.dept_manifest_path = "./depth_manifest.json"

    if "dept_annotations" not in st.session_state:
        manifest = load_depth_manifest(st.session_state.dept_manifest_path)
        st.session_state.dept_annotations = manifest.get("annotations", {})
        st.session_state.dept_image_index = 0

    st.title("📏 YOLO Depth Estimation Tagger")

    # Dataset Configuration & Settings
    st.markdown("### ⚙️ Dataset Configuration & Depth Settings")
    cfg_col1, cfg_col2 = st.columns(2)

    with cfg_col1:
        st.subheader("📁 Dataset Directory Configuration")
        default_src = str(Path(global_datasets_dir) / "raw_images") if Path(global_datasets_dir).exists() else "./raw_images"
        source_dir = st.text_input("RGB Images Source Directory", value=default_src, key="dept_source_dir_input")
        depth_out_dir = st.text_input("Depth Maps Output Directory (.npy)", value="./depth_maps", key="dept_out_dir_input")
        manifest_path = st.text_input("Manifest JSON Path", value=st.session_state.dept_manifest_path, key="dept_manifest_path_input")

        # Reload manifest if path changed
        if manifest_path != st.session_state.dept_manifest_path:
            manifest = load_depth_manifest(manifest_path)
            st.session_state.dept_manifest_path = manifest_path
            st.session_state.dept_annotations = manifest.get("annotations", {})
            st.session_state.dept_image_index = 0
            st.rerun()

    with cfg_col2:
        st.subheader("⚙️ Default Depth Settings")
        default_min_depth = st.number_input("Min Depth (meters)", value=0.5, step=0.1, key="dept_min_val")
        default_max_depth = st.number_input("Max Depth (meters)", value=10.0, step=0.5, key="dept_max_val")

    st.markdown("---")

    # Main Interface
    image_files = scan_depth_dataset(source_dir)

    if not image_files:
        st.warning(f"No valid RGB images found in `{source_dir}`. Please verify the folder path.")
        return

    total_imgs = len(image_files)
    
    if st.session_state.get("dept_image_index", 0) >= total_imgs:
        st.session_state.dept_image_index = total_imgs - 1
    elif st.session_state.get("dept_image_index", 0) < 0:
        st.session_state.dept_image_index = 0

    idx = st.session_state.dept_image_index
    current_file = image_files[idx]
    filename = current_file.name

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Images", total_imgs)
    col2.metric("Depth Maps Created", len(st.session_state.dept_annotations))
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

            existing_data = st.session_state.dept_annotations.get(filename, {})
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
        
        x_coord = st.number_input("Point X (px)", min_value=0, max_value=w if 'w' in locals() else 1920, value=w//2 if 'w' in locals() else 0, key="dept_px_x")
        y_coord = st.number_input("Point Y (px)", min_value=0, max_value=h if 'h' in locals() else 1080, value=h//2 if 'h' in locals() else 0, key="dept_px_y")
        depth_val = st.number_input("Distance / Depth (meters)", min_value=0.01, max_value=100.0, value=2.5, step=0.1, key="dept_dist_val")

        if st.button("➕ Add Depth Point", use_container_width=True, key="dept_add_btn"):
            if filename not in st.session_state.dept_annotations:
                st.session_state.dept_annotations[filename] = {"points": []}
            
            st.session_state.dept_annotations[filename]["points"].append({
                "x": int(x_coord),
                "y": int(y_coord),
                "depth": float(depth_val)
            })
            
            npy_path = export_depth_map_npy(
                depth_out_dir, 
                filename, 
                (w, h), 
                st.session_state.dept_annotations[filename]["points"]
            )
            st.session_state.dept_annotations[filename]["npy_file"] = npy_path
            
            save_depth_manifest(st.session_state.dept_manifest_path, st.session_state.dept_annotations)
            st.success(f"Added anchor: ({x_coord}, {y_coord}) = {depth_val}m")
            st.rerun()

        if filename in st.session_state.dept_annotations and st.session_state.dept_annotations[filename]["points"]:
            st.markdown("---")
            st.write("**Current Keypoints:**")
            for i, pt in enumerate(st.session_state.dept_annotations[filename]["points"]):
                st.text(f"#{i+1}: ({pt['x']}, {pt['y']}) -> {pt['depth']}m")

            if st.button("🗑️ Clear All Points for Image", use_container_width=True, key="dept_clear_btn"):
                st.session_state.dept_annotations[filename]["points"] = []
                export_depth_map_npy(depth_out_dir, filename, (w, h), [])
                save_depth_manifest(st.session_state.dept_manifest_path, st.session_state.dept_annotations)
                st.rerun()

        st.markdown("---")
        nav_prev, nav_next = st.columns(2)
        with nav_prev:
            if st.button("⬅️ Previous", use_container_width=True, key="dept_prev_btn"):
                if st.session_state.dept_image_index > 0:
                    st.session_state.dept_image_index -= 1
                    st.rerun()

        with nav_next:
            if st.button("Next ➡", use_container_width=True, key="dept_next_btn"):
                if st.session_state.dept_image_index < total_imgs - 1:
                    st.session_state.dept_image_index += 1
                    st.rerun()

# Module Entry Point Aliases for main tab loaders
def main():
    dept_tab()

def app():
    dept_tab()

def show():
    dept_tab()

def render():
    dept_tab()

render_tab = dept_tab
render_dept_tab = dept_tab
depth_tab = dept_tab

if __name__ == "__main__":
    st.set_page_config(
        page_title="YOLO Depth Tagger",
        page_icon="📏",
        layout="wide"
    )
    dept_tab()
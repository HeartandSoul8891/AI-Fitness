import streamlit as st
from PIL import Image, ImageDraw
from pathlib import Path
from scripts.tagger.pose_script import (
    load_pose_manifest,
    save_pose_manifest,
    scan_pose_dataset,
    DEFAULT_KEYPOINT_NAMES,
    DEFAULT_SKELETON_LINKS
)

st.set_page_config(
    page_title="YOLO Pose Tagger",
    page_icon="🦴",
    layout="wide"
)

st.title("🦴 YOLO Pose Estimation Tagger")

# ==========================================
# CONFIG & CLASS MANAGEMENT (MAIN PAGE)
# ==========================================
with st.expander("⚙️ Directory Settings & Class Management", expanded=False):
    cfg_col1, cfg_col2 = st.columns(2)
    with cfg_col1:
        st.subheader("📁 Directory Settings")
        source_dir = st.text_input("Source Directory (Raw Images)", value="./raw_images")
        manifest_path = st.text_input("Manifest JSON Path", value="./pose_manifest.json")

    with cfg_col2:
        st.subheader("🏷️ Class Management")
        new_class = st.text_input("Add Class Name")
        if st.button("➕ Add Class", use_container_width=True) and new_class:
            cls_clean = new_class.strip().lower()
            if cls_clean and cls_clean not in st.session_state.classes:
                st.session_state.classes.append(cls_clean)
                save_pose_manifest(
                    manifest_path, 
                    st.session_state.classes, 
                    st.session_state.keypoint_names, 
                    st.session_state.skeleton, 
                    st.session_state.annotations
                )
                st.success(f"Added: `{cls_clean}`")
                st.rerun()

# Initialize session state
if "manifest_path" not in st.session_state or st.session_state.manifest_path != manifest_path:
    manifest = load_pose_manifest(manifest_path)
    st.session_state.manifest_path = manifest_path
    st.session_state.classes = manifest.get("classes", ["person"])
    st.session_state.keypoint_names = manifest.get("keypoint_names", DEFAULT_KEYPOINT_NAMES)
    st.session_state.skeleton = manifest.get("skeleton", DEFAULT_SKELETON_LINKS)
    st.session_state.annotations = manifest.get("annotations", {})
    st.session_state.image_index = 0

# ==========================================
# MAIN INTERFACE
# ==========================================
image_files = scan_pose_dataset(source_dir)

if not image_files:
    st.warning(f"No valid images found in `{source_dir}`. Please verify directory path.")
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
    col2.metric("Tagged Images", len([k for k, v in st.session_state.annotations.items() if v.get("poses")]))
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
            poses = existing_data.get("poses", [])

            for pose in poses:
                kpts = pose["keypoints"]
                
                for link in st.session_state.skeleton:
                    idx1, idx2 = link
                    if idx1 < len(kpts) and idx2 < len(kpts):
                        p1, p2 = kpts[idx1], kpts[idx2]
                        if p1["v"] > 0 and p2["v"] > 0:
                            draw.line([(p1["x"], p1["y"]), (p2["x"], p2["y"])], fill="cyan", width=2)

                for i, kp in enumerate(kpts):
                    px, py, v = kp["x"], kp["y"], kp["v"]
                    if v > 0:
                        color = "lime" if v == 2 else "orange"
                        r = 4
                        draw.ellipse((px - r, py - r, px + r, py + r), fill=color, outline="white")

            st.image(annotated_img, use_container_width=True)
            st.caption(f"Resolution: {w}x{h} px | Green = Visible (v=2), Orange = Occluded (v=1)")
        except Exception as e:
            st.error(f"Error rendering image: {e}")

    with tag_col:
        st.subheader("Add Keypoint Pose Instance")

        selected_cls = st.selectbox("Select Class", options=st.session_state.classes)
        selected_kp = st.selectbox("Select Keypoint Joint", options=st.session_state.keypoint_names)

        kp_x = st.number_input("Joint X (px)", min_value=0, max_value=w if 'w' in locals() else 1920, value=w//2 if 'w' in locals() else 100)
        kp_y = st.number_input("Joint Y (px)", min_value=0, max_value=h if 'h' in locals() else 1080, value=h//2 if 'h' in locals() else 100)
        kp_v = st.radio("Visibility Flag", options=[2, 1, 0], format_func=lambda x: {2: "2 - Visible", 1: "1 - Occluded", 0: "0 - Absent"}[x], horizontal=True)

        if "current_pose_builder" not in st.session_state:
            st.session_state.current_pose_builder = {
                kp_name: {"x": 0, "y": 0, "v": 0} for kp_name in st.session_state.keypoint_names
            }

        if st.button("📌 Set Joint Position", use_container_width=True):
            st.session_state.current_pose_builder[selected_kp] = {
                "x": int(kp_x),
                "y": int(kp_y),
                "v": int(kp_v)
            }
            st.success(f"Set `{selected_kp}` to ({kp_x}, {kp_y}) [v={kp_v}]")

        st.markdown("---")
        
        if st.button("💾 Save Pose Instance to Image", type="primary", use_container_width=True):
            if filename not in st.session_state.annotations:
                st.session_state.annotations[filename] = {"poses": []}

            formatted_kpts = [
                st.session_state.current_pose_builder[kp_name]
                for kp_name in st.session_state.keypoint_names
            ]

            st.session_state.annotations[filename]["poses"].append({
                "class": selected_cls,
                "keypoints": formatted_kpts
            })

            st.session_state.current_pose_builder = {
                kp_name: {"x": 0, "y": 0, "v": 0} for kp_name in st.session_state.keypoint_names
            }

            save_pose_manifest(
                manifest_path, 
                st.session_state.classes, 
                st.session_state.keypoint_names, 
                st.session_state.skeleton, 
                st.session_state.annotations
            )
            st.success("Pose instance saved!")
            st.rerun()

        if filename in st.session_state.annotations and st.session_state.annotations[filename]["poses"]:
            st.markdown("---")
            st.write("**Tagged Poses on Image:**")
            for i, p in enumerate(st.session_state.annotations[filename]["poses"]):
                active_pts = sum(1 for k in p["keypoints"] if k["v"] > 0)
                st.text(f"Pose #{i+1}: Class '{p['class']}' ({active_pts}/{len(p['keypoints'])} joints)")

            if st.button("🗑️ Clear Poses for Image", use_container_width=True):
                st.session_state.annotations[filename]["poses"] = []
                save_pose_manifest(
                    manifest_path, 
                    st.session_state.classes, 
                    st.session_state.keypoint_names, 
                    st.session_state.skeleton, 
                    st.session_state.annotations
                )
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
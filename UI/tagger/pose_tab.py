import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent if FILE_PATH.parent.name == "tabs" else FILE_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image, ImageDraw

try:
    from scripts.settings.settings_script import load_settings
except ImportError:
    def load_settings():
        return {}

COCO_SKELETON_LINKS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6)
]

def scan_dataset_folders(base_datasets_dir: Path):
    if not base_datasets_dir.exists():
        return []
    subdirs = [d.name for d in base_datasets_dir.iterdir() if d.is_dir()]
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    root_has_images = any(p.suffix.lower() in valid_exts and not p.stem.endswith("-edit") for p in base_datasets_dir.iterdir() if p.is_file())
    
    options = []
    if root_has_images or not subdirs:
        options.append(".")
    options.extend(sorted(subdirs))
    return options

def scan_images(folder_path: Path):
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if not folder_path.exists():
        return []
    return [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in valid_exts and not p.stem.endswith("-edit")]

def get_pose_models_dir() -> Path:
    settings = load_settings()
    pose_folder = st.session_state.get(
        "settings_ultralytics_pose_folder",
        settings.get("ultralytics_pose_folder", "models/ultralytics/pose")
    )
    return Path(pose_folder).expanduser()

def scan_pose_models(models_folder: Path):
    if not models_folder.exists():
        return []
    return list(models_folder.glob("*.pt"))

def draw_pose_wireframes(image: Image.Image, poses: list) -> Image.Image:
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    for pose in poses:
        kps = pose.get("keypoints", [])
        for link in COCO_SKELETON_LINKS:
            i1, i2 = link
            if i1 < len(kps) and i2 < len(kps):
                p1, p2 = kps[i1], kps[i2]
                if p1["v"] > 0 and p2["v"] > 0:
                    draw.line([(p1["x"], p1["y"]), (p2["x"], p2["y"])], fill="cyan", width=3)
        for kp in kps:
            px, py, vis = kp["x"], kp["y"], kp["v"]
            if vis > 0:
                r = 4
                color = "lime" if vis == 2 else "orange"
                draw.ellipse((px - r, py - r, px + r, py + r), fill=color, outline="white")
    return annotated

def save_rendered_pose_artifact(img_path: Path, poses: list) -> Path:
    out_path = img_path.parent / f"{img_path.stem}-edit.jpeg"
    try:
        pil_img = Image.open(img_path)
        annotated = draw_pose_wireframes(pil_img, poses)
        annotated.save(out_path, format="JPEG", quality=90)
    except Exception as e:
        st.warning(f"Failed to save pose skeleton artifact: {e}")
    return out_path

def save_yolo_pose_label(img_path: Path, annotations: list, image_width: int, image_height: int):
    txt_path = img_path.with_suffix(".txt")
    lines = []
    for ann in annotations:
        bbox = ann.get("bbox_norm")
        kps = ann.get("keypoints", [])
        if not bbox: continue
        
        line_parts = [f"{ann.get('class_id', 0)}"] + [f"{val:.6f}" for val in bbox]
        for kp in kps:
            norm_x = kp["x"] / image_width if image_width > 0 else 0
            norm_y = kp["y"] / image_height if image_height > 0 else 0
            vis = kp.get("v", 2)
            line_parts.extend([f"{norm_x:.6f}", f"{norm_y:.6f}", str(vis)])
        lines.append(" ".join(line_parts))
        
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def run_pose_test_backend(model_path: str, image_files: list, conf_thresh: float = 0.35):
    from ultralytics import YOLO
    model = YOLO(model_path)
    test_results = []
    
    for img_p in image_files[:6]:
        res = model.predict(source=str(img_p), conf=conf_thresh, verbose=False)[0]
        detected_poses = []
        if res.keypoints is not None and len(res.keypoints) > 0:
            boxes = res.boxes
            for i, kp_data in enumerate(res.keypoints):
                xy_pts = kp_data.xy[0].cpu().numpy()
                confs = kp_data.conf[0].cpu().numpy() if kp_data.conf is not None else [1.0] * len(xy_pts)
                kps_formatted = [{"x": int(px), "y": int(py), "v": int(2 if c > conf_thresh else 1 if c > 0.1 else 0)} for (px, py), c in zip(xy_pts, confs)]
                
                if hasattr(boxes, 'xywhn'):
                    bbox_norm = boxes.xywhn[i].cpu().numpy().tolist()
                else:
                    bbox_norm = boxes.xywh[i].cpu().numpy().tolist()
                    
                cls_id = int(boxes.cls[i].item()) if boxes is not None else 0
                detected_poses.append({
                    "class_id": cls_id,
                    "bbox_norm": bbox_norm,
                    "keypoints": kps_formatted
                })
                
        img_pil = Image.open(img_p)
        # Just draw in memory, don't save to disk for the test preview
        annotated_img = draw_pose_wireframes(img_pil, detected_poses)
        test_results.append((img_p.name, annotated_img, len(detected_poses)))
        
    return test_results

def pose_tab():
    st.title("🦴 YOLO Pose Estimation & Auto-Tagger")
    
    saved_settings = load_settings()
    global_datasets_dir = st.session_state.get(
        "settings_datasets_folder", 
        saved_settings.get("datasets_folder", "./datasets")
    )
    base_datasets_dir = Path(global_datasets_dir).expanduser()
    pose_models_dir = get_pose_models_dir()
    available_models = scan_pose_models(pose_models_dir)
    
    st.markdown("### ⚙️ Dataset & Backend Pose Settings")
    cfg_col1, cfg_col2 = st.columns(2)
    
    with cfg_col1:
        dataset_options = scan_dataset_folders(base_datasets_dir)
        if dataset_options:
            selected_ds = st.selectbox("Select Dataset Folder", options=dataset_options, key="pose_ds_select")
            dataset_dir = base_datasets_dir / selected_ds if selected_ds != "." else base_datasets_dir
        else:
            st.warning(f"Datasets directory path not found: `{base_datasets_dir}`")
            dataset_dir = base_datasets_dir
            
    with cfg_col2:
        st.info(f"📁 **Pose Models Directory:** `{pose_models_dir}`")
        if available_models:
            model_options = [str(p) for p in available_models]
        else:
            model_options = ["yolo26n-pose.pt", "yolo11n-pose.pt", "yolov8n-pose.pt"]
        selected_model_path = st.selectbox("Select YOLO Pose Model", options=model_options, key="pose_model_select")
        
    image_files = scan_images(dataset_dir)
    
    st.markdown("---")
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 2])
    
    with ctrl_col1:
        conf_thresh = st.slider("Detection Confidence Threshold", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
        
    with ctrl_col2:
        st.write("")
        run_auto_detect = st.button("⚡ Run Pose Inference", type="primary", use_container_width=True)
        
    with ctrl_col3:
        st.write("")
        run_test_btn = st.button("🔍 Test Pose Tagging", use_container_width=True)
        
    if "pose_annotations_store" not in st.session_state:
        st.session_state.pose_annotations_store = {}
        
    if run_test_btn:
        if not image_files:
            st.warning("No images available to test.")
        else:
            with st.spinner("Executing pose backend and generating previews..."):
                test_previews = run_pose_test_backend(selected_model_path, image_files, conf_thresh)
            st.success("Test completed! Here are the pose previews.")
            st.markdown("#### Pose Previews")
            p_cols = st.columns(min(len(test_previews), 3))
            for i, (fname, annotated_img, count) in enumerate(test_previews):
                with p_cols[i % 3]:
                    # Streamlit can display PIL images directly
                    st.image(annotated_img, caption=f"{fname}\nPoses: {count}", use_container_width=True)
                    
    if not image_files:
        st.warning(f"No valid images found in: `{dataset_dir}`")
        return
        
    if "pose_img_idx" not in st.session_state:
        st.session_state.pose_img_idx = 0
        
    total_imgs = len(image_files)
    if st.session_state.pose_img_idx >= total_imgs:
        st.session_state.pose_img_idx = total_imgs - 1
        
    curr_idx = st.session_state.pose_img_idx
    current_file = image_files[curr_idx]
    
    if run_auto_detect:
        try:
            from ultralytics import YOLO
            with st.spinner("Running YOLO Pose detection and saving files..."):
                model = YOLO(selected_model_path)
                results = []
                for img_p in image_files:
                    res = model.predict(source=str(img_p), conf=conf_thresh, verbose=False)[0]
                    detected_poses = []
                    if res.keypoints is not None and len(res.keypoints) > 0:
                        boxes = res.boxes
                        for i, kp_data in enumerate(res.keypoints):
                            xy_pts = kp_data.xy[0].cpu().numpy()
                            confs = kp_data.conf[0].cpu().numpy() if kp_data.conf is not None else [1.0] * len(xy_pts)
                            kps_formatted = [{"x": int(px), "y": int(py), "v": int(2 if c > conf_thresh else 1 if c > 0.1 else 0)} for (px, py), c in zip(xy_pts, confs)]
                            
                            if hasattr(boxes, 'xywhn'):
                                bbox_norm = boxes.xywhn[i].cpu().numpy().tolist()
                            else:
                                bbox_norm = boxes.xywh[i].cpu().numpy().tolist()
                                
                            cls_id = int(boxes.cls[i].item()) if boxes is not None else 0
                            detected_poses.append({
                                "class_id": cls_id,
                                "bbox_norm": bbox_norm,
                                "keypoints": kps_formatted
                            })
                    
                    # --- SAVE TAGS AND SKELETON ARTIFACTS HERE ---
                    img_pil = Image.open(img_p)
                    save_yolo_pose_label(img_p, detected_poses, img_pil.width, img_pil.height)
                    save_rendered_pose_artifact(img_p, detected_poses)
                    # ---------------------------------------------
                    
                    st.session_state.pose_annotations_store[img_p.name] = detected_poses
                    results.append((img_p.name, detected_poses))
                st.toast(f"Detected and saved {len(results)} pose instance(s)!", icon="🦴")
        except Exception as err:
            st.error(f"Error during pose inference: {err}")
            
    st.markdown("---")
    st.subheader(f"🖼️ `{current_file.name}` ({curr_idx + 1} / {total_imgs})")
    
    img_col, tag_col = st.columns([3, 2])
    current_poses = st.session_state.pose_annotations_store.get(current_file.name, [])
    
    with img_col:
        try:
            image = Image.open(current_file).convert("RGB")
            w, h = image.size
            annotated_img = draw_pose_wireframes(image, current_poses)
            st.image(annotated_img, use_container_width=True)
            st.caption(f"Image Size: {w}x{h} px | Active Poses: {len(current_poses)}")
        except Exception as exc:
            st.error(f"Unable to load image: {exc}")
            
    with tag_col:
        st.markdown("### 📝 Save Tag & Render Output")
        if current_poses:
            if st.button("💾 Save Tag & Render Skeleton Artifact", type="primary", use_container_width=True):
                save_yolo_pose_label(current_file, current_poses, w, h)
                edited_file = save_rendered_pose_artifact(current_file, current_poses)
                st.success(f"Saved `.txt` label and rendered preview `{edited_file.name}`!")
                
            if st.button("🗑️ Clear Annotations", use_container_width=True):
                st.session_state.pose_annotations_store[current_file.name] = []
                st.rerun()
                
        st.markdown("---")
        nav_prev, nav_next = st.columns(2)
        
        with nav_prev:
            if st.button("⬅️ Previous", use_container_width=True):
                if st.session_state.pose_img_idx > 0:
                    st.session_state.pose_img_idx -= 1
                    st.rerun()
                    
        with nav_next:
            if st.button("Next ➡️", use_container_width=True):
                if st.session_state.pose_img_idx < total_imgs - 1:
                    st.session_state.pose_img_idx += 1
                    st.rerun()

# Module Entry Point Aliases for main tab loaders
render_tab = pose_tab
render_pose_tab = pose_tab
pose_tagger_tab = pose_tab

if __name__ == "__main__":
    pose_tab()
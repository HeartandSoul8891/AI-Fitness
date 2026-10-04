import os
import json
import cv2
import math
import shutil
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image, ImageDraw
from ultralytics import YOLO

# ==========================================
# 1. HELPER FUNCTIONS & STANDARDIZED JSON I/O
# ==========================================
def get_app_settings():
    try:
        from scripts.settings_script import load_settings
        return load_settings()
    except:
        return {"datasets_folder": "./datasets", "models_folder": "./models"}

def scan_folders(base_dir: Path):
    if not base_dir.exists(): return []
    return sorted([d.name for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

def scan_models(model_dir: Path, task_suffix: str = ""):
    if not model_dir.exists(): return []
    all_pts = list(model_dir.rglob("*.pt"))
    if task_suffix:
        prioritized = [p for p in all_pts if task_suffix in p.name]
        others = [p for p in all_pts if task_suffix not in p.name]
        return [str(p) for p in prioritized + others]
    return [str(p) for p in all_pts]

def get_model_metadata(model_path):
    """Reads and returns metadata from a YOLO model."""
    try:
        model = YOLO(model_path)
        names = model.names
        task = getattr(model, 'task', 'Unknown')
        kpt_shape = getattr(model, 'kpt_shape', None) if task == 'pose' else None
        return {"Task": task, "Classes": names, "Keypoint Shape": kpt_shape}
    except Exception as e:
        return {"Error": str(e)}

def save_standardized_yolo_json(image_path: Path, img_w: int, img_h: int, detections: list, shapes: list):
    """Saves a JSON file perfectly formatted for dataset_script.py's json_to_yolo_file function."""
    json_path = image_path.with_suffix(".json")
    data = {
        "imageWidth": img_w,
        "imageHeight": img_h,
        "detections": detections,  # Used for BBox & Pose
        "shapes": shapes           # Used for Segm & OBB
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return json_path

import os
import json
import cv2
import math
import shutil
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image, ImageDraw
from ultralytics import YOLO

# ==========================================
# 1. HELPER FUNCTIONS & STANDARDIZED JSON I/O
# ==========================================

# COCO 17-point skeleton bone connections (index pairs)
COCO_SKELETON_LINKS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),               # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)    # Legs
]

# Color palette for skeleton bones (cycles if more than 17)
BONE_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170),
]

def get_app_settings():
    try:
        from scripts.settings_script import load_settings
        return load_settings()
    except:
        return {"datasets_folder": "./datasets", "models_folder": "./models"}

def scan_folders(base_dir: Path):
    if not base_dir.exists(): return []
    return sorted([d.name for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

def scan_models(model_dir: Path, task_suffix: str = ""):
    if not model_dir.exists(): return []
    all_pts = list(model_dir.rglob("*.pt"))
    if task_suffix:
        prioritized = [p for p in all_pts if task_suffix in p.name]
        others = [p for p in all_pts if task_suffix not in p.name]
        return [str(p) for p in prioritized + others]
    return [str(p) for p in all_pts]

def get_model_metadata(model_path):
    """Reads and returns metadata from a YOLO model."""
    try:
        model = YOLO(model_path)
        names = model.names
        task = getattr(model, 'task', 'Unknown')
        kpt_shape = getattr(model, 'kpt_shape', None) if task == 'pose' else None
        return {"Task": task, "Classes": names, "Keypoint Shape": kpt_shape}
    except Exception as e:
        return {"Error": str(e)}

def save_standardized_yolo_json(image_path: Path, img_w: int, img_h: int, detections: list, shapes: list):
    """Saves a JSON file perfectly formatted for dataset_script.py's json_to_yolo_file function."""
    json_path = image_path.with_suffix(".json")
    data = {
        "imageWidth": img_w,
        "imageHeight": img_h,
        "detections": detections,
        "shapes": shapes
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return json_path

def draw_visual_preview(img: np.ndarray, detections: list, shapes: list, task_type: str) -> np.ndarray:
    """Draws overlays on the image for the Streamlit preview."""
    vis_img = img.copy()
    h, w = vis_img.shape[:2]
    
    # Standard COCO 17-keypoint skeleton connections
    coco_skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),          # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Arms
        (5, 11), (6, 12), (11, 12),              # Torso
        (11, 13), (12, 14), (13, 15), (14, 16)   # Legs
    ]
    
    if task_type in ["bbox", "pose"]:
         for det in detections:
             x1, y1, x2, y2 = map(int, det["box_xyxy"])
             tag = det["applied_tag"]
             conf = det.get("confidence", 0.0)
             
             # Draw Bounding Box
             cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
             cv2.putText(vis_img, f"{tag} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
             
             # Draw Pose Skeleton & Keypoints
             if task_type == "pose" and "keypoints" in det:
                 kps = det["keypoints"]
                 num_kps = len(kps)
                 
                 # Use COCO skeleton if 17 keypoints, otherwise fallback to connecting adjacent points
                 skeleton = coco_skeleton if num_kps == 17 else [(i, i+1) for i in range(num_kps - 1)]
                 
                 # 1. Draw skeleton lines first (so they appear visually under the keypoints)
                 for i, j in skeleton:
                     if i < num_kps and j < num_kps:
                         kp1, kp2 = kps[i], kps[j]
                         if kp1["v"] > 0 and kp2["v"] > 0:
                             cv2.line(vis_img, (int(kp1["x"]), int(kp1["y"])), (int(kp2["x"]), int(kp2["y"])), (255, 0, 0), 2)
                 
                 # 2. Draw keypoints on top
                 for kp in kps:
                     if kp["v"] > 0:
                         cv2.circle(vis_img, (int(kp["x"]), int(kp["y"])), 4, (0, 255, 255), -1)
                         
    elif task_type in ["segm", "pseudo_segm"]:
         # Draw semi-transparent filled polygons so the conversion is visually obvious
         overlay = vis_img.copy()
         for shape in shapes:
             pts = np.array(shape["points"], np.int32).reshape((-1, 1, 2))
             cv2.fillPoly(overlay, [pts], (255, 0, 255))
             cv2.polylines(vis_img, [pts], True, (255, 0, 255), 2)
             cv2.putText(vis_img, shape["label"], (int(pts[0][0][0]), int(pts[0][0][1]) - 5), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
         cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)
         
    return vis_img

# ==========================================
# 2. UNIFIED INFERENCE ENGINE
# ==========================================
def run_unified_tagger(model_path, image_paths, task_type, conf_thresh, target_class, new_tag, max_dim, min_dim, is_preview=False, limit=5):
    """Core engine that runs YOLO and returns standardized annotations."""
    model = YOLO(model_path)
    results_data = []
    
    paths_to_process = image_paths[:limit] if is_preview else image_paths
    
    for img_path in paths_to_process:
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        
        detections = []
        shapes = []
        
        results = model(img, conf=conf_thresh, verbose=False)
        
        for r in results:
            # --- BBOX & POSE ---
            if task_type in ["bbox", "pose"]:
                boxes = r.boxes
                kpts_data = r.keypoints if task_type == "pose" and r.keypoints is not None else None
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[int(box.cls[0])]
                    
                    if target_class and cls_name.lower() != target_class.lower(): continue
                    tag = new_tag if new_tag else cls_name
                    
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    if bw >= max_dim or bh >= max_dim or bw <= min_dim or bh <= min_dim: continue
                    
                    det = {"applied_tag": tag, "confidence": round(conf, 4), "box_xyxy": [x1, y1, x2, y2]}
                    
                    if task_type == "pose" and kpts_data is not None:
                        # FIX: Removed the erroneous [0] index which caused 0-d array iteration error
                        kp_xy = kpts_data.xy[i].cpu().numpy()
                        kp_conf = kpts_data.conf[i].cpu().numpy() if kpts_data.conf is not None else np.ones(len(kp_xy))
                        det["keypoints"] = [{"x": float(p[0]), "y": float(p[1]), "v": 2 if c > conf_thresh else 1 if c > 0.1 else 0} 
                                            for p, c in zip(kp_xy, kp_conf)]
                    detections.append(det)

            # --- NATIVE SEGMENTATION ---
            elif task_type == "segm":
                if r.masks is not None:
                    masks_xy = r.masks.xy
                    classes = r.boxes.cls
                    for poly_pts, cls_id in zip(masks_xy, classes):
                        cls_name = model.names[int(cls_id)]
                        tag = new_tag if new_tag else cls_name
                        if target_class and cls_name.lower() != target_class.lower(): continue
                        
                        shapes.append({
                            "label": tag,
                            "points": [[float(p[0]), float(p[1])] for p in poly_pts]
                        })

            # --- PSEUDO SEGMENTATION (BBox -> Poly) ---
            elif task_type == "pseudo_segm":
                for box in r.boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    cls_name = model.names[int(box.cls[0])]
                    if target_class and cls_name.lower() != target_class.lower(): continue
                    tag = new_tag if new_tag else cls_name
                    
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    if bw >= max_dim or bh >= max_dim or bw <= min_dim or bh <= min_dim: continue
                    
                    # Convert BBox to 4-point polygon
                    shapes.append({
                        "label": tag,
                        "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    })

            # --- CLASSIFICATION ---
            elif task_type == "cls":
                if r.probs is not None:
                    top_id = int(r.probs.top1)
                    cls_name = model.names[top_id]
                    detections.append({"applied_tag": cls_name, "confidence": float(r.probs.top1conf)})

        results_data.append({
            "path": img_path, "w": w, "h": h, 
            "detections": detections, "shapes": shapes,
            "img_bgr": img
        })
        
    return results_data

# ==========================================
# 3. MAIN GUI LAYOUT
# ==========================================
def tagger_gui():
    st.title("🎯 Unified Auto-Tagger & JSON Generator")
    st.write("Generate standardized `.json` annotations for BBox, Pose, Segmentation, and Classification. Use the **Dataset Prep** tab later to convert these to YOLO `.txt` files.")
    
    settings = get_app_settings()
    base_ds = Path(settings.get("datasets_folder", "./datasets"))
    base_models = Path(settings.get("models_folder", "./models"))

    st.markdown("### ⚙️ Global Configuration")
    col1, col2 = st.columns(2)
    with col1:
        ds_folders = scan_folders(base_ds)
        selected_ds = st.selectbox("Target Dataset Folder", ds_folders if ds_folders else ["No datasets found"])
        dataset_path = base_ds / selected_ds if selected_ds != "No datasets found" else base_ds
    with col2:
        st.info(f"Models Root: `{base_models}`")

    st.markdown("---")

    tab_bbox, tab_pose, tab_segm, tab_pseg, tab_cls = st.tabs([
        "📦 Bounding Box", "🦴 Pose Estimation", "✂️ Native Segmentation", 
        "🎭 Pseudo-Segmentation", "🏷️ Classification"
    ])

    def render_detection_tab(task_type, model_suffix, title):
        st.subheader(title)
        img_files = sorted([f for f in dataset_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']])
        
        if not img_files:
            st.warning(f"No images found in `{dataset_path}`.")
            return

        c1, c2, c3 = st.columns(3)
        with c1:
            models = scan_models(base_models, model_suffix)
            model_path = st.selectbox("Select Model", models if models else ["No models found"], key=f"model_{task_type}")
            
            # --- NEW: MODEL METADATA READER ---
            if model_path != "No models found":
                with st.expander(f"📊 Model Metadata: `{Path(model_path).name}`"):
                    meta = get_model_metadata(model_path)
                    if "Error" in meta:
                        st.error(meta["Error"])
                    else:
                        st.write(f"**Task:** `{meta['Task']}`")
                        if meta['Keypoint Shape']:
                            st.write(f"**Keypoint Shape:** `{meta['Keypoint Shape']}`")
                        st.write("**Class Names:**")
                        st.json(meta['Classes'])
                        
        with c2:
            conf = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05, key=f"conf_{task_type}")
        with c3:
            target_cls = st.text_input("Filter by Model Class (Optional)", value="", key=f"target_{task_type}")
            new_tag = st.text_input("Map to Custom Tag (Optional)", value="", key=f"tag_{task_type}")

        c4, c5 = st.columns(2)
        with c4:
            max_dim = st.slider("Max Box Size Ratio", 0.5, 1.0, 0.95, 0.01, key=f"max_{task_type}")
        with c5:
            min_dim = st.slider("Min Box Size Ratio", 0.0, 0.2, 0.01, 0.01, key=f"min_{task_type}")

        st.markdown("---")
        col_prev, col_run = st.columns(2)
        
        with col_prev:
            preview_btn = st.button(f"🔍 Preview on 5 Images", use_container_width=True, key=f"prev_{task_type}")
        with col_run:
            run_btn = st.button(f"🚀 Run Full Dataset & Save JSONs", type="primary", use_container_width=True, key=f"run_{task_type}")

        if preview_btn:
            if model_path == "No models found": st.error("Select a model.")
            else:
                with st.spinner("Running preview..."):
                    results = run_unified_tagger(model_path, img_files, task_type, conf, target_cls, new_tag, max_dim, min_dim, is_preview=True)
                
                st.success("Preview complete! Check the visual overlays and JSON structure below.")
                for res in results:
                    vis_img = draw_visual_preview(res["img_bgr"], res["detections"], res["shapes"], task_type)
                    st.image(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB), caption=f"{res['path'].name} (Dets: {len(res['detections'])}, Shapes: {len(res['shapes'])})", use_container_width=True)
                    with st.expander(f"View Generated JSON for {res['path'].name}"):
                        st.json({"imageWidth": res["w"], "imageHeight": res["h"], "detections": res["detections"], "shapes": res["shapes"]})

        if run_btn:
            if model_path == "No models found": st.error("Select a model.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = run_unified_tagger(model_path, img_files, task_type, conf, target_cls, new_tag, max_dim, min_dim, is_preview=False)
                
                for i, res in enumerate(results):
                    save_standardized_yolo_json(res["path"], res["w"], res["h"], res["detections"], res["shapes"])
                    progress_bar.progress((i + 1) / len(results))
                    status_text.text(f"Saved JSON for {res['path'].name}")
                
                st.success(f"✅ Successfully generated {len(results)} JSON files in `{dataset_path}`!")
                st.info("💡 **Next Step:** Go to the **Dataset Preparation** tab to convert these JSONs into YOLO `.txt` labels.")

    # --- TAB 1: BBOX ---
    with tab_bbox:
        render_detection_tab("bbox", "", "📦 Bounding Box Auto-Tagger")

    # --- TAB 2: POSE ---
    with tab_pose:
        render_detection_tab("pose", "-pose", "🦴 Pose Estimation Auto-Tagger")

    # --- TAB 3: NATIVE SEGMENTATION ---
    with tab_segm:
        render_detection_tab("segm", "-seg", "✂️ Native Semantic Segmentation")

    # --- TAB 4: PSEUDO SEGMENTATION ---
    with tab_pseg:
        render_detection_tab("pseudo_segm", "", "🎭 Pseudo-Segmentation (BBox to Polygons)")
        st.caption("Uses a standard BBox model but converts the detections into 4-point polygons for segmentation training.")

    # --- TAB 5: CLASSIFICATION (REBUILT) ---
    with tab_cls:
        st.subheader("🏷️ Classification Folder Sorter")
        st.write("Automatically classify images and sort them into class-specific subfolders.")
        
        cls_col1, cls_col2 = st.columns(2)
        with cls_col1:
            src_folders = scan_folders(base_ds)
            src_folder = st.selectbox("Source Dataset Folder (Unclassified Images)", src_folders if src_folders else ["No datasets found"], key="cls_src_folder")
            src_path = base_ds / src_folder if src_folder != "No datasets found" else base_ds
            
        with cls_col2:
            dst_folders = ["(Create New)"] + scan_folders(base_ds)
            dst_folder = st.selectbox("Destination Dataset Folder", dst_folders, key="cls_dst_folder")
            if dst_folder == "(Create New)":
                new_dst_name = st.text_input("New Destination Folder Name", value="classified_dataset", key="cls_new_dst")
                dst_path = base_ds / new_dst_name
            else:
                dst_path = base_ds / dst_folder
                
        cls_models = scan_models(base_models, "-cls")
        cls_model = st.selectbox("Classification Model", cls_models if cls_models else ["No models found"], key="cls_model")
        cls_conf = st.slider("Confidence Threshold", 0.1, 0.99, 0.5, 0.05, key="cls_conf")
        
        img_files = [f for f in src_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']] if src_path.exists() else []
        
        if st.button("🔍 Preview Classification", use_container_width=True):
            if cls_model == "No models found" or not img_files:
                st.error("Select a valid model and ensure source folder has images.")
            else:
                model = YOLO(cls_model)
                preview_data = []
                for img_path in img_files[:10]:
                    res = model(str(img_path), conf=cls_conf, verbose=False)[0]
                    if res.probs is not None:
                        top_cls = model.names[int(res.probs.top1)]
                        conf_val = float(res.probs.top1conf)
                        preview_data.append({"Image": img_path.name, "Predicted Class": top_cls, "Confidence": f"{conf_val:.2%}"})
                st.dataframe(preview_data, use_container_width=True)
                
        if st.button("🚀 Sort Images into Class Folders", type="primary", use_container_width=True):
            if cls_model == "No models found" or not img_files:
                st.error("Select a valid model and ensure source folder has images.")
            else:
                model = YOLO(cls_model)
                dst_path.mkdir(parents=True, exist_ok=True)
                progress = st.progress(0)
                sorted_count = 0
                for i, img_path in enumerate(img_files):
                    res = model(str(img_path), conf=cls_conf, verbose=False)[0]
                    if res.probs is not None:
                        top_cls = model.names[int(res.probs.top1)]
                        # Sanitize class name to ensure it's a valid folder name
                        safe_cls = "".join(c for c in top_cls if c.isalnum() or c in "._- ").strip()
                        if not safe_cls: safe_cls = "unknown"
                        
                        target_dir = dst_path / safe_cls
                        target_dir.mkdir(exist_ok=True)
                        
                        dest = target_dir / img_path.name
                        if not dest.exists():
                            shutil.copy2(str(img_path), str(dest)) # Copying to preserve original dataset
                            sorted_count += 1
                    progress.progress((i + 1) / len(img_files))
                st.success(f"Successfully sorted {sorted_count} images into `{dst_path}`!")

if __name__ == "__main__":
    tagger_gui()
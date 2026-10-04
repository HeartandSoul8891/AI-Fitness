import json
import os
import random
import shutil
from PIL import Image


def get_dataset_folders(base_path):
    """Returns a list of available dataset subdirectories."""
    if not os.path.exists(base_path):
        return []
    dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    return dirs if dirs else ["."]


def get_unique_labels_from_source(source_dir):
    """Scans all JSON files in the source directory to automatically find unique label names."""
    if not os.path.exists(source_dir):
        return {}
    unique_labels = set()
    for file in os.listdir(source_dir):
        if file.lower().endswith(".json"):
            path = os.path.join(source_dir, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for shape in data.get("shapes", []):
                    label = shape.get("label")
                    if label:
                        unique_labels.add(label)
                for det in data.get("detections", []):
                    label = det.get("applied_tag") or det.get("label") or det.get("raw_model_class")
                    if label:
                        unique_labels.add(label)
            except Exception:
                pass
    return {label: idx for idx, label in enumerate(sorted(unique_labels))}


def create_dataset_structure(training_root, dataset_name, task_type="bbox"):
    """Creates the standard dataset folder structure based on YOLO task type."""
    dataset_path = os.path.join(training_root, dataset_name)
    if task_type == "cls":
        for split in ["train", "val"]:
            os.makedirs(os.path.join(dataset_path, split), exist_ok=True)
    else:
        for split in ["train", "val"]:
            for sub in ["images", "json", "labels"]:
                os.makedirs(os.path.join(dataset_path, split, sub), exist_ok=True)
    return dataset_path


def cleanup_empty_jsons_and_media(source_dir):
    """Purges empty/invalid JSONs and matching image source/edit files."""
    if not os.path.exists(source_dir):
        return False, f"Source directory does not exist: {source_dir}"

    json_files = [f for f in os.listdir(source_dir) if f.lower().endswith(".json")]
    if not json_files:
        return False, "No JSON files found in the source directory to clean."

    removed_count = 0
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for json_file in json_files:
        json_path = os.path.join(source_dir, json_file)
        file_base = os.path.splitext(json_file)[0]
        is_empty_or_invalid = False

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            shapes = data.get("shapes", [])
            detections = data.get("detections", [])
            keypoints = data.get("keypoints", [])
            if not shapes and not detections and not keypoints:
                is_empty_or_invalid = True
        except Exception:
            is_empty_or_invalid = True

        if is_empty_or_invalid:
            if os.path.exists(json_path):
                os.remove(json_path)
            for ext in valid_exts:
                orig_img_path = os.path.join(source_dir, file_base + ext)
                if os.path.exists(orig_img_path):
                    os.remove(orig_img_path)
                edit_img_path = os.path.join(source_dir, f"{file_base}_yolo_edit{ext}")
                if os.path.exists(edit_img_path):
                    os.remove(edit_img_path)
            removed_count += 1

    return True, f"Successfully purged {removed_count} empty/invalid JSONs alongside matching images."


def move_and_split_files(source_dir, dataset_path, split_ratio=0.8):
    """Copies and splits raw images and matching JSONs into train/val directories."""
    if not os.path.exists(source_dir):
        return False, f"Source directory does not exist: {source_dir}"

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    all_files = os.listdir(source_dir)
    image_files = [
        f for f in all_files
        if f.lower().endswith(valid_exts) and "_yolo_edit" not in f.lower()
    ]

    if not image_files:
        return False, "No valid image files found in dataset directory."

    random.shuffle(image_files)
    split_idx = int(len(image_files) * split_ratio)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]

    def process_split(img_list, split_name):
        img_dest = os.path.join(dataset_path, split_name, "images")
        json_dest = os.path.join(dataset_path, split_name, "json")
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(json_dest, exist_ok=True)
        count = 0
        for img_file in img_list:
            base_name = os.path.splitext(img_file)[0]
            src_img_path = os.path.join(source_dir, img_file)
            dst_img_path = os.path.join(img_dest, img_file)
            shutil.copy2(src_img_path, dst_img_path)
            count += 1
            json_file = base_name + ".json"
            src_json_path = os.path.join(source_dir, json_file)
            if os.path.exists(src_json_path):
                dst_json_path = os.path.join(json_dest, json_file)
                shutil.copy2(src_json_path, dst_json_path)
        return count

    train_count = process_split(train_images, "train")
    val_count = process_split(val_images, "val")
    return True, f"Successfully distributed {train_count} train and {val_count} val images & annotations."


def json_to_yolo_file(json_path, txt_path, class_mapping, task_type="bbox"):
    """Converts annotation JSON into task-specific YOLO format (.txt)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data.get("imageWidth") or data.get("width") or data.get("image_width") or data.get("imgWidth")
    img_h = data.get("imageHeight") or data.get("height") or data.get("image_height") or data.get("imgHeight")

    if not img_w or not img_h:
        base_dir = os.path.dirname(json_path)
        parent_dir = os.path.dirname(base_dir)
        file_base = os.path.splitext(os.path.basename(json_path))[0]
        possible_image_dirs = [os.path.join(parent_dir, "images"), base_dir]
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for d in possible_image_dirs:
            for ext in valid_exts:
                potential_path = os.path.join(d, file_base + ext)
                if os.path.exists(potential_path):
                    try:
                        with Image.open(potential_path) as img:
                            img_w, img_h = img.size
                        break
                    except Exception:
                        pass
            if img_w and img_h:
                break

    if not img_w or not img_h:
        return False, "Missing image dimensions."

    yolo_lines = []

    # --- Process Shapes (Polygon / Point data) ---
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label not in class_mapping:
            continue
        class_id = class_mapping[label]
        points = shape.get("points", [])
        if not points:
            continue

        if task_type == "segm":
            norm_pts = []
            for px, py in points:
                norm_pts.extend([
                    f"{max(0.0, min(1.0, px / img_w)):.6f}",
                    f"{max(0.0, min(1.0, py / img_h)):.6f}"
                ])
            yolo_lines.append(f"{class_id} " + " ".join(norm_pts))

        elif task_type == "obb":
            if len(points) >= 4:
                pts = points[:4]
            else:
                xs, ys = [p[0] for p in points], [p[1] for p in points]
                pts = [(min(xs), min(ys)), (max(xs), min(ys)),
                       (max(xs), max(ys)), (min(xs), max(ys))]
            norm_pts = []
            for px, py in pts:
                norm_pts.extend([
                    f"{max(0.0, min(1.0, px / img_w)):.6f}",
                    f"{max(0.0, min(1.0, py / img_h)):.6f}"
                ])
            yolo_lines.append(f"{class_id} " + " ".join(norm_pts))

        elif task_type == "pose":
            # Pose: class_id xc yc w h kx1 ky1 v1 kx2 ky2 v2 ...
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            x_center = max(0.0, min(1.0, ((x_min + x_max) / 2.0) / img_w))
            y_center = max(0.0, min(1.0, ((y_min + y_max) / 2.0) / img_h))
            width = max(0.0, min(1.0, (x_max - x_min) / img_w))
            height = max(0.0, min(1.0, (y_max - y_min) / img_h))
            line_parts = [f"{class_id}", f"{x_center:.6f}", f"{y_center:.6f}",
                          f"{width:.6f}", f"{height:.6f}"]
            # If keypoints are embedded in the shape
            kps = shape.get("keypoints", [])
            for kp in kps:
                kx = kp.get("x", 0) / img_w
                ky = kp.get("y", 0) / img_h
                kv = kp.get("v", 2)
                line_parts.extend([f"{max(0.0, min(1.0, kx)):.6f}",
                                   f"{max(0.0, min(1.0, ky)):.6f}", str(kv)])
            yolo_lines.append(" ".join(line_parts))

        else:
            # Default bbox
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            x_center = max(0.0, min(1.0, ((x_min + x_max) / 2.0) / img_w))
            y_center = max(0.0, min(1.0, ((y_min + y_max) / 2.0) / img_h))
            width = max(0.0, min(1.0, (x_max - x_min) / img_w))
            height = max(0.0, min(1.0, (y_max - y_min) / img_h))
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # --- Process Detections (Bounding Box arrays) ---
    for det in data.get("detections", []):
        label = det.get("applied_tag") or det.get("label") or det.get("raw_model_class")
        if label not in class_mapping:
            continue
        class_id = class_mapping[label]
        box = det.get("box_xyxy", [])
        if len(box) != 4:
            continue
        x_min, y_min, x_max, y_max = box

        if task_type == "segm" or task_type == "obb":
            pts = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
            norm_pts = []
            for px, py in pts:
                norm_pts.extend([
                    f"{max(0.0, min(1.0, px / img_w)):.6f}",
                    f"{max(0.0, min(1.0, py / img_h)):.6f}"
                ])
            yolo_lines.append(f"{class_id} " + " ".join(norm_pts))

        elif task_type == "pose":
            x_center = max(0.0, min(1.0, ((x_min + x_max) / 2.0) / img_w))
            y_center = max(0.0, min(1.0, ((y_min + y_max) / 2.0) / img_h))
            width = max(0.0, min(1.0, (x_max - x_min) / img_w))
            height = max(0.0, min(1.0, (y_max - y_min) / img_h))
            line_parts = [f"{class_id}", f"{x_center:.6f}", f"{y_center:.6f}",
                          f"{width:.6f}", f"{height:.6f}"]
            kps = det.get("keypoints", [])
            for kp in kps:
                kx = kp.get("x", 0) / img_w
                ky = kp.get("y", 0) / img_h
                kv = kp.get("v", 2)
                line_parts.extend([f"{max(0.0, min(1.0, kx)):.6f}",
                                   f"{max(0.0, min(1.0, ky)):.6f}", str(kv)])
            yolo_lines.append(" ".join(line_parts))

        else:
            x_center = max(0.0, min(1.0, ((x_min + x_max) / 2.0) / img_w))
            y_center = max(0.0, min(1.0, ((y_min + y_max) / 2.0) / img_h))
            width = max(0.0, min(1.0, (x_max - x_min) / img_w))
            height = max(0.0, min(1.0, (y_max - y_min) / img_h))
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    if not yolo_lines:
        return False, "No valid annotations matched mapping."

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))
    return True, "Success"


def convert_dataset_jsons(dataset_path, class_mapping, task_type="bbox"):
    """Iterates through splits and converts JSONs to task-specific YOLO TXT labels."""
    converted_count = 0
    for split in ["train", "val"]:
        json_dir = os.path.join(dataset_path, split, "json")
        labels_dir = os.path.join(dataset_path, split, "labels")
        if os.path.exists(json_dir):
            os.makedirs(labels_dir, exist_ok=True)
            for file in os.listdir(json_dir):
                if file.lower().endswith(".json"):
                    json_file_path = os.path.join(json_dir, file)
                    txt_file_path = os.path.join(labels_dir, os.path.splitext(file)[0] + ".txt")
                    success, _ = json_to_yolo_file(json_file_path, txt_file_path, class_mapping, task_type)
                    if success:
                        converted_count += 1
    return True, f"Successfully converted {converted_count} files for task `{task_type}`."


def generate_yaml(dataset_path, class_mapping, task_type="bbox", kpt_shape=None):
    """Generates task-aware data.yaml for Ultralytics."""
    names_formatted = "\n".join([f"  {cid}: {name}" for name, cid in class_mapping.items()])
    yaml_content = f"""path: {dataset_path.replace(chr(92), '/')}
train: train/images
val: val/images
names:
{names_formatted}
"""
    if task_type == "pose" and kpt_shape:
        yaml_content += f"\nkpt_shape: {kpt_shape}\n"

    yaml_path = os.path.join(dataset_path, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    return True, f"Successfully generated data.yaml at `{yaml_path}`"
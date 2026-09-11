import os
import json
from PIL import Image

def get_dataset_folders(base_path):
    if not os.path.exists(base_path):
        return []
    dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    return dirs if dirs else ["."]

def get_image_files(folder_path):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if not os.path.exists(folder_path):
        return []
    return [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]

def prepare_display_image(image_path, max_width=768):
    image = Image.open(image_path)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    orig_w, orig_h = image.size

    if orig_w > max_width:
        scale_factor = max_width / float(orig_w)
        disp_w = max_width
        disp_h = int(orig_h * scale_factor)
    else:
        scale_factor = 1.0
        disp_w, disp_h = orig_w, orig_h

    display_image = image.resize((disp_w, disp_h))
    return display_image, scale_factor, orig_w, orig_h

def format_annotation_shapes(detected_bboxes, orig_w, orig_h, disp_w, disp_h):
    """Maps display canvas coordinates back to full-resolution JSON polygon coordinates."""
    shapes = []
    scale_x = orig_w / float(disp_w) if disp_w > 0 else 1.0
    scale_y = orig_h / float(disp_h) if disp_h > 0 else 1.0

    if not detected_bboxes:
        return shapes

    for item in detected_bboxes:
        bbox = item.get("bbox", [])
        label = item.get("label", "")
        if len(bbox) == 4:
            left = float(bbox[0]) * scale_x
            top = float(bbox[1]) * scale_y
            w = float(bbox[2]) * scale_x
            h = float(bbox[3]) * scale_y

            shapes.append({
                "label": label,
                "score": None,
                "points": [
                    [left, top],
                    [left + w, top],
                    [left + w, top + h],
                    [left, top + h],
                ],
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            })
    return shapes

def save_tag_json(image_path, shapes, orig_w, orig_h):
    json_path = os.path.splitext(image_path)[0] + ".json"
    tag_data = {
        "version": "4.0.2",
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": orig_h,
        "imageWidth": orig_w,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tag_data, f, indent=4)
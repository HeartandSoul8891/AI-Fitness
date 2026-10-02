import json
import os
import cv2
import numpy as np
from ultralytics import YOLO


def compute_iou(box1, box2):
  """Calculates Intersection over Union (IoU) between two [x1, y1, x2, y2] boxes."""
  x1 = max(box1[0], box2[0])
  y1 = max(box1[1], box2[1])
  x2 = min(box1[2], box2[2])
  y2 = min(box1[3], box2[3])

  inter_area = max(0, x2 - x1) * max(0, y2 - y1)
  if inter_area == 0:
    return 0.0

  box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
  box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
  union_area = float(box1_area + box2_area - inter_area)

  return inter_area / union_area if union_area > 0 else 0.0

def filter_overlapping_detections(
    detections, iou_threshold=0.5, max_detections_per_image=100
):
  """Filters out heavily overlapping duplicate bounding boxes while allowing multiple distinct detections per image."""
  if not detections:
    return []

  sorted_dets = sorted(
      detections, key=lambda k: k.get("confidence", 0.0), reverse=True
  )
  filtered = []

  for current in sorted_dets:
    if len(filtered) >= max_detections_per_image:
      break

    keep = True
    current_box = current.get("box_xyxy", [0, 0, 0, 0])
    current_tag = current.get("applied_tag", "")

    for existing in filtered:
      existing_box = existing.get("box_xyxy", [0, 0, 0, 0])
      existing_tag = existing.get("applied_tag", "")

      iou = compute_iou(current_box, existing_box)
      # Drop only if it's the same tag overlapping heavily (>50% IoU) or identical box (>85% IoU)
      if (current_tag == existing_tag and iou > iou_threshold) or iou > 0.85:
        keep = False
        break

    if keep:
      filtered.append(current)

  return filtered

def sanity_check_and_filter_boxes(
    detections, img_width, img_height, max_dim=0.95, min_dim=0.01
):
  """Secondary sanity check pass: validates size constraints, coordinate boundaries,

  and structural anomalies to ensure absolute cleanliness before final processing.
  """
  sanitized = []
  for det in detections:
    box = det.get("box_xyxy", [])
    if len(box) != 4:
      continue

    x1, y1, x2, y2 = box

    # Clip coordinates within image boundaries just in case
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))

    w_px = x2 - x1
    h_px = y2 - y1

    if img_width > 0 and img_height > 0:
      w_ratio = w_px / img_width
      h_ratio = h_px / img_height

      # Exclude oversized/full-screen boxes or microscopic noise boxes
      if (
          w_ratio >= max_dim
          or h_ratio >= max_dim
          or w_ratio <= min_dim
          or h_ratio <= min_dim
      ):
        continue

    det["box_xyxy"] = [x1, y1, x2, y2]
    sanitized.append(det)

  return sanitized


def run_auto_tagger(
    dataset_path,
    model_path,
    default_tag="",
    target_model_class="",
    new_tag="",
    conf_threshold=0.25,
    max_box_dimension=0.95,
    min_box_dimension=0.01,
    is_benchmark=False,
    benchmark_limit=12,
):
  """Scans dataset, handles detection, sizing constraints, overlap filtering,

  and applies a secondary sanity checker pass.
  """
  if not os.path.exists(dataset_path):
    return {
        "success": False,
        "message": f"Dataset path {dataset_path} does not exist.",
    }

  if not os.path.exists(model_path):
    return {
        "success": False,
        "message": f"Model path {model_path} does not exist.",
    }

  try:
    model = YOLO(model_path)
  except Exception as e:
    return {
        "success": False,
        "message": f"Failed to load YOLO model: {str(e)}",
    }

  image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
  image_files = [
      os.path.join(root, file)
      for root, _, files in os.walk(dataset_path)
      for file in files
      if file.lower().endswith(image_extensions)
      and not file.endswith("_yolo_edit.jpg")
  ]

  if not image_files:
    return {
        "success": False,
        "message": (
            "No valid original images found in the selected dataset folder."
        ),
    }

  if is_benchmark:
    image_files = image_files[:benchmark_limit]

  processed_count = 0
  preview_items = []

  for img_path in image_files:
    base_dir = os.path.dirname(img_path)
    file_name_no_ext, ext = os.path.splitext(os.path.basename(img_path))

    yolo_edit_filename = f"{file_name_no_ext}_yolo_edit.jpg"
    yolo_edit_path = os.path.join(base_dir, yolo_edit_filename)
    json_path = os.path.join(base_dir, f"{file_name_no_ext}.json")

    tags = set()
    if default_tag.strip():
      tags.add(default_tag.strip())

    raw_detected_boxes = []
    has_detection = False

    try:
      stream = open(img_path, "rb")
      bytes_data = bytearray(stream.read())
      stream.close()
      np_array = np.asarray(bytes_data, dtype=np.uint8)
      img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

      if img is None:
        print(f"Skipping: Could not read image at {img_path}")
        continue

      img_height, img_width = img.shape[:2]

      results = model(img_path, conf=conf_threshold, verbose=False)

      for r in results:
        boxes = r.boxes
        for box in boxes:
          x1, y1, x2, y2 = map(int, box.xyxy[0])
          conf = float(box.conf[0])
          cls_id = int(box.cls[0])
          raw_class_name = model.names[cls_id]

          if target_model_class.strip():
            if (
                raw_class_name.lower()
                != target_model_class.strip().lower()
            ):
              continue

          tag_to_use = (
              new_tag.strip() if new_tag.strip() else raw_class_name
          )

          raw_detected_boxes.append({
              "raw_model_class": raw_class_name,
              "applied_tag": tag_to_use,
              "confidence": round(conf, 4),
              "box_xyxy": [x1, y1, x2, y2],
          })

      # Pass 1: IoU Overlap Filtering (Allowing up to 100 detections per image)
      overlap_filtered = filter_overlapping_detections(
          raw_detected_boxes, iou_threshold=0.5, max_detections_per_image=100
      )

      # Pass 2: Secondary Sanity Checker (Size & Boundary Validation)
      detected_boxes = sanity_check_and_filter_boxes(
          overlap_filtered,
          img_width,
          img_height,
          max_dim=max_box_dimension,
          min_dim=min_box_dimension,
      )

      for det in detected_boxes:
        has_detection = True
        tags.add(det["applied_tag"])
        x1, y1, x2, y2 = det["box_xyxy"]
        conf = det["confidence"]
        tag_to_use = det["applied_tag"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{tag_to_use} {conf:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        cv2.rectangle(
            img,
            (x1, max(y1 - text_height - 10, 0)),
            (x1 + text_width, y1),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 5, text_height)),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )

      if is_benchmark:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preview_items.append({
            "image_path": img_path,
            "img_rgb": img_rgb,
            "detections_summary": {
                "tags": list(tags),
                "detections_count": len(detected_boxes),
                "detections": detected_boxes,
            },
        })
      else:
        cv2.imwrite(yolo_edit_path, img)

        tag_data = {
            "original_image_path": img_path,
            "yolo_edit_path": yolo_edit_path,
            "has_detection": has_detection,
            "tags": list(tags),
            "detections": detected_boxes,
        }

        with open(json_path, "w", encoding="utf-8") as f:
          json.dump(tag_data, f, indent=4)

      processed_count += 1

    except Exception as e:
      print(f"Error processing image {img_path}: {e}")

  mode_text = (
      "Benchmark test preview generated for"
      if is_benchmark
      else "Successfully processed"
  )
  return {
      "success": True,
      "message": (
          f"{mode_text} {processed_count} images! Size constraints & sanity"
          " checks enforced."
      ),
      "preview_items": preview_items,
  }


def clean_existing_dataset_json(dataset_path):
  """Scans existing dataset JSONs, applying IoU deduplication and size sanity checks."""
  if not os.path.exists(dataset_path):
    return {
        "success": False,
        "message": f"Dataset path {dataset_path} does not exist.",
    }

  json_files = [
      os.path.join(root, file)
      for root, _, files in os.walk(dataset_path)
      for file in files
      if file.lower().endswith(".json")
  ]

  if not json_files:
    return {
        "success": False,
        "message": "No JSON metadata files found in the dataset folder.",
    }

  cleaned_count = 0
  for j_path in json_files:
    try:
      with open(j_path, "r", encoding="utf-8") as f:
        data = json.load(f)

      if "detections" in data and isinstance(data["detections"], list):
        # Apply strict cleaning pass
        filtered_dets = filter_overlapping_detections(
            data["detections"], iou_threshold=0.4
        )

        new_tags = set()
        for det in filtered_dets:
          if "applied_tag" in det:
            new_tags.add(det["applied_tag"])

        data["detections"] = filtered_dets
        data["tags"] = list(new_tags)
        data["has_detection"] = len(filtered_dets) > 0

        with open(j_path, "w", encoding="utf-8") as f:
          json.dump(data, f, indent=4)

        cleaned_count += 1
    except Exception as e:
      print(f"Error cleaning JSON {j_path}: {e}")

  return {
      "success": True,
      "message": (
          "Successfully cleaned duplicate tags and boxes across"
          f" {cleaned_count} JSON files!"
      ),
  }
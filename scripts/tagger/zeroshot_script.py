import os
import cv2
import pandas as pd
from PIL import Image
from ultralytics import YOLOWorld


def load_zeroshot_model(model_name: str = "yolov8s-worldv2.pt"):
    """
    Load a YOLO-World open-vocabulary object detection model.
    """
    model = YOLOWorld(model_name)
    return model


def run_zeroshot_inference(
    model,
    source_path: str,
    prompts: list,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
):
    """
    Runs zero-shot inference on an image or directory using textual prompt classes.
    
    Args:
        model: Loaded YOLOWorld instance
        source_path (str): Path to image file or directory of images
        prompts (list): List of prompt string categories (e.g., ['cat', 'red car', 'coffee mug'])
        conf_threshold (float): Minimum confidence threshold
        iou_threshold (float): IoU NMS threshold
        
    Returns:
        tuple: (annotated_images_dict, dataframe_results)
    """
    # Set the custom text prompts as target detection classes
    model.set_classes(prompts)

    image_paths = []
    if os.path.isdir(source_path):
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for root, _, files in os.walk(source_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_exts:
                    image_paths.append(os.path.join(root, file))
    elif os.path.isfile(source_path):
        image_paths.append(source_path)

    annotated_results = {}
    detections_list = []

    for img_path in image_paths:
        # Run inference
        results = model.predict(
            source=img_path,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )[0]

        # Extract annotated BGR image and convert to RGB PIL Image
        res_bgr = results.plot()
        res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
        annotated_results[os.path.basename(img_path)] = Image.fromarray(res_rgb)

        # Parse detections for DataFrame
        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                label = prompts[cls_id] if cls_id < len(prompts) else "unknown"
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()

                detections_list.append({
                    "image": os.path.basename(img_path),
                    "class_id": cls_id,
                    "label": label,
                    "confidence": round(conf, 4),
                    "xmin": round(xyxy[0], 1),
                    "ymin": round(xyxy[1], 1),
                    "xmax": round(xyxy[2], 1),
                    "ymax": round(xyxy[3], 1),
                })

    df_results = pd.DataFrame(detections_list)
    return annotated_results, df_results
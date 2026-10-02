import os
from pathlib import Path
import pandas as pd
from ultralytics import YOLO


def get_test_images(image_dir):
    """Finds all JPG and PNG images in the given directory."""
    if not image_dir or not os.path.exists(image_dir):
        return []
    p = Path(image_dir)
    return list(p.rglob("*.jpg")) + list(p.rglob("*.jpeg")) + list(p.rglob("*.png"))


def run_stress_test_backend(
    model_path, image_dir, pass1_conf=0.75, pass2_conf=0.17, progress_callback=None
):
    """
    Executes a two-pass detection test on all images in image_dir.
    Returns summary dict and a DataFrame of detection confidences.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    image_paths = get_test_images(image_dir)
    total_images = len(image_paths)

    if total_images == 0:
        return {
            "total_images": 0,
            "pass1_detected": 0,
            "pass2_recovered": 0,
            "still_missed": 0,
        }, pd.DataFrame()

    model = YOLO(model_path)

    detections = []
    pass1_detected_imgs = set()
    missed_pass1_imgs = []

    # --- Pass 1: High Confidence ---
    for idx, img_path in enumerate(image_paths):
        results = model.predict(source=str(img_path), conf=pass1_conf, verbose=False)
        boxes = results[0].boxes if len(results) > 0 else []

        if len(boxes) > 0:
            pass1_detected_imgs.add(img_path)
            for box in boxes:
                detections.append({
                    "image": img_path.name,
                    "confidence": float(box.conf[0]),
                    "class": int(box.cls[0]),
                    "pass": "Pass 1 (High)",
                })
        else:
            missed_pass1_imgs.append(img_path)

        if progress_callback:
            progress_callback((idx + 1) / (total_images * 2), f"Pass 1: {idx + 1}/{total_images} images")

    # --- Pass 2: Fallback Low Confidence on Misses ---
    pass2_recovered_imgs = set()
    still_missed_imgs = []

    for idx, img_path in enumerate(missed_pass1_imgs):
        results = model.predict(source=str(img_path), conf=pass2_conf, verbose=False)
        boxes = results[0].boxes if len(results) > 0 else []

        if len(boxes) > 0:
            pass2_recovered_imgs.add(img_path)
            for box in boxes:
                detections.append({
                    "image": img_path.name,
                    "confidence": float(box.conf[0]),
                    "class": int(box.cls[0]),
                    "pass": "Pass 2 (Recovery)",
                })
        else:
            still_missed_imgs.append(img_path)

        if progress_callback:
            progress_step = 0.5 + ((idx + 1) / (len(missed_pass1_imgs) * 2 if missed_pass1_imgs else 1)) * 0.5
            progress_callback(min(progress_step, 1.0), f"Pass 2: {idx + 1}/{len(missed_pass1_imgs)} missed images")

    summary = {
        "total_images": total_images,
        "pass1_detected": len(pass1_detected_imgs),
        "pass2_recovered": len(pass2_recovered_imgs),
        "still_missed": len(still_missed_imgs),
    }

    df_results = pd.DataFrame(detections)
    return summary, df_results
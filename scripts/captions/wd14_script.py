import os
import csv
import numpy as np
import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download

try:
    import onnxruntime as ort
except ImportError:
    ort = None

_WD14_SESSIONS = {}
_WD14_TAGS_DATA = {}

AVAILABLE_WD14_MODELS = {
    "wd-v1-4-convnextv2-tagger-v2": "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
    "wd-v1-4-swinv2-tagger-v2": "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
    "wd-v1-4-vit-tagger-v2": "SmilingWolf/wd-v1-4-vit-tagger-v2",
    "wd-eva02-large-tagger-v3": "SmilingWolf/wd-eva02-large-tagger-v3",
}

@st.cache_resource(show_spinner="Loading WD14 ONNX model into memory...")
def load_wd14_model(
    model_name: str = "wd-v1-4-convnextv2-tagger-v2",
    model_root: str = None
):
    if ort is None:
        raise ImportError("onnxruntime is required for WD14 Tagger.")

    repo_id = AVAILABLE_WD14_MODELS.get(
        model_name,
        "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
    )

    if model_root:
        local_dir = os.path.join(model_root, model_name)
        os.makedirs(local_dir, exist_ok=True)

        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.onnx",
            local_dir=local_dir
        )

        tags_path = hf_hub_download(
            repo_id=repo_id,
            filename="selected_tags.csv",
            local_dir=local_dir
        )
    else:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.onnx"
        )

        tags_path = hf_hub_download(
            repo_id=repo_id,
            filename="selected_tags.csv"
        )

    providers = [
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "CPUExecutionProvider"
    ]

    available = ort.get_available_providers()

    selected = [
        p for p in providers
        if p in available
    ]

    session = ort.InferenceSession(
        model_path,
        providers=selected
    )

    tags = []
    category_map = {}

    with open(tags_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)

        for i, row in enumerate(reader):
            tag_name = row[1]
            category = int(row[2])

            tags.append(tag_name)
            category_map[i] = (
                tag_name,
                category
            )

    return session, (tags, category_map)

def preprocess_image_wd14(image: Image.Image, target_size: int = 448) -> np.ndarray:
    """
    Preprocess image for WD14 model input: Resize with padding and normalize.
    """
    img = np.array(image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    max_dim = max(h, w)
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2

    img_padded = cv2.copyMakeBorder(
        img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    img_resized = cv2.resize(img_padded, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # BGR to RGB and normalize float32
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_tensor = np.expand_dims(img_rgb, axis=0)
    return img_tensor

st.cache_resource(show_spinner="Loading WD14 ONNX model into memory...")


def run_wd14_tagger(
    image: Image.Image,
    model_name: str = "wd-v1-4-convnextv2-tagger-v2",
    model_root: str = None,
    general_thresh: float = 0.35,
    character_thresh: float = 0.85,
    replace_underscores: bool = True,
    prefix_tags: list = None,
    exclude_tags: list = None,
):
    """
    Runs WD14 tagger on an image and returns formatted tag strings and dataframe breakdown.
    """
    if prefix_tags is None:
        prefix_tags = []
    if exclude_tags is None:
        exclude_tags = []


    session, (tags, category_map) = load_wd14_model(model_name, model_root)

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    target_size = input_shape[1] if isinstance(input_shape[1], int) else 448

    processed_img = preprocess_image_wd14(image, target_size=target_size)
    outputs = session.run(None, {input_name: processed_img})[0][0]

    general_tags = []
    character_tags = []
    ratings = {}
    detailed_rows = []

    for idx, prob in enumerate(outputs):
        tag_name, category = category_map[idx]
        formatted_tag = tag_name.replace("_", " ") if replace_underscores else tag_name

        if formatted_tag in exclude_tags or tag_name in exclude_tags:
            continue

        prob_val = float(prob)

        # Rating tags (Category 9)
        if category == 9:
            ratings[formatted_tag] = round(prob_val, 4)

        # General tags (Category 0)
        elif category == 0 and prob_val >= general_thresh:
            general_tags.append((formatted_tag, prob_val))
            detailed_rows.append({"Tag": formatted_tag, "Category": "General", "Score": round(prob_val, 4)})

        # Character tags (Category 1)
        elif category == 1 and prob_val >= character_thresh:
            character_tags.append((formatted_tag, prob_val))
            detailed_rows.append({"Tag": formatted_tag, "Category": "Character", "Score": round(prob_val, 4)})

    # Sort tags by score descending
    general_tags.sort(key=lambda x: x[1], reverse=True)
    character_tags.sort(key=lambda x: x[1], reverse=True)

    gen_tag_strings = [t[0] for t in general_tags]
    char_tag_strings = [t[0] for t in character_tags]

    # Combine prefix tags, character tags, and general tags in sequence
    combined_tags = prefix_tags + char_tag_strings + gen_tag_strings
    all_tag_string = ", ".join(combined_tags)
    df_results = pd.DataFrame(detailed_rows)

    return all_tag_string, ratings, df_results


def save_caption_file(image_path: str, caption_text: str, extension: str = ".txt"):
    """
    Saves caption string to a text file matching the image filename.
    """
    base_path = os.path.splitext(image_path)[0]
    out_file = f"{base_path}{extension}"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(caption_text.strip())
    return out_file
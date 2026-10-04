import os
import csv
import torch
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    CLIPProcessor, CLIPModel,
    AutoProcessor, AutoModelForCausalLM,
    BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering,
)
from huggingface_hub import hf_hub_download

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import streamlit as st
except ImportError:
    st = None


# ==========================================
# 1. DEVICE MANAGEMENT
# ==========================================
def get_device() -> str:
    """Determines the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass
    return "cpu"


# ==========================================
# 2. CLIP (Zero-Shot Classification)
# ==========================================
_CLIP_MODEL = None
_CLIP_PROC = None

def run_clip_zero_shot(
    image: Image.Image,
    candidate_labels: list,
    model_id: str = "openai/clip-vit-base-patch32",
) -> pd.DataFrame:
    """
    Computes zero-shot class probabilities and cosine similarities for a list of candidate labels.
    """
    global _CLIP_MODEL, _CLIP_PROC
    device = get_device()
    
    if _CLIP_MODEL is None or _CLIP_MODEL.name_or_path != model_id:
        _CLIP_PROC = CLIPProcessor.from_pretrained(model_id)
        _CLIP_MODEL = CLIPModel.from_pretrained(model_id).to(device)
        
    inputs = _CLIP_PROC(
        text=candidate_labels,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(device)
    
    outputs = _CLIP_MODEL(**inputs)
    
    # Calculate softmax probabilities across candidate labels
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).cpu().detach().numpy()[0]
    
    # Calculate raw cosine similarities between image and text features
    image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
    cosine_sims = (image_embeds @ text_embeds.T).cpu().detach().numpy()[0]
    
    df_results = pd.DataFrame({
        "Label": candidate_labels,
        "Probability": [round(float(p), 4) for p in probs],
        "Cosine Similarity": [round(float(s), 4) for s in cosine_sims],
    }).sort_values(by="Probability", ascending=False).reset_index(drop=True)
    
    return df_results


# ==========================================
# 3. FLORENCE-2 (Vision-Language Tasks)
# ==========================================
_FLORENCE2_MODEL = None
_FLORENCE2_PROCESSOR = None
_LOADED_MODEL_ID = None

# Task prompt mapping supported by Florence-2
FLORENCE2_TASKS = {
    "Caption": "<CAPTION>",
    "Detailed Caption": "<DETAILED_CAPTION>",
    "More Detailed Caption": "<MORE_DETAILED_CAPTION>",
    "Object Detection": "<OD>",
    "Dense Region Caption": "<DENSE_REGION_CAPTION>",
    "Region Proposal": "<REGION_PROPOSAL>",
    "OCR": "<OCR>",
    "OCR with Region": "<OCR_WITH_REGION>",
    "Caption to Phrase Grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
}

def load_florence2_model(model_id: str = "microsoft/Florence-2-base"):
    """Loads Florence-2 model and processor from Hugging Face."""
    global _FLORENCE2_MODEL, _FLORENCE2_PROCESSOR, _LOADED_MODEL_ID
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    if _FLORENCE2_MODEL is None or _LOADED_MODEL_ID != model_id:
        _FLORENCE2_PROCESSOR = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        _FLORENCE2_MODEL = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        _LOADED_MODEL_ID = model_id
        
    return _FLORENCE2_MODEL, _FLORENCE2_PROCESSOR, device, dtype

def plot_boxes_on_image(image: Image.Image, detections: dict) -> Image.Image:
    """Overlays detected bounding boxes and text labels onto the input image."""
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)
    bboxes = detections.get("bboxes", [])
    labels = detections.get("labels", [])
    
    for bbox, label in zip(bboxes, labels):
        draw.rectangle(bbox, outline="red", width=3)
        if label:
            draw.text((bbox[0] + 4, bbox[1] + 4), str(label), fill="yellow")
            
    return annotated_img

def run_florence2_task(
    image: Image.Image,
    task_name: str,
    text_input: str = "",
    model_id: str = "microsoft/Florence-2-base",
    max_new_tokens: int = 1024,
    num_beams: int = 3,
):
    """
    Runs inference on Florence-2 for a specific vision-language task.
    Returns: tuple: (raw_generated_text, parsed_result_data, annotated_image_or_None)
    """
    model, processor, device, dtype = load_florence2_model(model_id)
    
    # Resolve task token string
    prompt = FLORENCE2_TASKS.get(task_name, "<CAPTION>")
    if task_name == "Caption to Phrase Grounding" and text_input.strip():
        prompt += text_input.strip()
        
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, dtype)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Post-process generation using Florence2Processor
    parsed_answer = processor.post_process_generation(
        generated_text, task=prompt, image_size=(image.width, image.height)
    )
    
    # Draw bounding boxes if output contains target coordinate data
    annotated_image = None
    if isinstance(parsed_answer, dict) and prompt in parsed_answer:
        task_data = parsed_answer[prompt]
        if isinstance(task_data, dict) and "bboxes" in task_data:
            annotated_image = plot_boxes_on_image(image, task_data)
            
    return generated_text, parsed_answer, annotated_image


# ==========================================
# 4. BLIP (Captioning & VQA)
# ==========================================
_BLIP_CAPTION_MODEL = None
_BLIP_CAPTION_PROC = None
_BLIP_VQA_MODEL = None
_BLIP_VQA_PROC = None

def generate_blip_caption(
    image: Image.Image,
    model_id: str = "Salesforce/blip-image-captioning-base",
    conditional_prompt: str = "",
    max_new_tokens: int = 50,
) -> str:
    """Generates a descriptive caption for the provided PIL Image."""
    global _BLIP_CAPTION_MODEL, _BLIP_CAPTION_PROC
    device = get_device()
    
    if _BLIP_CAPTION_MODEL is None or _BLIP_CAPTION_MODEL.name_or_path != model_id:
        _BLIP_CAPTION_PROC = BlipProcessor.from_pretrained(model_id)
        _BLIP_CAPTION_MODEL = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
        
    if conditional_prompt.strip():
        inputs = _BLIP_CAPTION_PROC(image, conditional_prompt, return_tensors="pt").to(device)
    else:
        inputs = _BLIP_CAPTION_PROC(image, return_tensors="pt").to(device)
        
    out = _BLIP_CAPTION_MODEL.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = _BLIP_CAPTION_PROC.decode(out[0], skip_special_tokens=True)
    return caption

def answer_blip_vqa(
    image: Image.Image,
    question: str,
    model_id: str = "Salesforce/blip-vqa-base",
) -> str:
    """Answers a natural language question about an input image."""
    global _BLIP_VQA_MODEL, _BLIP_VQA_PROC
    device = get_device()
    
    if _BLIP_VQA_MODEL is None or _BLIP_VQA_MODEL.name_or_path != model_id:
        _BLIP_VQA_PROC = BlipProcessor.from_pretrained(model_id)
        _BLIP_VQA_MODEL = BlipForQuestionAnswering.from_pretrained(model_id).to(device)
        
    inputs = _BLIP_VQA_PROC(image, question, return_tensors="pt").to(device)
    out = _BLIP_VQA_MODEL.generate(**inputs)
    answer = _BLIP_VQA_PROC.decode(out[0], skip_special_tokens=True)
    return answer


# ==========================================
# 5. WD14 (Anime/Booru Tagging)
# ==========================================
_WD14_SESSIONS = {}
_WD14_TAGS_DATA = {}

AVAILABLE_WD14_MODELS = {
    "wd-v1-4-convnextv2-tagger-v2": "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
    "wd-v1-4-swinv2-tagger-v2": "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
    "wd-v1-4-vit-tagger-v2": "SmilingWolf/wd-v1-4-vit-tagger-v2",
    "wd-eva02-large-tagger-v3": "SmilingWolf/wd-eva02-large-tagger-v3",
}

def _load_wd14_model_core(model_name: str, model_root: str = None):
    """Core logic to load WD14 ONNX model and tags."""
    if ort is None:
        raise ImportError("onnxruntime is required for WD14 Tagger.")
        
    repo_id = AVAILABLE_WD14_MODELS.get(model_name, "SmilingWolf/wd-v1-4-convnextv2-tagger-v2")
    
    if model_root:
        local_dir = os.path.join(model_root, model_name)
        os.makedirs(local_dir, exist_ok=True)
        model_path = hf_hub_download(repo_id=repo_id, filename="model.onnx", local_dir=local_dir)
        tags_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv", local_dir=local_dir)
    else:
        model_path = hf_hub_download(repo_id=repo_id, filename="model.onnx")
        tags_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv")
        
    providers = ["CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    selected = [p for p in providers if p in available]
    
    session = ort.InferenceSession(model_path, providers=selected)
    
    tags = []
    category_map = {}
    with open(tags_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            tag_name = row[1]
            category = int(row[2])
            tags.append(tag_name)
            category_map[i] = (tag_name, category)
            
    return session, (tags, category_map)

# Apply Streamlit caching if Streamlit is available, otherwise fallback to standard execution
if st is not None:
    load_wd14_model = st.cache_resource(show_spinner="Loading WD14 ONNX model into memory...")(_load_wd14_model_core)
else:
    load_wd14_model = _load_wd14_model_core

def preprocess_image_wd14(image: Image.Image, target_size: int = 448) -> np.ndarray:
    """Preprocess image for WD14 model input: Resize with padding and normalize."""
    img = np.array(image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    max_dim = max(h, w)
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2
    
    img_padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    img_resized = cv2.resize(img_padded, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_tensor = np.expand_dims(img_rgb, axis=0)
    return img_tensor

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
    """Runs WD14 tagger on an image and returns formatted tag strings and dataframe breakdown."""
    if prefix_tags is None: prefix_tags = []
    if exclude_tags is None: exclude_tags = []
    
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
        
        if category == 9: # Rating tags
            ratings[formatted_tag] = round(prob_val, 4)
        elif category == 0 and prob_val >= general_thresh: # General tags
            general_tags.append((formatted_tag, prob_val))
            detailed_rows.append({"Tag": formatted_tag, "Category": "General", "Score": round(prob_val, 4)})
        elif category == 1 and prob_val >= character_thresh: # Character tags
            character_tags.append((formatted_tag, prob_val))
            detailed_rows.append({"Tag": formatted_tag, "Category": "Character", "Score": round(prob_val, 4)})
            
    general_tags.sort(key=lambda x: x[1], reverse=True)
    character_tags.sort(key=lambda x: x[1], reverse=True)
    
    gen_tag_strings = [t[0] for t in general_tags]
    char_tag_strings = [t[0] for t in character_tags]
    
    combined_tags = prefix_tags + char_tag_strings + gen_tag_strings
    all_tag_string = ", ".join(combined_tags)
    df_results = pd.DataFrame(detailed_rows)
    
    return all_tag_string, ratings, df_results

def save_caption_file(image_path: str, caption_text: str, extension: str = ".txt"):
    """Saves caption string to a text file matching the image filename."""
    base_path = os.path.splitext(image_path)[0]
    out_file = f"{base_path}{extension}"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(caption_text.strip())
    return out_file
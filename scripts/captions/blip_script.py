import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)

_BLIP_CAPTION_MODEL = None
_BLIP_CAPTION_PROC = None
_BLIP_VQA_MODEL = None
_BLIP_VQA_PROC = None


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass
    return "cpu"


def generate_blip_caption(
    image: Image.Image,
    model_id: str = "Salesforce/blip-image-captioning-base",
    conditional_prompt: str = "",
    max_new_tokens: int = 50,
) -> str:
    """
    Generates a descriptive caption for the provided PIL Image.
    """
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
    """
    Answers a natural language question about an input image.
    """
    global _BLIP_VQA_MODEL, _BLIP_VQA_PROC
    device = get_device()

    if _BLIP_VQA_MODEL is None or _BLIP_VQA_MODEL.name_or_path != model_id:
        _BLIP_VQA_PROC = BlipProcessor.from_pretrained(model_id)
        _BLIP_VQA_MODEL = BlipForQuestionAnswering.from_pretrained(model_id).to(device)

    inputs = _BLIP_VQA_PROC(image, question, return_tensors="pt").to(device)
    out = _BLIP_VQA_MODEL.generate(**inputs)
    answer = _BLIP_VQA_PROC.decode(out[0], skip_special_tokens=True)
    return answer
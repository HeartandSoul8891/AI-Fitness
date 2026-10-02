import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

_CLIP_MODEL = None
_CLIP_PROC = None


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass
    return "cpu"


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
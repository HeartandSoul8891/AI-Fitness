import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM

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


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass
    return "cpu"


def load_florence2_model(model_id: str = "microsoft/Florence-2-base"):
    """
    Loads Florence-2 model and processor from Hugging Face.
    """
    global _FLORENCE2_MODEL, _FLORENCE2_PROCESSOR, _LOADED_MODEL_ID

    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    if _FLORENCE2_MODEL is None or _LOADED_MODEL_ID != model_id:
        _FLORENCE2_PROCESSOR = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        _FLORENCE2_MODEL = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        _LOADED_MODEL_ID = model_id

    return _FLORENCE2_MODEL, _FLORENCE2_PROCESSOR, device, dtype


def plot_boxes_on_image(image: Image.Image, detections: dict) -> Image.Image:
    """
    Overlays detected bounding boxes and text labels onto the input image.
    """
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)

    bboxes = detections.get("bboxes", [])
    labels = detections.get("labels", [])

    for bbox, label in zip(bboxes, labels):
        # Coordinates: [x1, y1, x2, y2]
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
    
    Returns:
        tuple: (raw_generated_text, parsed_result_data, annotated_image_or_None)
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
        generated_text,
        task=prompt,
        image_size=(image.width, image.height),
    )

    # Draw bounding boxes if output contains target coordinate data
    annotated_image = None
    if isinstance(parsed_answer, dict) and prompt in parsed_answer:
        task_data = parsed_answer[prompt]
        if isinstance(task_data, dict) and "bboxes" in task_data:
            annotated_image = plot_boxes_on_image(image, task_data)

    return generated_text, parsed_answer, annotated_image
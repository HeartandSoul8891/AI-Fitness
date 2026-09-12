import os
from pathlib import Path
import streamlit as st

from tabs.settings_tab import load_settings, DEFAULT_DATASETS_PATH, DEFAULT_OUTPUT_PATH
from scripts.textual_inversion.TI_script import (
    get_hardware_report,
    train_textual_inversion,
)


def get_datasets_path() -> str:
    """Retrieve datasets folder from session state, settings, or fallback."""
    if st.session_state.get("settings_datasets_folder", "").strip():
        return st.session_state["settings_datasets_folder"]
    settings = load_settings()
    return settings.get("datasets_folder", DEFAULT_DATASETS_PATH)


def get_output_path() -> str:
    """Retrieve output folder from session state, settings, or fallback."""
    if st.session_state.get("settings_output_folder", "").strip():
        return st.session_state["settings_output_folder"]
    settings = load_settings()
    return settings.get("output_folder", DEFAULT_OUTPUT_PATH)


def get_checkpoint_models() -> list[str]:
    """Retrieve available checkpoint models recursively from the configured checkpoint folder."""
    checkpoint_dir = st.session_state.get("settings_checkpoint_folder", "").strip()
    if not checkpoint_dir:
        settings = load_settings()
        checkpoint_dir = settings.get("checkpoint_folder", "")

    models = []
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        for root, dirs, files in os.walk(checkpoint_dir):
            # 1. Look for Diffusers directory models (directories containing model_index.json)
            if "model_index.json" in files:
                models.append(root)
                dirs.clear()  # Stop traversing deeper into a valid diffusers model directory
                continue

            # 2. Look for single file checkpoints (.safetensors, .ckpt, .bin)
            for file in files:
                if file.lower().endswith((".safetensors", ".ckpt", ".bin")):
                    models.append(os.path.join(root, file))

    return sorted(models)


def _hardware_section():
    hw = get_hardware_report()
    with st.container(border=True):
        st.subheader("Hardware")
        c1, c2, c3 = st.columns(3)
        c1.metric("Backend", hw["backend"])
        c2.metric("Device", hw["name"])
        c3.metric("Torch", hw["torch"])

        if hw["backend"] == "ROCm":
            st.caption(f"ROCm/HIP: {hw.get('hip', 'detected')}")
        elif hw["backend"] == "CUDA":
            st.caption(f"CUDA runtime: {hw.get('cuda', 'detected')}")
        elif hw["backend"] == "CPU":
            st.warning("No GPU was detected. Training will run on CPU.")


def render_ui():
    st.title("Textual Inversion Training")
    st.caption(
        "Train a lightweight Textual Inversion embedding for Stable Diffusion "
        "using the configured AI-Fitness dataset/output locations."
    )

    datasets_dir = get_datasets_path()
    output_dir = get_output_path()

    st.info(
        f"**Datasets Directory:** `{datasets_dir}`  \n"
        f"**Output Directory:** `{output_dir}`"
    )

    _hardware_section()

    with st.form("textual_inversion_form"):
        st.subheader("1. Concept & Model Setup")
        col1, col2 = st.columns(2)

        # Retrieve models from settings checkpoint folder
        available_models = get_checkpoint_models()
        default_hf_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"

        with col1:
            if available_models:
                # Provide a drop-down with local checkpoint models + Hugging Face fallback
                model_options = available_models + [default_hf_model, "Custom Path..."]
                selected_model_option = st.selectbox(
                    "Pretrained Model Selection",
                    options=model_options,
                    help="Select a model from your configured Checkpoints folder or use a Hugging Face ID.",
                )

                if selected_model_option == "Custom Path...":
                    pretrained_model_name_or_path = st.text_input(
                        "Custom Pretrained Model Name or Path",
                        value=default_hf_model,
                        help="Enter a Hugging Face model ID or custom local model path.",
                    )
                else:
                    pretrained_model_name_or_path = selected_model_option
            else:
                # Fallback if no checkpoint folder models are found
                pretrained_model_name_or_path = st.text_input(
                    "Pretrained Model Name or Path",
                    value=default_hf_model,
                    help="Hugging Face model ID or local Diffusers-format SD1.x model folder.",
                )

            placeholder_token = st.text_input(
                "Placeholder Token",
                value="<concept>",
                help="A new token that does not already exist in the tokenizer.",
            )
            initializer_token = st.text_input(
                "Initializer Token",
                value="object",
                help="Must resolve to exactly one tokenizer token.",
            )
            learnable_property = st.selectbox(
                "Learnable Property",
                ["object", "style"],
                index=0,
            )

        with col2:
            revision = st.text_input("Model Revision", value="main")
            tokenizer_name = st.text_input(
                "Tokenizer Name",
                value="",
                help="Optional separate tokenizer path/model ID.",
            )

            available_datasets = []
            if os.path.isdir(datasets_dir):
                available_datasets = sorted(
                    d for d in os.listdir(datasets_dir)
                    if os.path.isdir(os.path.join(datasets_dir, d))
                )

            selected_dataset_folder = ""
            if available_datasets:
                selected_dataset_folder = st.selectbox(
                    "Select Dataset Subfolder",
                    options=[""] + available_datasets,
                )

            default_train_data_dir = (
                os.path.join(datasets_dir, selected_dataset_folder)
                if selected_dataset_folder
                else datasets_dir
            )

            train_data_dir = st.text_input(
                "Train Data Directory",
                value=default_train_data_dir,
            )
            output_dir_input = st.text_input(
                "Output Directory",
                value=os.path.join(output_dir, "textual_inversion_output"),
            )

        st.divider()
        st.subheader("2. Dataset & Processing")
        c3, c4, c5 = st.columns(3)

        with c3:
            resolution = st.number_input("Resolution", value=512, min_value=64, step=64)
            center_crop = st.checkbox("Center Crop", value=False)

        with c4:
            repeats = st.number_input("Repeats", value=100, min_value=1)
            repeats_as_epoch = st.checkbox(
                "Use Epochs as Training Length",
                value=False,
                help="When enabled, Max Train Steps is calculated from Num Train Epochs.",
            )

        with c5:
            num_vectors = st.number_input("Num Vectors", value=1, min_value=1)
            concept_feature = st.text_input(
                "Concept Feature",
                value="",
                help="Optional metadata describing the concept.",
            )

        st.divider()
        st.subheader("3. Training & Hardware")
        c6, c7, c8 = st.columns(3)

        with c6:
            train_batch_size = st.number_input("Train Batch Size", value=1, min_value=1)
            gradient_accumulation_steps = st.number_input(
                "Gradient Accumulation Steps", value=1, min_value=1
            )
            learning_rate = st.number_input(
                "Learning Rate", value=5e-4, format="%.6f", min_value=1e-8
            )
            scale_lr = st.checkbox("Scale Learning Rate", value=False)

        with c7:
            max_train_steps = st.number_input("Max Train Steps", value=2000, step=100, min_value=1)
            num_train_epochs = st.number_input("Num Train Epochs", value=100, step=10, min_value=1)
            lr_scheduler = st.selectbox(
                "LR Scheduler",
                [
                    "constant",
                    "linear",
                    "cosine",
                    "cosine_with_restarts",
                    "polynomial",
                    "constant_with_warmup",
                ],
            )
            lr_warmup_steps = st.number_input("LR Warmup Steps", value=0, min_value=0)

        with c8:
            mixed_precision = st.selectbox(
                "Mixed Precision",
                ["no", "fp16", "bf16"],
                index=0,
                help="Use fp16/bf16 only when supported by the selected GPU/runtime.",
            )
            gradient_checkpointing = st.checkbox("Gradient Checkpointing", value=False)
            allow_tf32 = st.checkbox("Allow TF32", value=False)
            dataloader_num_workers = st.number_input(
                "Dataloader Workers", value=0, min_value=0
            )

        st.divider()
        st.subheader("4. Validation & Checkpoints")
        c9, c10 = st.columns(2)

        with c9:
            validation_prompt = st.text_input(
                "Validation Prompt",
                value="",
                help="Example: a portrait photo of <concept>, studio lighting",
            )
            num_validation_images = st.number_input(
                "Validation Images", value=4, min_value=1
            )
            validation_steps = st.number_input(
                "Validation Every N Steps", value=100, min_value=1
            )

        with c10:
            save_steps = st.number_input("Save Embedding Every N Steps", value=500, min_value=1)
            checkpointing_steps = st.number_input(
                "Checkpoint State Every N Steps", value=500, min_value=1
            )
            checkpoints_total_limit = st.number_input(
                "Checkpoint Limit", value=0, min_value=0
            )
            resume_from_checkpoint = st.text_input(
                "Resume From Checkpoint",
                value="",
                help="Checkpoint folder, or 'latest' inside the output folder.",
            )

        seed = st.number_input("Seed", value=42, min_value=0, step=1)

        start = st.form_submit_button(
            "Start Training",
            type="primary",
            width="stretch",
        )

    if start:
        config = {
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "revision": revision,
            "tokenizer_name": tokenizer_name or None,
            "train_data_dir": train_data_dir,
            "placeholder_token": placeholder_token,
            "initializer_token": initializer_token,
            "learnable_property": learnable_property,
            "num_vectors": int(num_vectors),
            "concept_feature": concept_feature or None,
            "output_dir": output_dir_input,
            "resolution": int(resolution),
            "center_crop": center_crop,
            "repeats": int(repeats),
            "repeats_as_epoch": repeats_as_epoch,
            "train_batch_size": int(train_batch_size),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "learning_rate": float(learning_rate),
            "scale_lr": scale_lr,
            "max_train_steps": int(max_train_steps),
            "num_train_epochs": int(num_train_epochs),
            "lr_scheduler": lr_scheduler,
            "lr_warmup_steps": int(lr_warmup_steps),
            "mixed_precision": mixed_precision,
            "gradient_checkpointing": gradient_checkpointing,
            "allow_tf32": allow_tf32,
            "dataloader_num_workers": int(dataloader_num_workers),
            "validation_prompt": validation_prompt or None,
            "num_validation_images": int(num_validation_images),
            "validation_steps": int(validation_steps),
            "save_steps": int(save_steps),
            "checkpointing_steps": int(checkpointing_steps),
            "checkpoints_total_limit": (
                int(checkpoints_total_limit) if checkpoints_total_limit > 0 else None
            ),
            "resume_from_checkpoint": resume_from_checkpoint or None,
            "seed": int(seed),
            "save_safetensors": True,
            "save_a1111_pt": True,
        }

        st.subheader("Training Log")
        log_box = st.empty()
        messages = []

        def progress(message):
            messages.append(message)
            log_box.code("\n".join(messages[-30:]))

        try:
            result = train_textual_inversion(config, progress_callback=progress)
            st.success("Textual Inversion training completed.")
            st.json(result)
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            st.exception(exc)


if __name__ == "__main__":
    st.set_page_config(page_title="Textual Inversion", layout="wide")
    render_ui()
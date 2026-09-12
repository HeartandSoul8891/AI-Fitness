import os
import streamlit as st
from tabs.settings_tab import load_settings, DEFAULT_DATASETS_PATH, DEFAULT_OUTPUT_PATH

def get_datasets_path() -> str:
    """Retrieve datasets folder from session_state, settings.json, or fallback default."""
    if "settings_datasets_folder" in st.session_state and st.session_state["settings_datasets_folder"].strip():
        return st.session_state["settings_datasets_folder"]
    settings = load_settings()
    return settings.get("datasets_folder", DEFAULT_DATASETS_PATH)

def get_output_path() -> str:
    """Retrieve output folder from session_state, settings.json, or fallback default."""
    if "settings_output_folder" in st.session_state and st.session_state["settings_output_folder"].strip():
        return st.session_state["settings_output_folder"]
    settings = load_settings()
    return settings.get("output_folder", DEFAULT_OUTPUT_PATH)

def render_ui():
    st.title("Hypernetwork Training")
    st.caption("Configure parameters for training a Hypernetwork module to steer UNet attention layers.")

    datasets_dir = get_datasets_path()
    output_dir = get_output_path()

    # Informational banner showing resolved path defaults
    st.info(f"**Datasets Directory:** `{datasets_dir}` | **Output Directory:** `{output_dir}`")

    with st.form("hypernetwork_form"):
        # Section 1: Base Model & Path Configuration
        st.subheader("1. Base Model & Dataset Setup")
        col1, col2 = st.columns(2)
        with col1:
            pretrained_model_name_or_path = st.text_input(
                "Pretrained Model Name or Path",
                value="runwayml/stable-diffusion-v1-5",
                help="Path to pretrained base model or Hugging Face hub model ID"
            )
            hypernetwork_name = st.text_input(
                "Hypernetwork Name",
                value="my_custom_style",
                help="Identifier or filename tag for the generated hypernetwork weights"
            )
            instance_prompt = st.text_input(
                "Instance Prompt",
                value="a photo of sks style",
                help="Prompt specifying the target concept or style"
            )

        with col2:
            revision = st.text_input(
                "Model Revision",
                value="main",
                help="Revision branch of pretrained model identifier"
            )
            
            # Auto-populate dataset directory selection if subfolders exist
            available_datasets = []
            if os.path.exists(datasets_dir):
                available_datasets = [d for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]
            
            if available_datasets:
                selected_dataset_folder = st.selectbox("Select Dataset Subfolder", options=[""] + available_datasets)
                default_train_data_dir = os.path.join(datasets_dir, selected_dataset_folder) if selected_dataset_folder else datasets_dir
            else:
                default_train_data_dir = datasets_dir

            train_data_dir = st.text_input(
                "Train Data Directory",
                value=default_train_data_dir,
                help="Folder containing training images and caption txt files"
            )

            output_dir_input = st.text_input(
                "Output Directory",
                value=os.path.join(output_dir, "hypernetwork_output"),
                help="Directory where trained hypernetwork weights and logs will be saved"
            )

        st.divider()

        # Section 2: Hypernetwork Architecture Options
        st.subheader("2. Hypernetwork Network Architecture")
        col3, col4, col5 = st.columns(3)
        with col3:
            layer_structure = st.text_input(
                "Layer Structure",
                value="1, 2, 1",
                help="Comma-separated multiplier/hidden dimension scaling (e.g. '1, 2, 1')"
            )
            activation_function = st.selectbox(
                "Activation Function",
                options=["relu", "leakyrelu", "elu", "linear", "mish", "swish"],
                index=0,
                help="Activation function for intermediate MLP layers"
            )
        with col4:
            add_layer_norm = st.checkbox("Add Layer Normalization", value=True, help="Apply LayerNorm for stability")
            use_dropout = st.checkbox("Use Dropout", value=False, help="Enable dropout layers inside hypernetwork MLPs")
        with col5:
            dropout_rate = st.number_input("Dropout Rate", value=0.1, min_value=0.0, max_value=0.9, step=0.05)
            activate_output = st.selectbox("Output Activation", options=["linear", "sigmoid", "tanh"], index=0)

        st.divider()

        # Section 3: Hyperparameters & Training Settings
        st.subheader("3. Training Hyperparameters")
        col6, col7, col8 = st.columns(3)
        with col6:
            resolution = st.number_input("Resolution", value=512, step=64)
            train_batch_size = st.number_input("Train Batch Size", value=1, min_value=1)
            gradient_accumulation_steps = st.number_input("Gradient Accumulation Steps", value=1, min_value=1)

        with col7:
            learning_rate = st.number_input("Learning Rate", value=1e-4, format="%.6f")
            max_train_steps = st.number_input("Max Train Steps", value=5000, step=500)
            lr_scheduler = st.selectbox(
                "LR Scheduler",
                options=["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup"],
                index=0
            )

        with col8:
            mixed_precision = st.selectbox("Mixed Precision", options=["no", "fp16", "bf16"], index=1)
            gradient_checkpointing = st.checkbox("Gradient Checkpointing", value=True)
            allow_tf32 = st.checkbox("Allow TF32", value=False)

        st.divider()

        # Section 4: Checkpointing & Validation
        st.subheader("4. Checkpointing & Validation")
        col9, col10 = st.columns(2)
        with col9:
            validation_prompt = st.text_input(
                "Validation Prompt",
                value="a photo of sks style portrait, highly detailed",
                help="Prompt used to generate preview images during training"
            )
            validation_steps = st.number_input("Validation Steps", value=250, min_value=1)
            num_validation_images = st.number_input("Num Validation Images", value=4, min_value=1)

        with col10:
            save_steps = st.number_input("Save Steps", value=500, min_value=1, help="Save hypernetwork checkpoint every X steps")
            checkpoints_total_limit = st.number_input("Checkpoints Total Limit", value=5, min_value=0)
            resume_from_checkpoint = st.text_input("Resume From Checkpoint", value="", help="Path to existing `.pt` file or checkpoint folder")

        st.divider()

        # Submit button
        submit_button = st.form_submit_button("Start Hypernetwork Training", type="primary", use_container_width=True)

    if submit_button:
        # Collect UI outputs into executable config dictionary
        config = {
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "revision": revision,
            "hypernetwork_name": hypernetwork_name,
            "instance_prompt": instance_prompt,
            "train_data_dir": train_data_dir,
            "output_dir": output_dir_input,
            "layer_structure": [int(x.strip()) for x in layer_structure.split(",") if x.strip().isdigit()],
            "activation_function": activation_function,
            "add_layer_norm": add_layer_norm,
            "use_dropout": use_dropout,
            "dropout_rate": dropout_rate,
            "activate_output": activate_output,
            "resolution": resolution,
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "max_train_steps": max_train_steps,
            "lr_scheduler": lr_scheduler,
            "mixed_precision": mixed_precision,
            "gradient_checkpointing": gradient_checkpointing,
            "allow_tf32": allow_tf32,
            "validation_prompt": validation_prompt if validation_prompt else None,
            "validation_steps": validation_steps,
            "num_validation_images": num_validation_images,
            "save_steps": save_steps,
            "checkpoints_total_limit": checkpoints_total_limit if checkpoints_total_limit > 0 else None,
            "resume_from_checkpoint": resume_from_checkpoint if resume_from_checkpoint else None,
        }

        st.success("Hypernetwork training configuration generated!")
        st.json(config)

if __name__ == "__main__":
    st.set_page_config(page_title="Hypernetwork Training", layout="wide")
    render_ui()
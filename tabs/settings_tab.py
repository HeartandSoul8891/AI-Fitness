import json
import os
import streamlit as st

# Application root defaults for local datasets & outputs
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASETS_PATH = os.path.join(APP_ROOT, "datasets")
DEFAULT_OUTPUT_PATH = os.path.join(APP_ROOT, "output")

COMFY_FOLDERS = {
    "checkpoint_folder": "checkpoints",
    "clip_folder": "clip",
    "clip_vision_folder": "clip_vision",
    "controlnet_folder": "controlnet",
    "diffusion_models_folder": "diffusion_models",
    "embeddings_folder": "embeddings",
    "hypernetworks_folder": "hypernetworks",
    "loras_folder": "loras",
    "text_encoders_folder": "text_encoders",
    "unet_folder": "unet",
    "upscale_models_folder": "upscale_models",
    "vae_folder": "vae",
    "ultralytics_bbox_folder": "ultralytics/bbox",
    "ultralytics_segm_folder": "ultralytics/segm",
}

def load_settings():
    try:
        with open("settings.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def update_paths_from_root():
    root = st.session_state.get("root_folder", "").strip()
    if not root:
        return

    for key, subfolder in COMFY_FOLDERS.items():
        current_path = st.session_state.get(f"settings_{key}", "").strip()
        if not current_path or current_path.startswith(st.session_state.get("_prev_root", "")):
            st.session_state[f"settings_{key}"] = os.path.join(root, subfolder)

    st.session_state["_prev_root"] = root

def save_settings():
    settings = {
        "root_folder": st.session_state.get("root_folder", ""),
        "datasets_folder": st.session_state.get("settings_datasets_folder", "").strip() or DEFAULT_DATASETS_PATH,
        "output_folder": st.session_state.get("settings_output_folder", "").strip() or DEFAULT_OUTPUT_PATH,
    }

    for key in COMFY_FOLDERS:
        settings[key] = st.session_state.get(f"settings_{key}", "")

    with open("settings.json", "w") as f:
        json.dump(settings, f, indent=4)

    st.success("Settings saved successfully!")

def render_ui():
    st.title("Settings Configuration")

    saved_settings = load_settings()

    if "root_folder" not in st.session_state:
        st.session_state["root_folder"] = saved_settings.get("root_folder", "")

    # Initialize datasets and output paths with fallback to local app root defaults
    if "settings_datasets_folder" not in st.session_state:
        st.session_state["settings_datasets_folder"] = saved_settings.get("datasets_folder", DEFAULT_DATASETS_PATH)

    if "settings_output_folder" not in st.session_state:
        st.session_state["settings_output_folder"] = saved_settings.get("output_folder", DEFAULT_OUTPUT_PATH)

    for key in COMFY_FOLDERS:
        session_key = f"settings_{key}"
        if session_key not in st.session_state:
            st.session_state[session_key] = saved_settings.get(key, "")

    st.text_input(
        "Root Folder", key="root_folder", on_change=update_paths_from_root
    )

    st.subheader("App Working Directories")
    col1, col2 = st.columns(2)
    with col1:
        st.text_input(
            "Datasets Folder Path",
            key="settings_datasets_folder",
            help="Leave empty or set path to default to local app root: ./datasets"
        )
        if st.button("Reset Datasets to Default"):
            st.session_state["settings_datasets_folder"] = DEFAULT_DATASETS_PATH

    with col2:
        st.text_input(
            "Output Folder Path",
            key="settings_output_folder",
            help="Leave empty or set path to default to local app root: ./output"
        )
        if st.button("Reset Output to Default"):
            st.session_state["settings_output_folder"] = DEFAULT_OUTPUT_PATH

    st.divider()

    st.subheader("Model Directory Paths")
    for key in COMFY_FOLDERS:
        label = key.replace("_", " ").title()
        st.text_input(label, key=f"settings_{key}")

    if st.button("Save Settings", type="primary", use_container_width=True):
        save_settings()

if __name__ == "__main__":
    render_ui()
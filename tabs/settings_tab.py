import json
import os
import streamlit as st
from pathlib import Path

# Application root defaults for local datasets & outputs
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASETS_PATH = os.path.join(APP_ROOT, "datasets")
DEFAULT_OUTPUT_PATH = os.path.join(APP_ROOT, "output")

# Setup user directory
BASE_DIR = Path(__file__).parent
USER_DIR = BASE_DIR / "user"
USER_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS_FILE = USER_DIR / "settings.json"
LEGACY_SETTINGS_FILE = BASE_DIR / "settings.json"

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
    """Loads settings.json from the user folder."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error reading settings.json: {e}")
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

def save_settings(settings_dict):
    """Saves settings_dict to user/settings.json."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings_dict, f, indent=4)
        st.toast("Application settings updated.", icon="⚙️")
    except Exception as e:
        st.error(f"Failed to save settings: {e}")

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
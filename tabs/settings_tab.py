import json
import os
import streamlit as st
from pathlib import Path

# Applicatie- en project roots instellen (één niveau omhoog vanuit de tabs/ map)
TAB_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = TAB_DIR.parent

APP_ROOT = str(PROJECT_ROOT)
DEFAULT_DATASETS_PATH = os.path.join(PROJECT_ROOT, "datasets")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output")
DEFAULT_TRAINING_PATH = os.path.join(PROJECT_ROOT, "training")

# Instellingen direct in root/user bewaren (in plaats van root/tabs/user)
USER_DIR = PROJECT_ROOT / "user"
USER_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS_FILE = USER_DIR / "settings.json"

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
    """Laadt settings.json vanuit de root/user map."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Fout bij het lezen van settings.json: {e}")
    return {}

def sync_paths_from_root():
    """Berekent en werkt paden in de session state bij op basis van de hoofdmap."""
    root = st.session_state.get("root_folder", "").strip()
    if not root:
        return

    for key, subfolder in COMFY_FOLDERS.items():
        session_key = f"settings_{key}"
        st.session_state[session_key] = os.path.normpath(os.path.join(root, subfolder))

def save_settings(settings_dict):
    """Slaat settings_dict op naar root/user/settings.json."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings_dict, f, indent=4)
        st.toast("Applicatie-instellingen bijgewerkt.", icon="⚙")
        st.success("Instellingen succesvol opgeslagen!")
    except Exception as e:
        st.error(f"Fout bij het opslaan van instellingen: {e}")

def render_ui():
    st.title("Settings Configuration")

    saved_settings = load_settings()

    if "root_folder" not in st.session_state:
        st.session_state["root_folder"] = saved_settings.get("root_folder", "")

    # Initialiseer werkmappen
    if "settings_datasets_folder" not in st.session_state:
        st.session_state["settings_datasets_folder"] = saved_settings.get("datasets_folder", DEFAULT_DATASETS_PATH)

    if "settings_output_folder" not in st.session_state:
        st.session_state["settings_output_folder"] = saved_settings.get("output_folder", DEFAULT_OUTPUT_PATH)

    if "settings_training_folder" not in st.session_state:
        st.session_state["settings_training_folder"] = saved_settings.get("training_folder", DEFAULT_TRAINING_PATH)

    # Initialiseer modelmappen
    for key in COMFY_FOLDERS:
        session_key = f"settings_{key}"
        if session_key not in st.session_state:
            saved_val = saved_settings.get(key, "")
            if not saved_val and st.session_state["root_folder"]:
                saved_val = os.path.normpath(os.path.join(st.session_state["root_folder"], COMFY_FOLDERS[key]))
            st.session_state[session_key] = saved_val

    # Invoer Root Folder
    st.text_input(
        "Root Folder",
        key="root_folder",
        on_change=sync_paths_from_root
    )

    # Opslaan-knop direct onder Root Folder
    if st.button("Save Settings", type="primary", use_container_width=True):
        if st.session_state.get("root_folder"):
            sync_paths_from_root()

        settings_dict = {
            "root_folder": st.session_state.get("root_folder", ""),
            "datasets_folder": st.session_state.get("settings_datasets_folder", DEFAULT_DATASETS_PATH),
            "output_folder": st.session_state.get("settings_output_folder", DEFAULT_OUTPUT_PATH),
            "training_folder": st.session_state.get("settings_training_folder", DEFAULT_TRAINING_PATH),
        }
        for key in COMFY_FOLDERS:
            settings_dict[key] = st.session_state.get(f"settings_{key}", "")

        save_settings(settings_dict)

    st.divider()

    st.subheader("App Working Directories")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text_input(
            "Datasets Folder Path",
            key="settings_datasets_folder",
            help="Standaard pad: AI-Fitness/datasets"
        )

    with col2:
        st.text_input(
            "Output Folder Path",
            key="settings_output_folder",
            help="Standaard pad: AI-Fitness/output"
        )

    with col3:
        st.text_input(
            "Training Folder Path",
            key="settings_training_folder",
            help="Standaard pad: AI-Fitness/training"
        )

    st.divider()

    st.subheader("Model Directory Paths")
    for key in COMFY_FOLDERS:
        label = key.replace("_", " ").title()
        st.text_input(label, key=f"settings_{key}")

if __name__ == "__main__":
    render_ui()
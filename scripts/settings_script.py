import json
import os
import streamlit as st

COMFY_FOLDERS = {
    "datasets_folder": "datasets",
    "output_folder": "output",
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
    settings = {"root_folder": st.session_state.get("root_folder", "")}

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

    for key in COMFY_FOLDERS:
        session_key = f"settings_{key}"
        if session_key not in st.session_state:
            st.session_state[session_key] = saved_settings.get(key, "")

    st.text_input(
        "Root Folder", key="root_folder", on_change=update_paths_from_root
    )

    st.divider()

    for key in COMFY_FOLDERS:
        label = key.replace("_", " ").title()
        st.text_input(label, key=f"settings_{key}")

    if st.button("Save Settings", type="primary", use_container_width=True):
        save_settings()

if __name__ == "__main__":
    render_ui()
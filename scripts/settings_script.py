import json
import os
from pathlib import Path
import streamlit as st

# ==========================================
# PATH RESOLUTION
# ==========================================
def get_project_root() -> Path:
    """Dynamically finds the project root directory."""
    current = Path(__file__).resolve().parent
    # If this script is inside a 'scripts' or 'tabs' folder, go up one level
    if current.name in ["scripts", "tabs", "src"]:
        current = current.parent
    return current

APP_ROOT = get_project_root()
USER_DIR = APP_ROOT / "user"
USER_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE = USER_DIR / "settings.json"

# ==========================================
# DEFAULT CONFIGURATION
# ==========================================
# This dictionary defines the exact structure of your settings.json.
# To add a new setting later, just add a new key-value pair here!
DEFAULT_SETTINGS = {
    "root_folder": str(APP_ROOT),
    "datasets_folder": str(APP_ROOT / "datasets"),
    "training_folder": str(APP_ROOT / "training"),
    "output_folder": str(APP_ROOT / "output"),
    "models_folder": str(APP_ROOT / "models"),
    "ultralytics_models_folder": str(APP_ROOT / "models" / "ultralytics"),
    "vision_models_folder": str(APP_ROOT / "models" / "vision"),
}

# ==========================================
# CORE FUNCTIONS
# ==========================================
def load_settings() -> dict:
    """Loads settings.json, merging with defaults to ensure all keys exist."""
    settings = DEFAULT_SETTINGS.copy()
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                # Update defaults with saved values (keeps new default keys if they don't exist in JSON)
                settings.update(saved)
        except Exception as e:
            st.error(f"Error reading settings.json: {e}")
    return settings

def save_settings(settings_dict: dict) -> tuple:
    """Saves settings_dict to settings.json."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings_dict, f, indent=4)
        return True, "Settings saved successfully!"
    except Exception as e:
        return False, f"Error saving settings: {e}"

def sync_paths_from_root(root_path: str) -> dict:
    """Generates a settings dictionary with paths relative to a new root."""
    if not root_path or not os.path.exists(root_path):
        return {}
    
    root = Path(root_path).resolve()
    return {
        "root_folder": str(root),
        "datasets_folder": str(root / "datasets"),
        "training_folder": str(root / "training"),
        "output_folder": str(root / "output"),
        "models_folder": str(root / "models"),
        "ultralytics_models_folder": str(root / "models" / "ultralytics"),
        "vision_models_folder": str(root / "models" / "vision"),
    }
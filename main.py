"""
main.py
Unified YOLO & Vision Suite - Main Entry Point
Run this file using: streamlit run main.py
"""

import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Unified YOLO & Vision Suite",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. IMPORT GUI MODULES
# ==========================================
# We use try/except blocks to prevent the whole app from crashing 
# if one of the UI scripts is missing or has a syntax error.

try:
    from UI import settings_gui
except ImportError:
    settings_gui = None

try:
    from UI import captions_gui
except ImportError:
    captions_gui = None

try:
    from UI import datasets_gui
except ImportError:
    datasets_gui = None

try:
    # Based on your tree, it's named yolo_tagger_gui.py
    from UI import yolo_tagger_gui 
except ImportError:
    try:
        # Fallback if you named it tagger_gui.py
        from UI import tagger_gui as yolo_tagger_gui
    except ImportError:
        yolo_tagger_gui = None

try:
    from UI import trainer_gui
except ImportError:
    trainer_gui = None


# ==========================================
# 3. MAIN APPLICATION LAYOUT
# ==========================================
def run_gui():
    # App Header
    st.title("🚀 Unified YOLO & Vision Suite")
    st.caption("A complete pipeline for Auto-Tagging, Captioning, Dataset Preparation, and YOLO Training.")
    st.markdown("---")

    # Create Main Navigation Tabs
    tab_settings, tab_tagger, tab_captions, tab_datasets, tab_trainer = st.tabs([
        "⚙️ Settings",
        "🏷️ YOLO Auto-Tagger",
        "👁️ Captions & Vision",
        "📂 Dataset Prep",
        "🚀 YOLO Trainer"
    ])

    # ---------------------------------------------------------
    # TAB 1: SETTINGS
    # ---------------------------------------------------------
    with tab_settings:
        if settings_gui and hasattr(settings_gui, 'render_settings_tab'):
            settings_gui.render_settings_tab()
        else:
            st.error("Settings GUI module not found or missing `render_settings_tab` function.")

    # ---------------------------------------------------------
    # TAB 2: YOLO AUTO-TAGGER
    # ---------------------------------------------------------
    with tab_tagger:
        if yolo_tagger_gui:
            # Safe execution depending on how the function was named in the script
            if hasattr(yolo_tagger_gui, 'tagger_gui'):
                yolo_tagger_gui.tagger_gui()
            elif hasattr(yolo_tagger_gui, 'yolo_tagger_tab'):
                yolo_tagger_gui.yolo_tagger_tab()
            elif hasattr(yolo_tagger_gui, 'main'):
                yolo_tagger_gui.main()
            else:
                st.warning("YOLO Tagger GUI loaded, but no main render function found.")
        else:
            st.error("YOLO Tagger GUI module (`yolo_tagger_gui.py`) not found in the `UI/` folder.")

    # ---------------------------------------------------------
    # TAB 3: CAPTIONS & VISION (BLIP, CLIP, Florence, WD14)
    # ---------------------------------------------------------
    with tab_captions:
        if captions_gui and hasattr(captions_gui, 'captions_gui'):
            captions_gui.captions_gui()
        else:
            st.error("Captions GUI module not found or missing `captions_gui` function.")

    # ---------------------------------------------------------
    # TAB 4: DATASET PREPARATION
    # ---------------------------------------------------------
    with tab_datasets:
        if datasets_gui:
            if hasattr(datasets_gui, 'dataset_preparation_gui'):
                datasets_gui.dataset_preparation_gui()
            elif hasattr(datasets_gui, 'dataset_preparation'):
                datasets_gui.dataset_preparation()
            else:
                st.warning("Dataset GUI loaded, but no main render function found.")
        else:
            st.error("Dataset GUI module (`datasets_gui.py`) not found in the `UI/` folder.")

    # ---------------------------------------------------------
    # TAB 5: YOLO TRAINER
    # ---------------------------------------------------------
    with tab_trainer:
        if trainer_gui:
            # ✅ FIX: Check for 'render_yolo_tab' instead of 'yolo_trainer_gui'
            if hasattr(trainer_gui, 'render_yolo_tab'):
                trainer_gui.render_yolo_tab()            
            else:
                st.info("Trainer GUI module found, but awaiting final implementation.")
        else:
            st.info("🚧 **YOLO Trainer is under construction.**")
            st.write("The backend (`yolo_trainer_script.py`) and GUI (`trainer_gui.py`) are the final pieces of the pipeline.")

# ==========================================
# 4. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    run_gui()
import streamlit as st

import UI.wiki.ultralytics as ultralytics
import UI.wiki.textual_inversion as TI
import UI.wiki.hypernetwork as hypernetwork


def render_wiki_tab():
    st.header("🔍 Wiki")

def render_trainer_tab():
    st.header("🏋️ Model Trainer Workspace")

    tab_yolo, tab_lora, tab_TI, tab_hypernetwork = st.tabs([
        "🎨 YOLO Trainer",
        "🎨 LoRA Trainer",
        "🧠 Textual Inversion & Hypernetworks",
        "🏋️ Train",
    ])

    with tab_yolo:
        try:
            ultralytics.render_ultralytics_tab()
        except Exception as e:
            st.error(f"Failed to load YOLO Trainer: {e}")

    with tab_TI:
        try:
            TI.render_textual_inversion_tab()
        except Exception as e:
            st.error(f"Failed to load Textual Inversion Trainer: {e}")

    with tab_hypernetwork:
        try:
            hypernetwork.render_hypernetwork_tab()
        except Exception as e:
            st.error(f"Failed to load Hypernetwork Trainer: {e}")
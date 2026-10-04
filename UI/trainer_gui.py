import streamlit as st
from transformers import trainer
import UI.trainer.yolo_trainer_gui as yolo
import UI.trainer.lora_tab as lora
import UI.trainer.textual_inversion as TI
import UI.trainer.hypernetwork_tab as hypernetwork

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
            yolo.render_yolo_tab()
        except Exception as e:
            st.error(f"Failed to load YOLO Trainer: {e}")

    with tab_lora:
        try:
            lora.render_lora_trainer_ui()
        except Exception as e:
            st.error(f"Failed to load LoRA Trainer: {e}")

    with tab_TI:
        try:
            TI.render_lora_trainer_ui()
        except Exception as e:
            st.error(f"Failed to load LoRA Trainer: {e}")

    with tab_hypernetwork:
        try:
            hypernetwork.render_lora_trainer_ui()
        except Exception as e:
            st.error(f"Failed to load LoRA Trainer: {e}")
import streamlit as st


def render_trainer_tab():
    st.header("🏋️ Model Trainer Workspace")

    tab_bbox, tab_segm, tab_lora, tab_embed, tab_train = st.tabs([
        "🎯 BBox Trainer",
        "✂️ Segm Trainer",
        "🎨 LoRA Trainer",
        "🧠 Textual Inversion & Hypernetworks",
        "🏋️ Train"
    ])

    with tab_bbox:
        try:
            from UI.trainer import bbox_trainer_tab
            bbox_trainer_tab.render_bbox_trainer_ui()
        except Exception as e:
            st.error(f"Failed to load BBox Trainer: {e}")

    with tab_segm:
        try:
            from UI.trainer import segm_trainer_tab
            segm_trainer_tab.render_segm_trainer_ui()
        except Exception as e:
            st.error(f"Failed to load Segm Trainer: {e}")

    with tab_lora:
        try:
            from UI.trainer import lora_tab
            lora_tab.render_lora_trainer_ui()
        except Exception as e:
            st.error(f"Failed to load LoRA Trainer: {e}")

    with tab_embed:
        col1, col2 = st.columns(2)
        with col1:
            try:
                from UI.trainer import textual_inversion
                textual_inversion.render_textual_inversion_ui()
            except Exception as e:
                st.error(f"Failed to load Textual Inversion: {e}")
        with col2:
            try:
                from UI.trainer import hypernetwork_tab
                hypernetwork_tab.render_hypernetwork_ui()
            except Exception as e:
                st.error(f"Failed to load Hypernetwork: {e}")

    with tab_train:
        try:
            from UI.trainer import yolo_complete_tab
            yolo_complete_tab.trainer_tab()
        except Exception as e:
            st.error(f"Failed to load YOLO Trainer: {e}")
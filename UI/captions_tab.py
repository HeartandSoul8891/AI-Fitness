import streamlit as st

def captions_tab():
    st.subheader("📝 Image Tagging & Text Captions")

    sub_wd14, sub_clip_blip, sub_manual = st.tabs([
        "🏷️ WD14 Tagger",
        " 💬 BLIP Captioneer |🔍 CLIP Interrogator",
        "✍️ Manual Caption Editor"
    ])

    with sub_wd14:
        try:
            from UI.captions import wd14
            wd14.wd14_tab()
        except Exception as e:
            st.info("WD14 module loading...")

    with sub_clip_blip:
        try:
            from UI.captions import blip_clip_tab
            blip_clip_tab.blip_clip_tab()
        except Exception as e:
            st.info("BLIP & CLIP module loading...")

    with sub_manual:
        try:
            from UI.captions import florence2
            florence2.florence2_tab()
        except Exception as e:
            st.info("Manual caption editor loading...")
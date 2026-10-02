import streamlit as st

def render_tagger_tab():
    st.header("🏷️ Dataset Tagger & Annotator")

    (
        tab_auto_bbox,
        tab_auto_segm,
        tab_cls,
        tab_sem,
        tab_obb,
        tab_pose,
        tab_dept,
    ) = st.tabs([
        "🎯 Auto BBox",
        "🎯 Auto Segm",
        "🏷️ Classification",
        "✂️ Semantic / Segm",
        "🔄 OBB Tagger",
        "🦴 Pose Tagger",
        "📏 Depth Tagger",
    ])

    with tab_auto_bbox:
        try:
            from UI.tagger import bbox_tab
            bbox_tab.bbox_tab()
        except Exception as e:
            st.info("Auto BBox tagger module loading...")

    with tab_auto_segm:
        try:
            from UI.tagger import segm_tab
            segm_tab.segm_tab()
        except Exception as e:
            st.info("Auto Segm tagger module loading...")

    with tab_cls:
        try:
            from UI.tagger import cls_tab
            cls_tab.cls_tab()
        except Exception as e:
            st.info("Classification tagger module loading...")

    with tab_sem:
        try:
            from UI.tagger import sem_tab
            sem_tab.sem_tab()
        except Exception as e:
            st.info("Semantic tagger module loading...")

    with tab_obb:
        try:
            from UI.tagger import obb_tab
            obb_tab.obb_tab()
        except Exception as e:
            st.info("OBB tagger module loading...")

    with tab_pose:
        try:
            from UI.tagger import pose_tab
            pose_tab.pose_tab()
        except Exception as e:
            st.info("Pose tagger module loading...")

    with tab_dept:
        try:
            from UI.tagger import dept_tab
            dept_tab.dept_tab()
        except Exception as e:
            st.info("Depth tagger module loading...")
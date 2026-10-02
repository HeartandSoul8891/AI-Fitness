import streamlit as st


def render_tagger_tab():
    st.header("🏷️ Dataset Tagger & Annotator")

    (
        tab_annotator,
        tab_auto_bbox,
        tab_auto_segm,
        tab_zeroshot,
        tab_cls,
        tab_sem,
        tab_obb,
        tab_pose,
        tab_dept,
    ) = st.tabs([
        "✏️ Manual Annotator",
        "🎯 Auto BBox",
        "🎯 Auto Segm",
        "🌐 Zero-Shot Tagger",
        "🏷️ Classification",
        "✂️ Semantic / Segm",
        "🔄 OBB Tagger",
        "🦴 Pose Tagger",
        "📏 Depth Tagger",
    ])

    with tab_annotator:
        render_manual_annotator_inline()

    with tab_auto_bbox:
        try:
            from UI.tagger import bbox_tab
            bbox_tab.bbox_tagger_tab()
        except Exception:
            st.info("Auto BBox tagger module loading...")

    with tab_auto_segm:
        try:
            from UI.tagger import segm_tab
            segm_tab.segm_tagger_tab()
        except Exception:
            st.info("Auto Segm tagger module loading...")

    with tab_zeroshot:
        try:
            from UI.tagger import zeroshot_tab
            zeroshot_tab.zeroshot_tagger_tab()
        except Exception:
            st.info("Auto Zero-Shot tagger module loading...")

    with tab_cls:
        try:
            from UI.tagger import cls_tab
            cls_tab.render_cls_inline()
        except Exception:
            st.info("Classification tagger module loading...")         
                
    with tab_sem:
        try:
            from UI.tagger import sem_tab
            sem_tab.render_sem_inline()
        except Exception:
            st.info("Semantic / Segmentation tagger module loading...")

    with tab_obb:
        try:
            from UI.tagger import obb_tab
            obb_tab.render_obb_inline()
        except Exception:
            st.info("OBB tagger module loading...")

    with tab_pose:
        try:
            from UI.tagger import pose_tab
            pose_tab.render_pose_inline()
        except Exception:
            st.info("Pose tagger module loading...")

    with tab_dept:
        try:
            from UI.tagger import depth_tab
            depth_tab.render_depth_inline()
        except Exception:
            st.info("Depth tagger module loading...")


def render_manual_annotator_inline():
    st.subheader("Image Annotator")

    # Inline file paths and settings replaces st.sidebar
    with st.expander("📁 File Paths & Config", expanded=True):
        col_src, col_man = st.columns(2)
        source_dir = col_src.text_input("Source Directory (Raw Images)", value="./raw_images", key="man_src_dir")
        manifest_path = col_man.text_input("Manifest JSON Path", value="./tags_manifest.json", key="man_manifest_path")

    with st.expander("🏷️ Class Management", expanded=False):
        col_add, col_del = st.columns(2)
        with col_add:
            new_class = st.text_input("Add Class Name", key="man_new_class")
            if st.button("➕ Add Class", key="man_add_btn", use_container_width=True) and new_class:
                st.success(f"Added class: `{new_class}`")

        with col_del:
            cls_to_remove = st.selectbox("Remove Class", options=["", "class_a", "class_b"], key="man_rem_class")
            if st.button("🗑️ Delete Class", key="man_del_btn", use_container_width=True) and cls_to_remove:
                st.warning(f"Deleted class: `{cls_to_remove}`")

    # Main workspace controls
    dataset_folder = st.text_input("Select Dataset Folder", value=".", key="man_folder")
    tags_str = st.text_input("Specify tags (separated by ';', max 6)", key="man_tags")

    if st.button("Select All Images & Start Annotation", type="primary"):
        st.info("Annotation session initialized.")
    else:
        st.info("Define your tags above and click 'Select All Images & Start Annotation' to begin.")
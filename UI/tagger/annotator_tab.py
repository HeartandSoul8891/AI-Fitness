import os
import streamlit as st
from streamlit_image_annotation import detection

from scripts.settings.settings_script import load_settings
from scripts.tagger.annotator_script import (
    get_dataset_folders,
    get_image_files,
    prepare_display_image,
    format_annotation_shapes,
    save_tag_json,
)

def annotator():
    st.title("Image Annotator")

    # Settings and dataset root
    settings = load_settings()
    base_dataset_dir = settings.get("custom_datasets_folder") or "datasets"

    if not os.path.exists(base_dataset_dir):
        st.error(f"Dataset root missing: `{base_dataset_dir}`. Check Settings.")
        return

    # Folder selection
    selected_subfolder = st.selectbox("Select Dataset Folder", get_dataset_folders(base_dataset_dir))
    target_folder = base_dataset_dir if selected_subfolder == "." else os.path.join(base_dataset_dir, selected_subfolder)

    # Tag input starts completely blank for user to define
    raw_tags = st.text_input("Specify tags (separated by ';', max 6)", value="")
    tags = [t.strip() for t in raw_tags.split(";") if t.strip()][:6]

    # Session state init
    if "img_index" not in st.session_state:
        st.session_state.img_index = 0

    if "active_session" not in st.session_state:
        st.session_state.active_session = False

    # Start session button
    if st.button("Select All Images & Start Annotation"):
        if not tags:
            st.warning("Please specify at least one tag before starting.")
            return
        st.session_state.image_queue = get_image_files(target_folder)
        st.session_state.img_index = 0
        st.session_state.active_session = True
        st.rerun()

    if not st.session_state.active_session or not st.session_state.get("image_queue"):
        st.info("Define your tags above and click 'Select All Images & Start Annotation' to begin.")
        return

    queue = st.session_state.image_queue
    idx = st.session_state.img_index

    # End of queue check
    if idx >= len(queue):
        st.success("All images in this folder processed!")
        if st.button("Restart Queue"):
            st.session_state.img_index = 0
            st.rerun()
        return

    current_image_name = queue[idx]
    image_path = os.path.join(target_folder, current_image_name)

    display_img, scale_factor, orig_w, orig_h = prepare_display_image(image_path, max_width=768)

    st.markdown(f"**Image {idx + 1} of {len(queue)}:** `{current_image_name}` ({orig_w}x{orig_h}px)")

    # Controls positioned above annotation tool
    col_save, col_skip = st.columns(2)
    save_clicked = col_save.button("Save Tag & Load Next Image", type="primary")
    skip_clicked = col_skip.button("Skip Image")

    if skip_clicked:
        st.session_state.img_index += 1
        st.rerun()

    st.info("💡 Select your desired tag from the **Class** dropdown on the right side of the canvas before drawing each bounding box.")

    # Native image annotation component with stable tag list order
    new_bboxes = detection(
        image_path=image_path,
        label_list=tags,
        bboxes=[],
        labels=[],
        key=f"anno_{current_image_name}_{idx}"
    )

    # Save logic
    if save_clicked:
        if new_bboxes:
            shapes = format_annotation_shapes(
                new_bboxes, 
                orig_w, 
                orig_h, 
                display_img.width, 
                display_img.height
            )
            save_tag_json(image_path, shapes, orig_w, orig_h)
            st.session_state.img_index += 1
            st.rerun()
        else:
            st.warning("Please draw at least one bounding box before saving.")
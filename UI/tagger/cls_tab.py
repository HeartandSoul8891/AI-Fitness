import streamlit as st
from PIL import Image
from pathlib import Path
from scripts.tagger.cls_script import load_manifest, save_manifest, scan_images

st.set_page_config(
    page_title="Dataset Tagger",
    page_icon="🏷️",
    layout="wide"
)

st.title("🏷️ Dataset Tagger")

# ==========================================
# CONFIG & CLASS MANAGEMENT (MAIN PAGE)
# ==========================================
with st.expander("⚙️ File Paths & Class Management", expanded=False):
    cfg_col1, cfg_col2 = st.columns(2)
    with cfg_col1:
        st.subheader("📁 File Paths")
        source_dir = st.text_input("Source Directory (Raw Images)", value="./raw_images")
        manifest_path = st.text_input("Manifest JSON Path", value="./tags_manifest.json")

    with cfg_col2:
        st.subheader("🏷️ Class Management")
        new_class = st.text_input("Add Class Name")
        if st.button("➕ Add Class", use_container_width=True) and new_class:
            cls_clean = new_class.strip()
            if cls_clean and cls_clean not in st.session_state.classes:
                st.session_state.classes.append(cls_clean)
                save_manifest(manifest_path, st.session_state.classes, st.session_state.annotations)
                st.success(f"Added: `{cls_clean}`")
                st.rerun()

        if st.session_state.get("classes"):
            cls_to_remove = st.selectbox("Remove Class", options=[""] + st.session_state.classes)
            if st.button("🗑️ Delete Class", use_container_width=True) and cls_to_remove:
                st.session_state.classes.remove(cls_to_remove)
                save_manifest(manifest_path, st.session_state.classes, st.session_state.annotations)
                st.warning(f"Deleted class: `{cls_to_remove}`")
                st.rerun()

# Load existing manifest on initial run or path change
if "manifest_path" not in st.session_state or st.session_state.manifest_path != manifest_path:
    manifest = load_manifest(manifest_path)
    st.session_state.manifest_path = manifest_path
    st.session_state.classes = manifest.get("classes", ["class_a", "class_b"])
    st.session_state.annotations = manifest.get("annotations", {})
    st.session_state.image_index = 0

# ==========================================
# MAIN INTERFACE
# ==========================================
image_files = scan_images(source_dir)

if not image_files:
    st.warning(f"No valid images found in `{source_dir}`. Check directory path.")
else:
    total_imgs = len(image_files)
    
    if st.session_state.image_index >= total_imgs:
        st.session_state.image_index = total_imgs - 1
    elif st.session_state.image_index < 0:
        st.session_state.image_index = 0

    idx = st.session_state.image_index
    current_file = image_files[idx]
    filename = current_file.name

    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("Total Images", total_imgs)
    col_stat2.metric("Tagged Images", len(st.session_state.annotations))
    col_stat3.metric("Progress", f"{idx + 1} / {total_imgs}")

    st.progress((idx + 1) / total_imgs)

    img_col, tag_col = st.columns([2, 1])

    with img_col:
        st.subheader(f"🖼️ `{filename}`")
        try:
            image = Image.open(current_file)
            st.image(image, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")

    with tag_col:
        st.subheader("Assign Tag")
        current_tag = st.session_state.annotations.get(filename, None)

        if current_tag:
            st.success(f"Assigned Tag: **{current_tag}**")
        else:
            st.info("Status: **Unassigned**")

        st.write("Click class to assign tag and advance:")

        for i, cls_name in enumerate(st.session_state.classes):
            shortcut_label = f"[{i+1}] " if i < 9 else ""
            
            if st.button(f"{shortcut_label}🏷️ {cls_name}", key=f"cls_btn_{cls_name}", use_container_width=True):
                st.session_state.annotations[filename] = cls_name
                save_manifest(manifest_path, st.session_state.classes, st.session_state.annotations)
                
                if st.session_state.image_index < total_imgs - 1:
                    st.session_state.image_index += 1
                st.rerun()

        st.markdown("---")
        
        nav_prev, nav_next = st.columns(2)
        with nav_prev:
            if st.button("⬅️ Previous", use_container_width=True):
                if st.session_state.image_index > 0:
                    st.session_state.image_index -= 1
                    st.rerun()
                    
        with nav_next:
            if st.button("Next ➡️", use_container_width=True):
                if st.session_state.image_index < total_imgs - 1:
                    st.session_state.image_index += 1
                    st.rerun()

        target_idx = st.number_input("Jump to Image #", min_value=1, max_value=total_imgs, value=idx + 1)
        if target_idx - 1 != idx:
            st.session_state.image_index = target_idx - 1
            st.rerun()
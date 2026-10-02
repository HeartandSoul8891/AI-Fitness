import streamlit as st
from pathlib import Path
from PIL import Image, ImageDraw
from scripts.settings.settings_script import load_settings

def pose_tab():
    # Heavy tagger import happens ONLY when this tab is actually opened.
    from scripts.tagger.pose_script import (
        load_pose_manifest,
        save_pose_manifest,
        scan_pose_dataset,
        DEFAULT_KEYPOINT_NAMES,
        DEFAULT_SKELETON_LINKS,
    )

    saved_settings = load_settings()
    global_datasets_dir = st.session_state.get(
    "settings_datasets_folder", 
    saved_settings.get("datasets_folder", "./datasets")
    )

    st.title("🦴 YOLO Pose Estimation Tagger")

    # 1. Fetch Global Settings from session state or settings.json
    saved_settings = load_settings()
    global_datasets_dir = st.session_state.get(
        "settings_datasets_folder", 
        saved_settings.get("datasets_folder", "./raw_images")
    )

    # 2. Synchronize tab-specific state with global dataset setting
    if "pose_source_dir" not in st.session_state or st.session_state.pose_source_dir != global_datasets_dir:
        st.session_state.pose_source_dir = global_datasets_dir

    # Configuration Header UI
    st.markdown("### ⚙️ Dataset Configuration & Manifest")
    cfg_col1, cfg_col2 = st.columns(2)

    with cfg_col1:
        # Binds directly to st.session_state.pose_source_dir
        source_dir = st.text_input(
            "Dataset Directory Path", 
            key="pose_source_dir"
        )

    project_root = Path(saved_settings.get("app_root", Path(__file__).resolve().parents[1]))
    manifest_path = project_root / "pose_manifest.json"

    with cfg_col2:
        st.text_input("Manifest JSON Path", value=str(manifest_path), disabled=True)

    datasets_folder = Path(source_dir).expanduser()

    if not datasets_folder.exists():
        st.error(f"Dataset folder does not exist:\n\n`{datasets_folder}`\n\nPlease check the path in Settings or update above.")
        return

    if "pose_manifest_loaded" not in st.session_state:
        manifest = load_pose_manifest(str(manifest_path))

        st.session_state.pose_manifest_loaded = True
        st.session_state.pose_classes = manifest.get("classes", ["person"])
        st.session_state.pose_keypoint_names = manifest.get("keypoint_names", DEFAULT_KEYPOINT_NAMES)
        st.session_state.pose_skeleton = manifest.get("skeleton", DEFAULT_SKELETON_LINKS)
        st.session_state.pose_annotations = manifest.get("annotations", {})
        st.session_state.pose_image_index = 0

    classes = st.session_state.pose_classes
    keypoint_names = st.session_state.pose_keypoint_names
    skeleton = st.session_state.pose_skeleton
    annotations = st.session_state.pose_annotations

    image_files = scan_pose_dataset(str(datasets_folder))

    if not image_files:
        st.warning(f"No valid images were found in:\n\n`{datasets_folder}`")
        return

    total_imgs = len(image_files)

    if st.session_state.pose_image_index >= total_imgs:
        st.session_state.pose_image_index = total_imgs - 1

    if st.session_state.pose_image_index < 0:
        st.session_state.pose_image_index = 0

    idx = st.session_state.pose_image_index
    current_file = image_files[idx]
    filename = current_file.name

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Images", total_imgs)
    col2.metric(
        "Tagged Images",
        len([k for k, v in annotations.items() if v.get("poses")]),
    )
    col3.metric("Progress", f"{idx + 1} / {total_imgs}")

    st.progress((idx + 1) / total_imgs)

    img_col, tag_col = st.columns([2, 1])

    with img_col:
        st.subheader(f"🖼️ `{filename}`")

        try:
            image = Image.open(current_file).convert("RGB")
            w, h = image.size

            annotated_img = image.copy()
            draw = ImageDraw.Draw(annotated_img)

            existing_data = annotations.get(filename, {})
            poses = existing_data.get("poses", [])

            for pose in poses:
                keypoints = pose["keypoints"]

                for link in skeleton:
                    idx1, idx2 = link

                    if idx1 >= len(keypoints) or idx2 >= len(keypoints):
                        continue

                    p1 = keypoints[idx1]
                    p2 = keypoints[idx2]

                    if p1["v"] > 0 and p2["v"] > 0:
                        draw.line(
                            [(p1["x"], p1["y"]), (p2["x"], p2["y"])],
                            fill="cyan",
                            width=2,
                        )

                for kp in keypoints:
                    px = kp["x"]
                    py = kp["y"]
                    visibility = kp["v"]

                    if visibility > 0:
                        radius = 4
                        point_color = "lime" if visibility == 2 else "orange"

                        draw.ellipse(
                            (px - radius, py - radius, px + radius, py + radius),
                            fill=point_color,
                            outline="white",
                        )

            st.image(annotated_img, use_container_width=True)

            st.caption(
                f"Resolution: {w}x{h} px | Green = Visible | Orange = Occluded"
            )

        except Exception as exc:
            st.error(f"Error rendering image: {exc}")
            return

    with tag_col:
        st.subheader("Add Keypoint Pose Instance")

        selected_cls = st.selectbox(
            "Select Class",
            options=classes,
            key="pose_selected_class",
        )

        selected_kp = st.selectbox(
            "Select Keypoint Joint",
            options=keypoint_names,
            key="pose_selected_keypoint",
        )

        kp_x = st.number_input(
            "Joint X (px)",
            min_value=0,
            max_value=w if 'w' in locals() else 1920,
            value=w // 2 if 'w' in locals() else 0,
        )

        kp_y = st.number_input(
            "Joint Y (px)",
            min_value=0,
            max_value=h if 'h' in locals() else 1080,
            value=h // 2 if 'h' in locals() else 0,
        )

        kp_v = st.radio(
            "Visibility Flag",
            options=[2, 1, 0],
            format_func=lambda value: {
                2: "2 - Visible",
                1: "1 - Occluded",
                0: "0 - Absent",
            }[value],
            horizontal=True,
        )

        if "pose_builder" not in st.session_state:
            st.session_state.pose_builder = {
                name: {"x": 0, "y": 0, "v": 0} for name in keypoint_names
            }

        if st.button("📌 Set Joint Position", use_container_width=True):
            st.session_state.pose_builder[selected_kp] = {
                "x": int(kp_x),
                "y": int(kp_y),
                "v": int(kp_v),
            }

            st.success(f"Set `{selected_kp}` to ({kp_x}, {kp_y}) [v={kp_v}]")

        st.markdown("---")

        if st.button(
            "💾 Save Pose Instance to Image",
            type="primary",
            use_container_width=True,
        ):
            if filename not in annotations:
                annotations[filename] = {"poses": []}

            formatted_keypoints = [
                st.session_state.pose_builder[name] for name in keypoint_names
            ]

            annotations[filename]["poses"].append(
                {
                    "class": selected_cls,
                    "keypoints": formatted_keypoints,
                }
            )

            st.session_state.pose_builder = {
                name: {"x": 0, "y": 0, "v": 0} for name in keypoint_names
            }

            save_pose_manifest(
                str(manifest_path),
                classes,
                keypoint_names,
                skeleton,
                annotations,
            )

            st.success("Pose instance saved!")
            st.rerun()

        current_poses = annotations.get(filename, {}).get("poses", [])

        if current_poses:
            st.markdown("---")
            st.write("**Tagged Poses on Image:**")

            for number, pose in enumerate(current_poses, start=1):
                active_points = sum(
                    1 for point in pose["keypoints"] if point["v"] > 0
                )

                st.text(
                    f"Pose #{number}: {pose['class']} ({active_points}/{len(pose['keypoints'])} joints)"
                )

            if st.button("🗑️ Clear Poses for Image", use_container_width=True):
                annotations[filename]["poses"] = []

                save_pose_manifest(
                    str(manifest_path),
                    classes,
                    keypoint_names,
                    skeleton,
                    annotations,
                )

                st.rerun()

        st.markdown("---")

        nav_prev, nav_next = st.columns(2)

        with nav_prev:
            if st.button("⬅️ Previous", use_container_width=True):
                if st.session_state.pose_image_index > 0:
                    st.session_state.pose_image_index -= 1
                    st.rerun()

        with nav_next:
            if st.button("Next ➡️", use_container_width=True):
                if st.session_state.pose_image_index < total_imgs - 1:
                    st.session_state.pose_image_index += 1
                    st.rerun()



if __name__ == "__main__":
    st.set_page_config(page_title="YOLO Pose Tagger", layout="wide")
    pose_tab()
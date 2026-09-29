import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Import settings and backend test runner
from scripts.settings_script import load_settings
from scripts.ultralytics.bbox_test_script import run_stress_test_backend


def render_bbox_test_ui():
    st.title("🎯 YOLO Stress & Threshold Tester")
    st.write(
        "Run two-pass batch evaluations across dataset folders to verify confidence distributions and edge cases."
    )

    # --- Load Settings ---
    settings = load_settings()
    default_datasets_dir = settings.get("datasets_folder", "training")
    default_models_dir = settings.get("ultralytics_bbox_folder", "ultralytics/bbox")

    # --- Sidebar / Controls ---
    st.sidebar.header("📁 Directory & Model Settings")

    # Image Directory Selection
    test_image_dir = st.sidebar.text_input(
        "Test Images Directory",
        value=default_datasets_dir,
        help="Path to folder containing test images (reads recursively).",
    )

    # Model Checkpoint Path
    model_path = st.sidebar.text_input(
        "YOLO Model Weight Path (.pt)",
        value=os.path.join(default_models_dir, "best.pt"),
        help="Path to trained YOLO PyTorch weights.",
    )

    st.sidebar.header("⚙️ Two-Pass Confidence Settings")

    # Slider Controls
    pass1_conf = st.sidebar.slider(
        "Pass 1 Confidence Threshold (High Precision)",
        min_value=0.50,
        max_value=0.95,
        value=0.75,
        step=0.01,
        help="Initial pass to lock in high-confidence detections.",
    )

    pass2_conf = st.sidebar.slider(
        "Pass 2 Confidence Threshold (Recovery / Edge Cases)",
        min_value=0.05,
        max_value=0.49,
        value=0.17,
        step=0.01,
        help="Fallback threshold run only on images missed during Pass 1.",
    )

    # --- Run Stress Test Trigger ---
    if st.button("🚀 Run Batch Stress Test", type="primary"):
        if not os.path.exists(test_image_dir):
            st.error(f"Directory not found: `{test_image_dir}`")
            return
        if not os.path.exists(model_path):
            st.error(f"Model file not found: `{model_path}`")
            return

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(pct, text):
            progress_bar.progress(pct)
            status_text.text(text)

        with st.spinner("Running inference across images..."):
            summary, df_results = run_stress_test_backend(
                model_path=model_path,
                image_dir=test_image_dir,
                pass1_conf=pass1_conf,
                pass2_conf=pass2_conf,
                progress_callback=update_progress,
            )

        progress_bar.empty()
        status_text.empty()

        if summary["total_images"] == 0:
            st.warning("No images found in the specified directory.")
            return

        # Store in session state for dynamic filtering without re-running
        st.session_state["stress_test_summary"] = summary
        st.session_state["stress_test_df"] = df_results
        st.success("Stress Test Execution Finished!")

    # --- Results Display ---
    if "stress_test_summary" in st.session_state:
        summary = st.session_state["stress_test_summary"]
        df_results = st.session_state["stress_test_df"]

        st.markdown("---")
        st.subheader("📊 Execution Summary")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Images Evaluated", summary["total_images"])
        m2.metric(f"Pass 1 Hits (≥ {pass1_conf})", summary["pass1_detected"])
        m3.metric(f"Pass 2 Recovered (≥ {pass2_conf})", summary["pass2_recovered"])
        m4.metric("Truly Undetected", summary["still_missed"])

        if not df_results.empty:
            st.subheader("📈 Confidence Score Distribution")

            # Chart Generation
            fig, ax = plt.subplots(figsize=(10, 4.5))
            sns.histplot(
                data=df_results,
                x="confidence",
                hue="pass",
                bins=30,
                kde=True,
                palette={"Pass 1 (High)": "#1f77b4", "Pass 2 (Recovery)": "#ff7f0e"},
                ax=ax,
            )
            ax.axvline(
                pass1_conf,
                color="blue",
                linestyle="--",
                label=f"Pass 1 Threshold ({pass1_conf})",
            )
            ax.axvline(
                pass2_conf,
                color="orange",
                linestyle="--",
                label=f"Pass 2 Threshold ({pass2_conf})",
            )
            ax.set_title("Detection Confidence Spread Across Test Set")
            ax.set_xlabel("Confidence Score")
            ax.set_ylabel("Detection Count")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend()

            st.pyplot(fig)
            plt.close(fig)

            # Raw Data Inspection
            with st.expander("🔍 View Raw Detection Results"):
                st.dataframe(df_results, use_container_width=True)


if __name__ == "__main__":
    render_bbox_test_ui()
import streamlit as st
import os
import sys

# Ensure project root is in system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="AI-Fitness",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🏋️ AI-Fitness")

    tab_datasets, tab_tagger,tab_captions, tab_trainer, tab_benchmark, tab_settings = st.tabs([
        "📁 Datasets", 
        "🏷️ Tagger",
        "📝 Captions",
        "🏋️ Trainer", 
        "📊 Benchmark",
        "⚙️ Settings"
    ])

    with tab_datasets:
        try:
            from UI.datasets import datasets_tab
            datasets_tab.dataset_preparation()
        except ImportError:
            from UI import datasets_tab
            datasets_tab.dataset_preparation()

    with tab_tagger:
        try:
            from UI import tagger_tab
            tagger_tab.render_tagger_tab()
        except Exception as e:
            st.error(f"Failed to load Tagger module: {e}")

    with tab_captions:
        try:
            from UI import captions_tab
            captions_tab.captions_tab()
        except Exception as e:
            st.error(f"Failed to load Captions module: {e}")

    with tab_trainer:
        try:
            from UI import trainer_tab
            trainer_tab.render_trainer_tab()
        except Exception as e:
            st.error(f"Failed to load Trainer module: {e}")

    with tab_benchmark:
        try:
            from UI.benchmark import benchmark_tab
            benchmark_tab.render_benchmark_tab()
        except ImportError:
            st.info("Benchmark module loading...")

    with tab_settings:
        try:
            from UI import settings_tab
            settings_tab.render_settings_tab()
        except Exception as e:
            st.error(f"Failed to load Settings module: {e}")

if __name__ == "__main__":
    main()
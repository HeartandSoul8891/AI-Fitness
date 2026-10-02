import streamlit as st
import os

def render_datasets_tab():
    st.header("📁 Dataset Manager & History Log")
    
    tab_browser, tab_history = st.tabs(["📂 Dataset Viewer", "📊 Training Track Log"])

    with tab_browser:
        st.subheader("Browse Active Datasets")
        datasets_dir = st.text_input("Datasets Base Directory Path", value="./datasets")
        if os.path.exists(datasets_dir):
            folders = [f for f in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, f))]
            st.selectbox("Select Dataset Folder", folders if folders else ["No datasets found"])
        else:
            st.warning("Specified directory does not exist.")

    with tab_history:
        st.subheader("Dataset Model Usage History")
        st.info("Track models trained using specific dataset versions.")
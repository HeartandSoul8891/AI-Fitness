import os
import streamlit as st
from scripts.settings_script import load_settings, save_settings

def settings():
    st.title("Settings")
    st.write("Configure your settings here.")

    # Load existing settings
    current_settings = load_settings()

    # Define default app root folders
    default_model_dir = "models"
    default_work_dir = "datasets"
    default_datasets_dir = "datasets"
    default_training_dir = "runs/train"

    # Model folder selection
    saved_model_folder = current_settings.get("custom_model_folder", default_model_dir)
    use_default_model = st.checkbox("Use Default Models Folder (./models)", value=(saved_model_folder == default_model_dir or not saved_model_folder))
    
    if use_default_model:
        custom_model_folder = default_model_dir
        st.text_input("Custom Ultralytics Model Folder", value=default_model_dir, disabled=True)
    else:
        custom_model_folder = st.text_input("Custom Ultralytics Model Folder", value=saved_model_folder if saved_model_folder != default_model_dir else "")

    # Work folder selection
    saved_work_folder = current_settings.get("custom_work_folder", default_work_dir)
    use_default_work = st.checkbox("Use Default Work Folder (./datasets)", value=(saved_work_folder == default_work_dir or not saved_work_folder))
    
    if use_default_work:
        custom_work_folder = default_work_dir
        st.text_input("Custom Work Folder", value=default_work_dir, disabled=True)
    else:
        custom_work_folder = st.text_input("Custom Work Folder", value=saved_work_folder if saved_work_folder != default_work_dir else "")

    # Datasets folder selection
    saved_datasets_folder = current_settings.get("custom_datasets_folder", default_datasets_dir)
    use_default_datasets = st.checkbox("Use Default Datasets Folder (./datasets)", value=(saved_datasets_folder == default_datasets_dir or not saved_datasets_folder))
    
    if use_default_datasets:
        custom_datasets_folder = default_datasets_dir
        st.text_input("Custom Datasets Folder", value=default_datasets_dir, disabled=True)
    else:
        custom_datasets_folder = st.text_input("Custom Datasets Folder", value=saved_datasets_folder if saved_datasets_folder != default_datasets_dir else "")

    # Training output folder selection
    saved_training_folder = current_settings.get("custom_training_folder", default_training_dir)
    use_default_training = st.checkbox("Use Default Training Output Folder (./runs/train)", value=(saved_training_folder == default_training_dir or not saved_training_folder))
    
    if use_default_training:
        custom_training_folder = default_training_dir
        st.text_input("Custom Training Output Folder", value=default_training_dir, disabled=True)
    else:
        custom_training_folder = st.text_input("Custom Training Output Folder", value=saved_training_folder if saved_training_folder != default_training_dir else "")

    # Save button
    if st.button("Save Settings"):
        settings_data = {
            "custom_model_folder": custom_model_folder,
            "custom_work_folder": custom_work_folder,
            "custom_datasets_folder": custom_datasets_folder,
            "custom_training_folder": custom_training_folder
        }
        save_settings(settings_data)
        st.success("Settings saved successfully!")
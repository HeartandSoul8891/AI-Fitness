import streamlit as st
import UI.datasets_gui as datasets
import UI.captions_gui as captions
import UI.settings_gui as settings
import UI.trainer_gui as trainer
import UI.wiki_gui as wiki

st.set_page_config(page_title="Pix Manager", page_icon="🖼️", layout="wide")

st.title("AI Fitness v3.0")

if "selected_folder" not in st.session_state:
    st.session_state["selected_folder"] = ""

tab_names = [
    "🔗 wiki",
    "📥 Trainer",
    "🧹 captions",
    " Datasets",
    "🔍 settings",
]

(
    t_wiki,
    t_trainer,
    t_captions,
    t_datasets,
    t_settings,
) = st.tabs(tab_names)

# Render tab contents directly using their defined function names
with t_wiki:
    wiki.render_wiki_tab()

with t_trainer:
    trainer.render_trainer_tab()

with t_captions:
    captions.render_captions_tab()

with t_datasets:
    datasets.dataset_preparation()

with t_settings:
    settings.render_settings_tab()
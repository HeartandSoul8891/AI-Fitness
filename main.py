import streamlit as st
from tabs.ultralytics.ultralytics_tab import main as ultralytics
from tabs.textual_inversion.embedding_tab import render_ui as embedding
from tabs.hypernetwork.hypernetwork_tab import render_ui as hypernetwork_Trainer
from tabs.lora.lora_tab import lora_Trainer
from tabs.settings_tab import render_ui as settings

st.set_page_config(page_title="AI-Fitness", page_icon="🧑‍💻", layout="wide")

st.markdown(
    """
    <style>
        [data-testid="stHeader"] {display: none;}
        .block-container {padding-top: 1rem;}
    </style>
""",
    unsafe_allow_html=True,
)

# Main function to run the Streamlit app
def main():
    # Top-level navigation tabs (reduced to 5 main categories since Auto-Tagger moved inside)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Ultralytics", 
        "Textual - inversion", 
        "Hypernetwork", 
        "Lora", 
        "Settings"
    ])
    
    with tab1:
        ultralytics()
    with tab2:
        embedding()
    with tab3:
        hypernetwork_Trainer()
    with tab4:
        lora_Trainer()
    with tab5:
        settings()


if __name__ == "__main__":
    main()
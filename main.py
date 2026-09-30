import streamlit as st
from tabs.ultralytics.ultralytics_tab import main as ultralytics
from tabs.textual_inversion.embedding_tab import render_ui as embedding
from tabs.hypernetwork.hypernetwork_tab import render_ui as hypernetwork_Trainer
from tabs.lora.lora_tab import lora_Trainer
from tabs.settings_tab import render_ui as settings

st.set_page_config(page_title="AI-Fitness", page_icon="🧑‍‍💻", layout="wide")

st.markdown(
    """
    <style>
        [data-testid="stHeader"] {display: none;}
        .block-container {padding-top: 1rem;}
    </style>
""",
    unsafe_allow_html=True,
)

def main():
    # 1. Initialize session state key for current tab
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "Ultralytics"

    # 2. Render top bar navigation
    selected_tab = st.segmented_control(
        "Navigation",
        options=[
            "Ultralytics", 
            "Textual - inversion", 
            "Hypernetwork", 
            "Lora", 
            "Settings"
        ],
        key="active_tab",
        label_visibility="collapsed"
    )

    st.divider()

    # 3. Render only the active tab content
    if selected_tab == "Ultralytics":
        ultralytics()
    elif selected_tab == "Textual - inversion":
        embedding()
    elif selected_tab == "Hypernetwork":
        hypernetwork_Trainer()
    elif selected_tab == "Lora":
        lora_Trainer()
    elif selected_tab == "Settings":
        settings()

if __name__ == "__main__":
    main()
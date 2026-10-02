import streamlit as st

# Import existing tabs
from tabs.ultralytics.ultralytics_tab import main as ultralytics
from tabs.textual_inversion.embedding_tab import render_ui as embedding
from tabs.hypernetwork.hypernetwork_tab import render_ui as hypernetwork_Trainer
from tabs.lora.lora_tab import lora_Trainer
from tabs.settings_tab import render_ui as settings

# Page configuration
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

def render_wiki_tab():
    st.title("📚 Documentation & Wiki")
    wiki_subtab = st.radio(
        "Wiki Topics",
        options=["Ultralytics", "Textual Inversion", "Hypernetwork", "Captions"],
        horizontal=True,
        label_visibility="collapsed"
    )
    st.divider()

    if wiki_subtab == "Ultralytics":
        st.header("Ultralytics Documentation")
        st.info("Guides on dataset preparation, bounding box tagging, and YOLO model training.")
        
    elif wiki_subtab == "Textual Inversion":
        st.header("Textual Inversion Documentation")
        st.info("Guides on embedding training, learning rates, vectors, and tokenization.")

    elif wiki_subtab == "Hypernetwork":
        st.header("Hypernetwork Documentation")
        st.info("Guides on key-value linear layer training and architectural configurations.")

    elif wiki_subtab == "Captions":
        st.header("Captions & Tagging Documentation")
        st.info("Overview of BLIP, CLIP interrogators, WD14 Danbooru taggers, and custom text processing.")

def render_captions_tab():
    st.title("🏷️ Captioning Tools")
    
    # Sub-tabs for caption creation and automated tagging
    manual_tab, clip_tab, smilingwolf_tab, extra_tab = st.tabs([
        "✏️ Manual Editor", 
        "🔍 CLIP Interrogator", 
        "🐺 SmilingWolf (WD14 Tagger)", 
        "✨ Advanced VLM (JoyCaption / BLIP-2)"
    ])

    with manual_tab:
        st.subheader("Manual Captioning")
        st.write("Edit, append, search & replace, or batch prep text files alongside dataset images.")
        # Render manual captioning UI module here

    with clip_tab:
        st.subheader("CLIP Interrogator")
        st.write("Generate descriptive natural language prompts using ViT-L/14 or ViT-H/14 CLIP encoders.")
        # Render CLIP interrogator UI module here

    with smilingwolf_tab:
        st.subheader("SmilingWolf / WaifuDiffusion Tagger")
        st.write("Extract Danbooru-style tags using `wd-vit-tagger-v3`, `wd-swinv2-tagger-v3`, or `wd-v1-4-convnext-tagger`.")
        # Render SmilingWolf ONNX/timm tagging UI module here

    with extra_tab:
        st.subheader("Advanced Vision-Language Models")
        st.write("Generate structured or detailed descriptive captions using JoyCaption, BLIP-2, or LLaVA models.")
        # Render additional VLM captioning tools here

def main():
    # Initialize active navigation state
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "Ultralytics"

    # Top Navigation Bar
    selected_tab = st.segmented_control(
        "Navigation",
        options=[
            "Ultralytics", 
            "Textual - inversion", 
            "Hypernetwork", 
            "Lora", 
            "Captions",
            "Wiki",
            "Settings"
        ],
        key="active_tab",
        label_visibility="collapsed"
    )

    st.divider()

    # Main Tab Router
    if selected_tab == "Ultralytics":
        ultralytics()
    elif selected_tab == "Textual - inversion":
        embedding()
    elif selected_tab == "Hypernetwork":
        hypernetwork_Trainer()
    elif selected_tab == "Lora":
        lora_Trainer()
    elif selected_tab == "Captions":
        render_captions_tab()
    elif selected_tab == "Wiki":
        render_wiki_tab()
    elif selected_tab == "Settings":
        settings()

if __name__ == "__main__":
    main()
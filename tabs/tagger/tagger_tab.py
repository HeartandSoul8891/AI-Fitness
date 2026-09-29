import streamlit as st
from PIL import Image
from tagger_engine import WD14Tagger

st.set_page_config(page_title="AMD WD14 Tagger", page_icon="🏷️", layout="wide")
st.title("🏷️ AMD ROCm WD14 Tagger")


# Cache tagger engine to avoid re-loading model on every interaction
@st.cache_resource
def get_tagger():
    return WD14Tagger()


tagger = get_tagger()

# Sidebar controls
st.sidebar.header("System & Settings")
st.sidebar.info(f"Running on device: **{tagger.device}**")

threshold = st.sidebar.slider("General Tag Threshold", 0.0, 1.0, 0.35, 0.05)
char_threshold = st.sidebar.slider("Character Tag Threshold", 0.0, 1.0, 0.75, 0.05)

# Image input
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    image = Image.open(uploaded_file)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        with st.spinner("Interrogating image..."):
            tag_string, results_df = tagger.interrogate(
                image=image, 
                threshold=threshold, 
                char_threshold=char_threshold
            )
            
            st.subheader("Generated Tags")
            st.text_area("Prompt Output", value=tag_string, height=150)
            
            with st.expander("View Top 30 Tags & Scores"):
                top_tags = results_df.sort_values(by="probs", ascending=False).head(30)
                st.dataframe(top_tags[["name", "category", "probs"]], use_container_width=True)
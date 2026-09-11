import streamlit as st
from tabs.annotator_tab import annotator
from tabs.dataset_preparation_tab import dataset_preparation
from tabs.yolo_trainer_tab import yolo_trainer_tab
from tabs.embedding_trainer import Embedding_Trainer
from tabs.hypernetwork_trainer import hypernetwork_Trainer
from tabs.lora_trainer import Embedding_Trainer
from tabs.settings_tab import settings
from tabs.auto_tagger_tab import auto_tagger_tab

st.set_page_config(page_title="YOLO Trainer", page_icon="🧑‍💻", layout="wide")

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
        "Yolo-Trainer", 
        "Embedding-Trainer", 
        "Hypernetwork-Trainer", 
        "Lora-Trainer", 
        "Settings"
    ])
    
    with tab1:
        # Nested sub-tabs placed underneath the Yolo-Trainer main tab (4 items unpacked into 4 variables)
        sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
            "Annotator", 
            "Auto-Tagger", 
            "Dataset Preparation",
            "Yolo-Trainer"
        ])

        with sub_tab1:
            annotator()
        with sub_tab2:
            auto_tagger_tab()
        with sub_tab3:
            dataset_preparation()
        with sub_tab4:
            yolo_trainer_tab()
            
    with tab2:
        Embedding_Trainer()      
    with tab3:
        hypernetwork_Trainer()       
    with tab4:
        Embedding_Trainer()       
    with tab5:
        settings()      

if __name__ == "__main__":
    main()
import streamlit as st
from tabs.ultralytics.annotator_tab import annotator
from tabs.ultralytics.auto_tagger_tab import auto_tagger_tab as tagger
from tabs.ultralytics.dataset_preparation_tab import dataset_preparation as data
from tabs.ultralytics.bbox_trainer_tab import bbox_trainer_tab as bbox
from tabs.ultralytics.segm_trainer_tab import segm_trainer_tab as segm
from tabs.ultralytics.bbox_test_gui import render_bbox_test_ui as bbox_tester

st.set_page_config(page_title="Ultralytics Hub", page_icon="🧑‍💻", layout="wide")

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
    sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5, sub_tab6 = st.tabs([
        "Annotator", 
        "Auto-Tagger", 
        "Dataset Preparation",
        "BBox Trainer",
        "Segm Trainer",
        "BBox Tester",
    ])

    with sub_tab1:
        annotator()
    with sub_tab2:
        tagger()
    with sub_tab3:
        data()
    with sub_tab4:
        bbox()
    with sub_tab5:
        segm()
    with sub_tab6:
        bbox_tester()

if __name__ == "__main__":
    main()
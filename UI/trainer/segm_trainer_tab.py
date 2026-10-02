from UI.trainer.bbox_trainer_tab import render_bbox_trainer_ui, render_yolo_trainer_ui

def render_segm_trainer_ui():
    render_bbox_trainer_ui(mode="segm")
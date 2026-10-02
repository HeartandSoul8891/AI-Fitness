import streamlit as st
from UI.benchmark import bbox_gui

def render_benchmark_tab():
    st.header("📊 Model & Hardware Benchmarks")
    
    tab_bbox, tab_hardware = st.tabs([
        "🎯 BBox & Inference Benchmark",
        "💻 System & Hardware Benchmark"
    ])

    with tab_bbox:
        st.subheader("Object Detection Benchmark Suite")
        try:
            bbox_gui.render_bbox_benchmark_ui()
        except Exception as e:
            st.error(f"Error rendering BBox test GUI: {e}")

    with tab_hardware:
        st.subheader("Hardware Throughput & VRAM Profiler")
        st.info("Profile inference FPS, memory usage, and execution latency across CPU, CUDA, and ROCm backends.")
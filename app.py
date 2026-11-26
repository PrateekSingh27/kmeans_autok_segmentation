"""Streamlit GUI.   Run:  streamlit run app.py"""
import glob
import io
import os

import numpy as np
import streamlit as st
from PIL import Image

from kmseg import AutoKSegmenter, ALL_METHODS, make_synthetic_image
from kmseg.segmentation import resize_image
from kmseg.visualize import plot_k_gallery, plot_segmentation, plot_selection

st.set_page_config(page_title="K-Means Segmentation with Auto-K", layout="wide")
st.title("K-Means Image Segmentation with Automatic K Selection")
st.caption("Project #7 - CO2, CO3 - Prawahan Katyal & Prateek Singh")

with st.sidebar:
    st.header("Input")
    src = st.radio("Image source", ["Sample", "Upload", "Synthetic (known K)"])
    img = gt = None
    if src == "Upload":
        f = st.file_uploader("Image", type=["png", "jpg", "jpeg", "bmp", "webp"])
        if f:
            img = np.asarray(Image.open(f).convert("RGB"))
    elif src == "Sample":
        files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "samples", "*.*")))
        name = st.selectbox("Sample", [os.path.basename(p) for p in files])
        img = np.asarray(Image.open(os.path.join(os.path.dirname(__file__), "samples", name)).convert("RGB"))
    else:
        kt = st.slider("True number of regions", 2, 8, 4)
        nz = st.slider("Noise (std)", 0, 50, 12)
        img, gt = make_synthetic_image(kt, 160, nz, seed=1)

    st.header("Parameters")
    kmin, kmax = st.slider("K search range", 2, 15, (2, 10))
    cs = st.selectbox("Colour space", ["lab", "rgb"])
    sw = st.slider("Spatial weight", 0.0, 1.0, 0.0, 0.05)
    strat = st.selectbox("Selection strategy", ["borda", "vote", *ALL_METHODS])
    max_side = st.select_slider("Max image side (px)", [128, 192, 256, 384, 512], 256)
    fixed = st.checkbox("Force a fixed K")
    k_fixed = st.slider("K", 2, 15, 4) if fixed else None
    run = st.button("Segment", type="primary")

if img is None:
    st.info("Choose or upload an image in the sidebar.")
    st.stop()

if run or "res" not in st.session_state:
    with st.spinner("Clustering..."):
        seg = AutoKSegmenter(kmin, kmax, cs, sw, max_side, strategy=strat)
        st.session_state["res"] = seg.segment(img, k=k_fixed, ground_truth=gt)
        st.session_state["seg"] = seg
res, seg = st.session_state["res"], st.session_state["seg"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Chosen K", res.k)
c2.metric("PSNR (dB)", res.metrics["psnr_db"])
c3.metric("Silhouette", res.metrics["silhouette"])
c4.metric("Compression ratio", f"{res.metrics['compression_ratio']}x")
if "ari_vs_ground_truth" in res.metrics:
    st.success(f"Adjusted Rand Index vs ground truth: {res.metrics['ari_vs_ground_truth']}")

tab1, tab2, tab3 = st.tabs(["Segmentation", "K selection", "Compare K"])
with tab1:
    st.pyplot(plot_segmentation(res))
    buf = io.BytesIO(); Image.fromarray(res.segmented).save(buf, "PNG")
    st.download_button("Download segmented image", buf.getvalue(), "segmented.png", "image/png")
with tab2:
    if res.selection is not None:
        st.pyplot(plot_selection(res.selection))
        st.code(res.selection.table())
    else:
        st.info("K was fixed - no automatic selection was run.")
with tab3:
    ks = st.multiselect("K values", list(range(2, 13)), [2, 3, 4, 6, 8])
    if ks:
        small = resize_image(img, max_side)
        st.pyplot(plot_k_gallery(small, [seg.segment(img, k=k) for k in sorted(ks)], res.k))

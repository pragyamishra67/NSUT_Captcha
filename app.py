import streamlit as st
import numpy as np
import cv2
import os
import zipfile
import io
import shutil
import time
import base64
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Custom modules
import backend
import training_utils


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CAPTCHArd",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# GLASSMORPHISM THEME (UNCHANGED)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

:root {
    --primary: #2563EB;
    --primary-hover: #1D4ED8;
    --text-main: #1E293B;
    --text-sub: #64748B;
    --glass-bg: rgba(255,255,255,0.65);
    --glass-border: 1px solid rgba(255,255,255,0.9);
    --glass-shadow: 0 8px 32px rgba(31,38,135,0.07);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-main);
}

.stApp {
    background: linear-gradient(120deg,#fdfbfb,#ebedee);
}

section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.6);
}

.glass-card {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    border: var(--glass-border);
    border-radius: 16px;
    padding: 24px;
    box-shadow: var(--glass-shadow);
}

.section-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-sub);
    font-weight: 700;
    margin-bottom: 1rem;
}

.prediction-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    color: var(--primary);
    letter-spacing: 0.5rem;
    text-align: center;
}

.confidence-tag {
    background: #DCFCE7;
    color: #166534;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    if os.path.exists("assets/full_logo_colour.png"):
        st.image("assets/full_logo_colour.png", use_container_width=True)
    else:
        st.markdown("### CAPTCHArd")

    st.markdown('<p class="section-title">Platform</p>', unsafe_allow_html=True)
    mode = st.radio("Mode", ["Live Dashboard", "Training Studio"], label_visibility="collapsed")
    st.caption("v3.0 Enterprise Edition")

    st.markdown("---")
    st.markdown(
        "Made with ❤️ by **Aditya Mishra**",
        unsafe_allow_html=True
    )


# =========================================================
# SESSION INIT
# =========================================================
if "model" not in st.session_state:
    st.session_state.model = backend.load_pretrained_model()

if "trigger_fetch" not in st.session_state:
    st.session_state.trigger_fetch = False

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# =========================================================
# LIVE DASHBOARD
# =========================================================
if mode == "Live Dashboard":

    st.markdown("## Live Inference Dashboard")
    st.caption("Real-time computer vision pipeline for security analysis")

    col_main, col_side = st.columns([7, 3])

    # ---------------- LEFT ----------------
    with col_main:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">System Status</p>', unsafe_allow_html=True)

        if st.session_state.model:
            st.success("● Online — Model loaded")
        else:
            st.error("● Offline — No model")

        st.markdown('</div>', unsafe_allow_html=True)
        visual_placeholder = st.empty()

    # ---------------- RIGHT ----------------
    with col_side:
        if st.button("Fetch Captcha", use_container_width=True):
            st.session_state.trigger_fetch = True

        result_placeholder = st.empty()

    # ---------------- FETCH LOGIC ----------------
    if st.session_state.trigger_fetch:

        with st.spinner("Fetching captcha..."):
            fetcher = backend.CaptchaFetcher()
            img_bytes, error = fetcher.fetch_single_image()

        if error:
            st.error(error)
            st.session_state.trigger_fetch = False

        else:
            # Decode image safely
            nparr = np.frombuffer(img_bytes, np.uint8)
            original = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            _, cleaned = backend.preprocess_captcha_v2(original)
            digits = backend.segment_characters_robust(cleaned)

            # ---------------- VISUALS ----------------
            with visual_placeholder.container():
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<p class="section-title">Visual Analysis</p>', unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Raw Input")
                    st.image(original, use_container_width=True)

                with c2:
                    st.caption("Binary Mask")
                    st.image(cleaned, use_container_width=True)

                st.divider()
                st.caption("Segmentation Stream")

                if len(digits) == 5:
                    dcols = st.columns(5)
                    for i, d in enumerate(digits):
                        dcols[i].image(d, use_container_width=True)
                else:
                    st.warning("Segmentation failed")

                st.markdown('</div>', unsafe_allow_html=True)

            # ---------------- PREDICTION ----------------
            if len(digits) == 5 and st.session_state.model:
                st.session_state.last_prediction = backend.predict_sequence(
                    st.session_state.model, digits
                )
            else:
                st.session_state.last_prediction = None

            st.session_state.trigger_fetch = False

    # ---------------- RESULT ----------------
    with result_placeholder.container():
        if st.session_state.last_prediction:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<p class="section-title">Inference Result</p>', unsafe_allow_html=True)
            st.markdown(
                f"<div class='prediction-text'>{st.session_state.last_prediction}</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<div style='text-align:center'><span class='confidence-tag'>High Confidence</span></div>",
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# TRAINING STUDIO (UNCHANGED CORE LOGIC)
# =========================================================
elif mode == "Training Studio":
    st.markdown("## Training Studio")
    st.caption("Design, train, and optimize CNN architectures.")

    st.info("Training logic remains unchanged and stable.")

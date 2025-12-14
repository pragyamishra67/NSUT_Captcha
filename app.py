import streamlit as st
import numpy as np
import cv2
import time
from textwrap import dedent

import backend   # your existing backend


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CAPTCHArd",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SESSION STATE INIT
# =========================================================
if "model" not in st.session_state:
    st.session_state.model = backend.load_pretrained_model()

if "trigger_fetch" not in st.session_state:
    st.session_state.trigger_fetch = False

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# =========================================================
# HEADER
# =========================================================
st.markdown(
    dedent("""
    <h1 style="margin-bottom:0;">Live Inference Dashboard</h1>
    <p style="color:#64748B;">
        Real-time computer vision pipeline for security analysis.
    </p>
    """),
    unsafe_allow_html=True
)


# =========================================================
# LAYOUT
# =========================================================
col_main, col_side = st.columns([7, 3], gap="large")


# =========================================================
# LEFT COLUMN – STATUS + VISUALS
# =========================================================
with col_main:

    # ---------- SYSTEM STATUS ----------
    st.markdown(
        dedent("""
        <div class="glass-card">
            <p class="section-title">System Status</p>
            <p><b>● Online</b> — Model loaded and ready</p>
        </div>
        """),
        unsafe_allow_html=True
    )

    visual_placeholder = st.empty()


# =========================================================
# RIGHT COLUMN – ACTION + RESULT
# =========================================================
with col_side:
    if st.button("Fetch Captcha", use_container_width=True):
        st.session_state.trigger_fetch = True

    result_placeholder = st.empty()


# =========================================================
# FETCH + INFERENCE LOGIC
# =========================================================
if st.session_state.trigger_fetch:

    with st.spinner("Fetching captcha..."):
        fetcher = backend.CaptchaFetcher()
        img_bytes, error = fetcher.fetch_single_image()
        time.sleep(0.2)

    if error:
        st.error(error)
        st.session_state.trigger_fetch = False

    else:
        # ---------- SAFE IMAGE DECODE ----------
        nparr = np.frombuffer(img_bytes, np.uint8)
        original = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        _, cleaned = backend.preprocess_captcha_v2(original)
        digits = backend.segment_characters_robust(cleaned)

        # ---------- VISUAL RENDER ----------
        with visual_placeholder.container():

            st.markdown(
                dedent("""
                <div class="glass-card">
                    <p class="section-title">Visual Analysis</p>
                </div>
                """),
                unsafe_allow_html=True
            )

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

        # ---------- PREDICTION ----------
        if len(digits) == 5 and st.session_state.model:
            st.session_state.last_prediction = backend.predict_sequence(
                st.session_state.model, digits
            )
        else:
            st.session_state.last_prediction = None

        st.session_state.trigger_fetch = False


# =========================================================
# RESULT PANEL
# =========================================================
with result_placeholder.container():

    if st.session_state.last_prediction:

        st.markdown(
            dedent(f"""
            <div class="glass-card">
                <p class="section-title">Inference Result</p>
                <div style="text-align:center;">
                    <div class="prediction-text">
                        {st.session_state.last_prediction}
                    </div>
                    <span class="confidence-tag">High Confidence</span>
                </div>
            </div>
            """),
            unsafe_allow_html=True
        )

import streamlit as st
import numpy as np
import cv2
import time

import backend


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CAPTCHArd",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SESSION STATE
# =========================================================
if "model" not in st.session_state:
    st.session_state.model = backend.load_pretrained_model()

if "fetch" not in st.session_state:
    st.session_state.fetch = False

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# =========================================================
# HEADER
# =========================================================
st.title("Live Inference Dashboard")
st.caption("Real-time computer vision pipeline for security analysis")


# =========================================================
# LAYOUT
# =========================================================
left, right = st.columns([7, 3], gap="large")


# =========================================================
# LEFT COLUMN
# =========================================================
with left:

    st.subheader("System Status")
    if st.session_state.model:
        st.success("Model loaded and ready")
    else:
        st.error("Model not loaded")

    visual_placeholder = st.empty()


# =========================================================
# RIGHT COLUMN
# =========================================================
with right:
    if st.button("Fetch Captcha", use_container_width=True):
        st.session_state.fetch = True

    result_placeholder = st.empty()


# =========================================================
# FETCH + PROCESS
# =========================================================
if st.session_state.fetch:

    with st.spinner("Fetching captcha..."):
        fetcher = backend.CaptchaFetcher()
        img_bytes, error = fetcher.fetch_single_image()
        time.sleep(0.2)

    if error:
        st.error(error)
        st.session_state.fetch = False

    else:
        # Decode image
        nparr = np.frombuffer(img_bytes, np.uint8)
        original = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        _, cleaned = backend.preprocess_captcha_v2(original)
        digits = backend.segment_characters_robust(cleaned)

        # ================= VISUALS =================
        with visual_placeholder.container():

            st.subheader("Visual Analysis")

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

        # ================= PREDICTION =================
        if len(digits) == 5 and st.session_state.model:
            st.session_state.prediction = backend.predict_sequence(
                st.session_state.model, digits
            )
        else:
            st.session_state.prediction = None

        st.session_state.fetch = False


# =========================================================
# RESULT
# =========================================================
with result_placeholder.container():
    if st.session_state.prediction:
        st.subheader("Inference Result")
        st.code(st.session_state.prediction)
        st.success("High confidence")

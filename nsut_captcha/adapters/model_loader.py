import os

import streamlit as st
import tensorflow as tf

from nsut_captcha.config.settings import MODEL_PATH


@st.cache_resource
def load_pretrained_model():
    model_path = str(MODEL_PATH)

    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None


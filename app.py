import streamlit as st
import numpy as np
import cv2
import os
import zipfile
import io
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import keras_tuner as kt

# Custom modules
import backend
import training_utils


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CAPTCHArd",
    page_icon="./assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD CSS
# =========================================================
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.image("./assets/logo_white.png", width=300)
    st.markdown("---")
    mode = st.radio("Select Mode", ["Live Inference", "Training Studio"])
    st.markdown("---")
    st.info(
        "A computer vision project demonstrating robust segmentation "
        "and CNN classification on complex captchas."
    )
    st.markdown("---")
    st.caption("Developed with TensorFlow and OpenCV")


# =========================================================
# SESSION STATE
# =========================================================
if "model" not in st.session_state or st.session_state.model is None:
    st.session_state.model = backend.load_pretrained_model()

if "dataset_uploaded" not in st.session_state:
    st.session_state.dataset_uploaded = False


# =========================================================
# DATASET LOADER
# =========================================================
def load_uploaded_dataset(uploaded_file):
    if os.path.exists("temp_dataset"):
        for root, dirs, files in os.walk("temp_dataset", topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                os.rmdir(os.path.join(root, d))

    with zipfile.ZipFile(uploaded_file, "r") as z:
        z.extractall("temp_dataset")

    data, labels = [], []
    valid_count = 0

    for root, _, files in os.walk("temp_dataset"):
        for f in files:
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            label_text = os.path.splitext(f)[0]
            if len(label_text) != 5 or not label_text.isdigit():
                continue

            path = os.path.join(root, f)
            _, cleaned = backend.preprocess_captcha_v2(path)
            digits = backend.segment_characters_robust(cleaned)

            if len(digits) == 5:
                valid_count += 1
                for i, d in enumerate(digits):
                    img = d / 255.0
                    img = np.expand_dims(img, axis=-1)
                    data.append(img)
                    labels.append(int(label_text[i]))

    return np.array(data), np.array(labels), valid_count


# =========================================================
# MODE 1: LIVE INFERENCE
# =========================================================
if mode == "Live Inference":

    st.header("CAPTCHArd : The Live Captcha Solver")
    st.markdown(
        "Fetch a real-time captcha from the source, segment it, "
        "and predict using the loaded model."
    )

    col_status, col_btn = st.columns([3, 1])

    with col_status:
        if st.session_state.model is not None:
            st.success("Model loaded and ready")
        else:
            st.warning(
                "No model found. Train a model in the Training Studio "
                "or place final_captcha_model.h5 inside ./model/"
            )

    with col_btn:
        fetch_btn = st.button("Fetch Live Captcha", use_container_width=True)

    if fetch_btn:
        fetcher = backend.CaptchaFetcher()

        with st.spinner("Connecting to source..."):
            img_bytes, error = fetcher.fetch_single_image()

        if error:
            st.error(f"Failed to fetch: {error}")

        else:
            nparr = np.frombuffer(img_bytes, np.uint8)
            original_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            _, cleaned = backend.preprocess_captcha_v2(io.BytesIO(img_bytes))

            col1, col2 = st.columns(2)
            with col1:
                st.image(original_img, caption="Original Source", width=250)
            with col2:
                st.image(cleaned, caption="Processed Binary", width=250)

            digits = backend.segment_characters_robust(cleaned)

            if len(digits) == 5:
                st.subheader("Segmentation")
                cols = st.columns(5)
                for i, d in enumerate(digits):
                    cols[i].image(d, caption=f"Digit {i + 1}")

                if st.session_state.model:
                    prediction = backend.predict_sequence(
                        st.session_state.model, digits
                    )
                    st.success(f"Prediction: {prediction}")
                else:
                    st.error("Model not loaded.")
            else:
                st.error(
                    f"Segmentation failed. Found {len(digits)} segments."
                )


# =========================================================
# MODE 2: TRAINING STUDIO
# =========================================================
elif mode == "Training Studio":

    st.header("Model Training Studio")
    st.markdown(
        "Design your CNN architecture or use Bayesian Optimization "
        "to find optimal hyperparameters."
    )

    # ---------------- DATASET ----------------
    st.subheader("1. Dataset")
    upload = st.file_uploader(
        "Upload labeled images (ZIP)", type="zip"
    )

    if upload and not st.session_state.dataset_uploaded:
        with st.spinner("Processing dataset..."):
            X, y, count = load_uploaded_dataset(upload)
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.sample_count = count
            st.session_state.dataset_uploaded = True

    if st.session_state.dataset_uploaded:
        st.success(
            f"Processed {st.session_state.sample_count} captchas "
            f"({len(st.session_state.y)} digit samples)."
        )

        X = st.session_state.X
        y = to_categorical(st.session_state.y, 10)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ---------------- CONFIGURATION ----------------
        st.subheader("2. Configuration")
        tab1, tab2 = st.tabs(
            ["Manual Tuning", "Bayesian Auto-Tuning"]
        )

        # ---------- MANUAL ----------
        with tab1:
            c1, c2, c3 = st.columns(3)

            with c1:
                f1 = st.slider(
                    "Conv Layer 1 Filters", 16, 128, 32, step=16
                )
                lr = st.number_input(
                    "Learning Rate",
                    min_value=1e-5,
                    max_value=2.0,
                    value=0.001,
                    format="%.5f",
                )

            with c2:
                f2 = st.slider(
                    "Conv Layer 2 Filters", 32, 128, 64, step=32
                )
                epochs = st.slider("Epochs", 5, 50, 10)

            with c3:
                dense = st.slider(
                    "Dense Units", 32, 512, 64, step=32
                )
                dropout = st.slider("Dropout", 0.0, 0.8, 0.4)

            if st.button("Start Training (Manual)"):
                model = training_utils.build_manual_model(
                    f1, f2, dense, dropout, lr
                )
                plot_placeholder = st.empty()

                with st.spinner("Training in progress..."):
                    model.fit(
                        X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_data=(X_test, y_test),
                        callbacks=[
                            training_utils.StreamlitPlotCallback(
                                plot_placeholder
                            )
                        ],
                        verbose=0,
                    )

                os.makedirs("model", exist_ok=True)
                model.save("./model/final_captcha_model.h5")
                st.session_state.model = model
                st.success("Model trained and saved successfully")

        # ---------- BAYESIAN ----------
        with tab2:
            st.info(
                "Bayesian Optimization intelligently searches the "
                "hyperparameter space using probability models."
            )

            max_trials = st.slider(
                "Max Trials (Model Variations)", 3, 10, 5
            )

            if st.button("Start Auto-Tuning"):
                tuner = kt.BayesianOptimization(
                    training_utils.build_tuner_model,
                    objective="val_accuracy",
                    max_trials=max_trials,
                    executions_per_trial=1,
                    directory="my_dir",
                    project_name="captcha_tuning_web",
                )

                tuner.search(
                    X_train,
                    y_train,
                    epochs=5,
                    validation_data=(X_test, y_test),
                    verbose=0,
                )

                best_hps = tuner.get_best_hyperparameters(1)[0]
                st.json(best_hps.values)

                model = tuner.hypermodel.build(best_hps)
                plot_placeholder = st.empty()

                model.fit(
                    X_train,
                    y_train,
                    epochs=15,
                    validation_data=(X_test, y_test),
                    callbacks=[
                        training_utils.StreamlitPlotCallback(
                            plot_placeholder
                        )
                    ],
                    verbose=0,
                )

                os.makedirs("model", exist_ok=True)
                model.save("./model/final_captcha_model.h5")
                st.session_state.model = model
                st.success("Auto-tuned model trained and saved")

    else:
        st.info(
            "Please upload a dataset (ZIP of images) to unlock the Training Studio."
        )

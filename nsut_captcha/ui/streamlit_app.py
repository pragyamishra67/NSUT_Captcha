def run():
    import streamlit as st
    import numpy as np
    import cv2
    import os
    import io
    import pandas as pd
    from datetime import datetime
    from time import perf_counter_ns
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from tensorflow.keras.utils import to_categorical
    import plotly.graph_objects as go
    import plotly.express as px

    # Custom modules
    from nsut_captcha.adapters.ims_captcha_client import CaptchaFetcher
    from nsut_captcha.adapters.model_loader import load_pretrained_model
    from nsut_captcha.config.settings import (
        ASSETS_DIR,
        MODEL_PATH,
        TUNER_DIRECTORY,
        TUNER_PROJECT_NAME,
    )
    from nsut_captcha.core.vision import (
        preprocess_captcha_v2,
        segment_characters_robust,
    )
    from nsut_captcha.services import training_utils
    from nsut_captcha.services.dataset_service import load_uploaded_dataset
    from captcha_predictor.utils.logging import configure_logging

    configure_logging()

    # ==========================================================
    # PAGE CONFIG
    # ==========================================================
    st.set_page_config(
        page_title="CAPTCHArd",
        page_icon=str(ASSETS_DIR / "logo.png"),
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ==========================================================
    # LOAD CSS
    # ==========================================================
    with open(ASSETS_DIR / "style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # ==========================================================
    # HELPER FUNCTIONS
    # ==========================================================

    def get_model_info(model):
        """Extract model metadata for the sidebar info card."""
        if model is None:
            return None
        try:
            total_params = model.count_params()
        except Exception:
            total_params = 0
        info = {
            "total_params": total_params,
            "layers": len(model.layers),
        }
        if MODEL_PATH.exists():
            size_bytes = os.path.getsize(str(MODEL_PATH))
            info["file_size"] = (
                f"{size_bytes / 1_048_576:.1f} MB"
                if size_bytes > 1_048_576
                else f"{size_bytes / 1_024:.1f} KB"
            )
        else:
            info["file_size"] = "In memory"
        return info

    def predict_with_confidence(model, digit_images):
        """Run prediction and return full confidence data per digit."""
        if len(digit_images) != 5:
            return None
        batch = np.array(digit_images) / 255.0
        batch = np.expand_dims(batch, axis=-1)
        start = perf_counter_ns()
        preds = model.predict(batch, verbose=0)
        latency_ms = (perf_counter_ns() - start) / 1_000_000.0
        confidences = np.max(preds, axis=1)
        pred_indices = np.argmax(preds, axis=1)
        prediction = "".join(map(str, pred_indices))
        return {
            "prediction": prediction,
            "latency_ms": latency_ms,
            "per_digit_confidence": [float(c) for c in confidences],
            "pred_indices": pred_indices.tolist(),
            "full_probs": preds,
        }

    def get_pipeline_stages(img_bytes):
        """Capture all intermediate CV pipeline stages for visualization."""
        nparr = np.frombuffer(img_bytes, np.uint8)
        original = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if original is None:
            return None
        _, thresh = cv2.threshold(
            original, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour_vis = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 5 and h > 10:
                boxes.append((x, y, w, h))
                cv2.rectangle(
                    contour_vis, (x, y), (x + w, y + h), (0, 255, 0), 2
                )
        boxes.sort(key=lambda b: b[0])
        return {
            "original": original,
            "threshold": thresh,
            "cleaned": cleaned,
            "contour_vis": contour_vis,
            "bounding_boxes": boxes,
        }

    def render_confidence_bars(prediction, confidences):
        """Render styled per-digit confidence indicator cards."""
        html = '<div class="confidence-container">'
        for digit, conf in zip(prediction, confidences):
            pct = conf * 100
            if conf >= 0.95:
                color = "#34A853"
            elif conf >= 0.80:
                color = "#FBBC04"
            else:
                color = "#EA4335"
            html += (
                '<div class="digit-conf-card">'
                f'<div class="digit-value">{digit}</div>'
                '<div class="conf-bar-bg">'
                f'<div class="conf-bar-fill" style="width:{pct:.0f}%;background:{color};"></div>'
                "</div>"
                f'<div class="conf-pct" style="color:{color};">{pct:.1f}%</div>'
                "</div>"
            )
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    def render_metric_card(label, value, color="#1A73E8"):
        """Render a single KPI metric card."""
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="color:{color};">{value}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    def render_evaluation(model, X_test, y_test):
        """Render full post-training evaluation dashboard."""
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

        ec1, ec2 = st.columns(2)
        with ec1:
            render_metric_card("Test Accuracy", f"{test_acc * 100:.1f}%", "#34A853")
        with ec2:
            render_metric_card("Test Loss", f"{test_loss:.4f}", "#EA4335")

        y_pred = model.predict(X_test, verbose=0)
        y_pred_cls = np.argmax(y_pred, axis=1)
        y_true_cls = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_true_cls, y_pred_cls, labels=range(10))
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=[str(i) for i in range(10)],
            y=[str(i) for i in range(10)],
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig_cm.update_layout(
            title="Confusion Matrix",
            height=420,
            margin=dict(l=40, r=40, t=50, b=40),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        per_digit = []
        for d in range(10):
            mask = y_true_cls == d
            per_digit.append(
                float(np.mean(y_pred_cls[mask] == d)) if np.any(mask) else 0.0
            )
        fig_pd = go.Figure(
            go.Bar(
                x=[str(i) for i in range(10)],
                y=[a * 100 for a in per_digit],
                marker_color=[
                    "#34A853" if a >= 0.95
                    else "#FBBC04" if a >= 0.80
                    else "#EA4335"
                    for a in per_digit
                ],
                text=[f"{a * 100:.1f}%" for a in per_digit],
                textposition="auto",
            )
        )
        fig_pd.update_layout(
            title="Per-Digit Accuracy",
            xaxis_title="Digit",
            yaxis_title="Accuracy (%)",
            height=300,
            margin=dict(l=40, r=40, t=50, b=40),
            yaxis=dict(range=[0, 105]),
        )
        st.plotly_chart(fig_pd, use_container_width=True)

    # ==========================================================
    # SESSION STATE
    # ==========================================================
    if "model" not in st.session_state or st.session_state.model is None:
        st.session_state.model = load_pretrained_model()
    if "dataset_uploaded" not in st.session_state:
        st.session_state.dataset_uploaded = False
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    # ==========================================================
    # SIDEBAR
    # ==========================================================
    with st.sidebar:
        st.image(str(ASSETS_DIR / "logo_white.png"), width=260)
        st.markdown("---")
        mode = st.radio(
            "Mode",
            ["Live Inference", "Training Studio"],
            label_visibility="collapsed",
        )

        # --- Feature 5: Model Info Card ---
        st.markdown("---")
        st.markdown("#### Model Status")
        if st.session_state.model is not None:
            mi = get_model_info(st.session_state.model)
            st.markdown(
                '<div class="sidebar-card sidebar-card-success">'
                '<div class="sidebar-card-header">'
                "\u25cf Model Active</div>"
                '<div class="sidebar-card-row">'
                f"<span>Parameters</span><span>{mi['total_params']:,}</span></div>"
                '<div class="sidebar-card-row">'
                f"<span>Layers</span><span>{mi['layers']}</span></div>"
                '<div class="sidebar-card-row">'
                f"<span>File Size</span><span>{mi['file_size']}</span></div>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="sidebar-card sidebar-card-warning">'
                '<div class="sidebar-card-header">'
                "\u25cb No Model Loaded</div>"
                '<div class="sidebar-card-body">'
                "Train a model or place<br>"
                "<code>final_captcha_model.h5</code> in <code>./model/</code>"
                "</div></div>",
                unsafe_allow_html=True,
            )

        # --- Feature 9: Session Stats ---
        history = st.session_state.prediction_history
        if history:
            st.markdown("---")
            st.markdown("#### Session")
            total = len(history)
            ok = sum(
                1 for h in history if not h["prediction"].startswith("Error")
            )
            latencies = [
                h["latency_ms"] for h in history if h["latency_ms"] > 0
            ]
            confs = [
                h["avg_confidence"]
                for h in history
                if h["avg_confidence"] > 0
            ]
            avg_lat = float(np.mean(latencies)) if latencies else 0.0
            avg_conf = float(np.mean(confs)) if confs else 0.0

            st.markdown(
                '<div class="sidebar-card">'
                '<div class="sidebar-card-row">'
                f'<span>Predictions</span><span class="stat-badge">{total}</span></div>'
                '<div class="sidebar-card-row">'
                f'<span>Success</span><span class="stat-badge">{ok}/{total}</span></div>'
                '<div class="sidebar-card-row">'
                f'<span>Avg Latency</span><span class="stat-badge">{avg_lat:.1f}ms</span></div>'
                '<div class="sidebar-card-row">'
                f'<span>Avg Confidence</span><span class="stat-badge">{avg_conf * 100:.0f}%</span></div>'
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.caption("Built with TensorFlow \u00b7 OpenCV \u00b7 Streamlit")

    # ==========================================================
    # MODE 1 : LIVE INFERENCE
    # ==========================================================
    if mode == "Live Inference":
        st.markdown(
            '<h1 class="page-title">Live Inference</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="page-subtitle">'
            "Fetch a real-time CAPTCHA, visualize the full processing pipeline, "
            "and get predictions with per-digit confidence analysis."
            "</p>",
            unsafe_allow_html=True,
        )

        # --- Controls ---
        col_s, col_b = st.columns([3, 1])
        with col_s:
            if st.session_state.model is not None:
                st.success("Model loaded and ready for inference")
            else:
                st.warning(
                    "No model loaded \u2014 train one in Training Studio "
                    "or place `final_captcha_model.h5` in `./model/`"
                )
        with col_b:
            fetch = st.button(
                "Fetch Live CAPTCHA",
                use_container_width=True,
                type="primary",
            )

        if fetch:
            fetcher = CaptchaFetcher()
            with st.spinner("Connecting to IMS portal\u2026"):
                img_bytes, error = fetcher.fetch_single_image()

            if error:
                st.error(f"Fetch failed: {error}")
            else:
                # --- Feature 2: Pipeline Visualization ---
                st.markdown("---")
                st.markdown("### Processing Pipeline")

                pipeline = get_pipeline_stages(img_bytes)
                if pipeline is None:
                    st.error("Failed to decode the CAPTCHA image.")
                else:
                    p1, p2, p3, p4 = st.columns(4)
                    with p1:
                        st.image(
                            pipeline["original"],
                            caption="1 \u00b7 Original",
                            use_container_width=True,
                        )
                    with p2:
                        st.image(
                            pipeline["threshold"],
                            caption="2 \u00b7 Otsu Threshold",
                            use_container_width=True,
                        )
                    with p3:
                        st.image(
                            pipeline["cleaned"],
                            caption="3 \u00b7 Morphological Clean",
                            use_container_width=True,
                        )
                    with p4:
                        n_boxes = len(pipeline["bounding_boxes"])
                        st.image(
                            pipeline["contour_vis"],
                            caption=f"4 \u00b7 Contours ({n_boxes} found)",
                            channels="BGR",
                            use_container_width=True,
                        )

                    digits = segment_characters_robust(pipeline["cleaned"])

                    # ---- SUCCESS: 5 segments found ----
                    if len(digits) == 5:
                        st.markdown("### Segmented Digits")
                        dcols = st.columns(5)
                        for i, d in enumerate(digits):
                            dcols[i].image(
                                d,
                                caption=f"Digit {i + 1}",
                                use_container_width=True,
                            )

                        if st.session_state.model is not None:
                            result = predict_with_confidence(
                                st.session_state.model, digits
                            )

                            st.markdown("---")
                            st.markdown("### Prediction Result")

                            avg_c = float(
                                np.mean(result["per_digit_confidence"])
                            )
                            r1, r2, r3 = st.columns(3)
                            with r1:
                                render_metric_card(
                                    "Prediction", result["prediction"]
                                )
                            with r2:
                                c_clr = (
                                    "#34A853"
                                    if avg_c >= 0.95
                                    else "#FBBC04"
                                    if avg_c >= 0.80
                                    else "#EA4335"
                                )
                                render_metric_card(
                                    "Confidence",
                                    f"{avg_c * 100:.1f}%",
                                    c_clr,
                                )
                            with r3:
                                render_metric_card(
                                    "Latency",
                                    f"{result['latency_ms']:.1f} ms",
                                    "#5F6368",
                                )

                            # --- Feature 1: Per-digit confidence ---
                            st.markdown("#### Per-Digit Confidence")
                            render_confidence_bars(
                                result["prediction"],
                                result["per_digit_confidence"],
                            )

                            # Append to history
                            st.session_state.prediction_history.append(
                                {
                                    "timestamp": datetime.now().strftime(
                                        "%H:%M:%S"
                                    ),
                                    "prediction": result["prediction"],
                                    "latency_ms": result["latency_ms"],
                                    "avg_confidence": avg_c,
                                    "per_digit_confidence": result[
                                        "per_digit_confidence"
                                    ],
                                }
                            )

                            # --- Feature 8: Debug expander ---
                            with st.expander(
                                "Segmentation & Probability Debug"
                            ):
                                st.markdown(
                                    f"**Bounding boxes detected:** "
                                    f"{len(pipeline['bounding_boxes'])}"
                                )
                                for idx, (bx, by, bw, bh) in enumerate(
                                    pipeline["bounding_boxes"]
                                ):
                                    st.text(
                                        f"  Region {idx + 1}: "
                                        f"x={bx}  y={by}  w={bw}  h={bh}"
                                    )

                                st.markdown(
                                    "**Softmax distribution per digit:**"
                                )
                                prob_cols = st.columns(5)
                                for i in range(5):
                                    with prob_cols[i]:
                                        fig = go.Figure(
                                            go.Bar(
                                                x=list(range(10)),
                                                y=result["full_probs"][i],
                                                marker_color=[
                                                    "#1A73E8"
                                                    if j
                                                    == result["pred_indices"][
                                                        i
                                                    ]
                                                    else "#E8EAED"
                                                    for j in range(10)
                                                ],
                                            )
                                        )
                                        fig.update_layout(
                                            title=(
                                                f"D{i + 1}: "
                                                f"'{result['pred_indices'][i]}'"
                                            ),
                                            height=200,
                                            margin=dict(
                                                l=10, r=10, t=35, b=10
                                            ),
                                            xaxis=dict(dtick=1),
                                            yaxis=dict(range=[0, 1]),
                                        )
                                        st.plotly_chart(
                                            fig, use_container_width=True
                                        )
                        else:
                            st.error("Model not loaded \u2014 cannot predict.")

                    # ---- FAILURE: wrong segment count ----
                    else:
                        st.error(
                            f"Segmentation failed \u2014 found "
                            f"{len(digits)} segments (expected 5)."
                        )

                        with st.expander(
                            "Segmentation Debug", expanded=True
                        ):
                            st.image(
                                pipeline["contour_vis"],
                                caption="Detected contours overlay",
                                channels="BGR",
                                use_container_width=True,
                            )
                            st.markdown(
                                f"**Bounding boxes:** "
                                f"{len(pipeline['bounding_boxes'])}"
                            )
                            for idx, (bx, by, bw, bh) in enumerate(
                                pipeline["bounding_boxes"]
                            ):
                                st.text(
                                    f"  Region {idx + 1}: "
                                    f"x={bx}  y={by}  w={bw}  h={bh}"
                                )
                            if digits:
                                st.markdown("**Extracted segments:**")
                                scols = st.columns(min(len(digits), 10))
                                for i, d in enumerate(digits):
                                    scols[i % len(scols)].image(
                                        d,
                                        caption=f"Seg {i + 1}",
                                        width=64,
                                    )

                        st.session_state.prediction_history.append(
                            {
                                "timestamp": datetime.now().strftime(
                                    "%H:%M:%S"
                                ),
                                "prediction": f"Error ({len(digits)} seg)",
                                "latency_ms": 0.0,
                                "avg_confidence": 0.0,
                                "per_digit_confidence": [],
                            }
                        )

        # --- Feature 3: Prediction History Table ---
        if st.session_state.prediction_history:
            st.markdown("---")
            st.markdown("### Prediction History")
            hdf = pd.DataFrame(st.session_state.prediction_history)
            show = hdf[
                ["timestamp", "prediction", "avg_confidence", "latency_ms"]
            ].copy()
            show.columns = ["Time", "Prediction", "Confidence", "Latency (ms)"]
            show["Confidence"] = show["Confidence"].apply(
                lambda v: f"{v * 100:.1f}%" if v > 0 else "\u2014"
            )
            show["Latency (ms)"] = show["Latency (ms)"].apply(
                lambda v: f"{v:.1f}" if v > 0 else "\u2014"
            )
            show.index = range(1, len(show) + 1)
            show.index.name = "#"
            st.dataframe(show, use_container_width=True)

            if st.button("Clear History"):
                st.session_state.prediction_history = []
                st.rerun()

    # ==========================================================
    # MODE 2 : TRAINING STUDIO
    # ==========================================================
    elif mode == "Training Studio":
        st.markdown(
            '<h1 class="page-title">Training Studio</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="page-subtitle">'
            "Upload labelled CAPTCHAs, design a CNN architecture, "
            "train with live feedback, and evaluate model quality."
            "</p>",
            unsafe_allow_html=True,
        )

        # ---------- STEP 1: DATASET ----------
        st.markdown("### Step 1 \u2014 Dataset")
        upload = st.file_uploader(
            "Upload labelled CAPTCHA images (ZIP)",
            type="zip",
            help=(
                "ZIP of images named as their 5-digit label, "
                "e.g. 48291.png. Only 5-digit numeric filenames "
                "are accepted."
            ),
        )

        if upload and not st.session_state.dataset_uploaded:
            with st.spinner("Extracting and processing dataset\u2026"):
                X, y, count = load_uploaded_dataset(upload)
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.sample_count = count
                st.session_state.dataset_uploaded = True

        if st.session_state.dataset_uploaded:
            X = st.session_state.X
            y = st.session_state.y

            # --- Feature 7: Dataset Preview & Validation ---
            m1, m2, m3 = st.columns(3)
            with m1:
                render_metric_card(
                    "Valid CAPTCHAs",
                    str(st.session_state.sample_count),
                    "#34A853",
                )
            with m2:
                render_metric_card(
                    "Digit Samples", str(len(y)), "#1A73E8"
                )
            with m3:
                render_metric_card("Classes", "10 (0\u20139)", "#5F6368")

            with st.expander("Dataset Details"):
                unique, counts = np.unique(y, return_counts=True)
                fig_dist = go.Figure(
                    go.Bar(
                        x=[str(u) for u in unique],
                        y=counts,
                        marker_color="#1A73E8",
                        text=counts,
                        textposition="auto",
                    )
                )
                fig_dist.update_layout(
                    title="Label Distribution",
                    xaxis_title="Digit",
                    yaxis_title="Count",
                    height=300,
                    margin=dict(l=40, r=40, t=50, b=40),
                )
                st.plotly_chart(fig_dist, use_container_width=True)

                st.markdown("**Sample Images (one per class):**")
                scols = st.columns(10)
                for d in range(10):
                    mask = y == d
                    if np.any(mask):
                        idx = np.where(mask)[0][0]
                        img = (X[idx].squeeze() * 255).astype(np.uint8)
                        scols[d].image(img, caption=str(d), width=48)

            y_cat = to_categorical(y, 10)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_cat, test_size=0.2, random_state=42
            )
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

            # ---------- STEP 2: CONFIGURATION ----------
            st.markdown("---")
            st.markdown("### Step 2 \u2014 Configuration")
            tab_manual, tab_bayes = st.tabs(
                [
                    "\U0001f527 Manual Tuning",
                    "\U0001f9ea Bayesian Auto-Tuning",
                ]
            )

            # ---- MANUAL TAB ----
            with tab_manual:
                st.markdown("Configure CNN hyperparameters and train.")
                c1, c2, c3 = st.columns(3)

                with c1:
                    f1 = st.slider(
                        "Conv Layer 1 Filters",
                        16,
                        128,
                        32,
                        step=16,
                        help=(
                            "Feature detectors in the first conv layer. "
                            "More filters capture more patterns but "
                            "increase computation. Start with 32."
                        ),
                    )
                    lr = st.number_input(
                        "Learning Rate",
                        min_value=1e-5,
                        max_value=2.0,
                        value=0.001,
                        format="%.5f",
                        help=(
                            "Step size for gradient descent. Typical "
                            "range: 0.0001\u20130.01. Default 0.001 is "
                            "a safe starting point."
                        ),
                    )

                with c2:
                    f2 = st.slider(
                        "Conv Layer 2 Filters",
                        32,
                        128,
                        64,
                        step=32,
                        help=(
                            "Second layer typically has 2\u00d7 the first "
                            "to capture more complex patterns."
                        ),
                    )
                    epochs = st.slider(
                        "Epochs",
                        5,
                        50,
                        10,
                        help=(
                            "Full passes through the training data. "
                            "More = better learning but risk of "
                            "overfitting."
                        ),
                    )

                with c3:
                    dense = st.slider(
                        "Dense Units",
                        32,
                        512,
                        64,
                        step=32,
                        help=(
                            "Neurons in the fully-connected layer. "
                            "Controls capacity for combining learned "
                            "features."
                        ),
                    )
                    dropout = st.slider(
                        "Dropout",
                        0.0,
                        0.8,
                        0.4,
                        help=(
                            "Fraction of neurons disabled during "
                            "training to prevent overfitting. "
                            "0.3\u20130.5 is typical."
                        ),
                    )

                if st.button(
                    "Start Training", type="primary", key="btn_manual"
                ):
                    model = training_utils.build_manual_model(
                        f1, f2, dense, dropout, lr
                    )
                    plot_ph = st.empty()

                    with st.spinner("Training in progress\u2026"):
                        hist = model.fit(
                            X_train,
                            y_train,
                            epochs=epochs,
                            batch_size=32,
                            validation_data=(X_test, y_test),
                            callbacks=[
                                training_utils.StreamlitPlotCallback(plot_ph)
                            ],
                            verbose=0,
                        )

                    os.makedirs("model", exist_ok=True)
                    model.save("./model/final_captcha_model.h5")
                    st.session_state.model = model
                    st.success(
                        "Model trained and saved to "
                        "`./model/final_captcha_model.h5`"
                    )

                    # --- Feature 6: Evaluation Dashboard ---
                    st.markdown("---")
                    st.markdown("### Step 3 \u2014 Evaluation")
                    render_evaluation(model, X_test, y_test)

            # ---- BAYESIAN TAB (Feature 11: StreamlitTuner) ----
            with tab_bayes:
                st.markdown(
                    "Bayesian Optimisation intelligently searches the "
                    "hyperparameter space using probabilistic models to "
                    "find optimal configurations faster than grid or "
                    "random search."
                )
                max_trials = st.slider(
                    "Max Trials",
                    3,
                    10,
                    5,
                    help=(
                        "Number of hyperparameter configurations to "
                        "evaluate. More trials = better search but "
                        "longer time."
                    ),
                )

                if st.button(
                    "Start Auto-Tuning", type="primary", key="btn_bayes"
                ):
                    # Feature 11: use the custom StreamlitTuner
                    status_ctr = st.container()
                    metrics_ctr = st.container()

                    tuner = training_utils.StreamlitTuner(
                        st_status_container=status_ctr,
                        st_metrics_container=metrics_ctr,
                        hypermodel=training_utils.build_tuner_model,
                        objective="val_accuracy",
                        max_trials=max_trials,
                        executions_per_trial=1,
                        directory=TUNER_DIRECTORY,
                        project_name=TUNER_PROJECT_NAME,
                    )

                    tuner.search(
                        X_train,
                        y_train,
                        epochs=5,
                        validation_data=(X_test, y_test),
                        verbose=0,
                    )

                    best_hps = tuner.get_best_hyperparameters(1)[0]

                    st.markdown("---")
                    st.markdown("#### Best Hyperparameters Found")
                    st.json(best_hps.values)

                    st.markdown("#### Training Best Model (15 epochs)")
                    model = tuner.hypermodel.build(best_hps)
                    plot_ph = st.empty()

                    hist = model.fit(
                        X_train,
                        y_train,
                        epochs=15,
                        validation_data=(X_test, y_test),
                        callbacks=[
                            training_utils.StreamlitPlotCallback(plot_ph)
                        ],
                        verbose=0,
                    )

                    os.makedirs("model", exist_ok=True)
                    model.save("./model/final_captcha_model.h5")
                    st.session_state.model = model
                    st.success("Auto-tuned model trained and saved")

                    # Feature 6: Evaluation
                    st.markdown("---")
                    st.markdown("### Evaluation")
                    render_evaluation(model, X_test, y_test)

        else:
            st.info(
                "Upload a labelled dataset (ZIP of CAPTCHA images) "
                "to unlock the Training Studio."
            )

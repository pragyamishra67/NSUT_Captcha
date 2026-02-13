"""Prediction service with latency measurement."""

import logging
from pathlib import Path
from time import perf_counter_ns
from typing import BinaryIO, Optional, Union


logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, bytes, BinaryIO]


def _normalize_image_input(image: ImageInput):
    import io

    if isinstance(image, Path):
        return str(image)
    if isinstance(image, bytes):
        return io.BytesIO(image)
    return image


def predict_from_digits(model, digit_images):
    """
    Predict from already-segmented digit images and measure latency.

    Returns:
        dict: {"prediction": str, "latency_ms": float}
    """
    from captcha_predictor.core.vision import predict_sequence

    start_ns = perf_counter_ns()
    prediction = predict_sequence(model, digit_images)
    latency_ms = (perf_counter_ns() - start_ns) / 1_000_000.0

    logger.info(
        "event=inference_complete stage=digits prediction=%s latency_ms=%.3f",
        prediction,
        latency_ms,
    )
    return {"prediction": prediction, "latency_ms": latency_ms}


def predict_from_image(image: ImageInput, model):
    """
    Predict from raw image input and measure latency for segmentation + model inference.

    Returns:
        dict: {"prediction": str, "latency_ms": float}
    """
    from captcha_predictor.core.vision import (
        preprocess_captcha_v2,
        segment_characters_robust,
    )

    normalized_image = _normalize_image_input(image)

    start_ns = perf_counter_ns()
    _, cleaned = preprocess_captcha_v2(normalized_image)
    if cleaned is None:
        prediction = "Error: Segmentation Failed"
        latency_ms = (perf_counter_ns() - start_ns) / 1_000_000.0
        logger.info(
            "event=inference_complete stage=image prediction=%s latency_ms=%.3f",
            prediction,
            latency_ms,
        )
        return {"prediction": prediction, "latency_ms": latency_ms}

    digit_images = segment_characters_robust(cleaned)
    result = predict_from_digits(model, digit_images)
    latency_ms = (perf_counter_ns() - start_ns) / 1_000_000.0

    logger.info(
        "event=inference_complete stage=image prediction=%s latency_ms=%.3f",
        result["prediction"],
        latency_ms,
    )
    return {"prediction": result["prediction"], "latency_ms": latency_ms}


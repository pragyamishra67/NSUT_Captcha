"""High-level public API for CAPTCHA prediction."""

import io
from pathlib import Path
from typing import BinaryIO, Optional, Union

from captcha_predictor.adapters.model_loader import load_pretrained_model


ImageInput = Union[str, Path, bytes, BinaryIO]


def load_model(model_path: Optional[Union[str, Path]] = None):
    """Load a trained model from disk."""
    return load_pretrained_model(model_path=model_path)


def _normalize_image_input(image: ImageInput):
    if isinstance(image, Path):
        return str(image)
    if isinstance(image, bytes):
        return io.BytesIO(image)
    return image


def _extract_digits(image: ImageInput):
    from captcha_predictor.core.vision import (
        preprocess_captcha_v2,
        segment_characters_robust,
    )

    normalized_image = _normalize_image_input(image)
    _, cleaned = preprocess_captcha_v2(normalized_image)
    if cleaned is None:
        return []
    return segment_characters_robust(cleaned)


def _predict_from_digits(model, digit_images):
    from captcha_predictor.core.vision import predict_sequence

    return predict_sequence(model, digit_images)


def segment_captcha(image: ImageInput):
    """Preprocess and segment a CAPTCHA image into digit tiles."""
    return _extract_digits(image)


def predict_captcha(
    image: ImageInput,
    model=None,
    model_path: Optional[Union[str, Path]] = None,
):
    """
    Predict a 5-digit CAPTCHA string from an image input.

    The image input may be:
    - a file path string
    - a pathlib.Path
    - raw bytes
    - a file-like stream supporting .read()
    """
    active_model = model if model is not None else load_model(model_path=model_path)
    if active_model is None:
        return "Error: Model not loaded"

    digit_images = _extract_digits(image)
    return _predict_from_digits(active_model, digit_images)


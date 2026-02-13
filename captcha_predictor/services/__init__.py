"""Service layer."""

from captcha_predictor.services.predict_service import (
    predict_from_digits,
    predict_from_image,
)

__all__ = ["predict_from_digits", "predict_from_image"]


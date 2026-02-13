import io
from typing import Optional, Tuple

from nsut_captcha.adapters.ims_captcha_client import CaptchaFetcher
from nsut_captcha.core.vision import (
    predict_sequence,
    preprocess_captcha_v2,
    segment_characters_robust,
)
from nsut_captcha.schemas.inference import CaptchaFetchResult


def fetch_live_captcha(fetcher: Optional[CaptchaFetcher] = None) -> CaptchaFetchResult:
    client = fetcher or CaptchaFetcher()
    image_bytes, error = client.fetch_single_image()
    return CaptchaFetchResult(image_bytes=image_bytes, error=error)


def preprocess_and_segment(image_bytes: bytes) -> Tuple[object, object, list]:
    original, cleaned = preprocess_captcha_v2(io.BytesIO(image_bytes))
    digits = segment_characters_robust(cleaned)
    return original, cleaned, digits


def predict_captcha(model, digit_images):
    return predict_sequence(model, digit_images)


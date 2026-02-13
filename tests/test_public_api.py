import unittest
from unittest.mock import patch

import captcha_predictor


class PublicApiTests(unittest.TestCase):
    def test_public_exports_are_callable(self):
        self.assertTrue(callable(captcha_predictor.load_model))
        self.assertTrue(callable(captcha_predictor.segment_captcha))
        self.assertTrue(callable(captcha_predictor.predict_captcha))

    @patch("captcha_predictor.api.predict._predict_from_digits", return_value="12345")
    @patch(
        "captcha_predictor.api.predict._extract_digits",
        return_value=["d1", "d2", "d3", "d4", "d5"],
    )
    def test_predict_captcha_with_explicit_model(
        self, extract_digits_mock, predict_from_digits_mock
    ):
        model = object()
        result = captcha_predictor.predict_captcha("dummy.png", model=model)

        self.assertEqual(result, "12345")
        extract_digits_mock.assert_called_once_with("dummy.png")
        predict_from_digits_mock.assert_called_once_with(
            model, ["d1", "d2", "d3", "d4", "d5"]
        )

    @patch("captcha_predictor.api.predict.load_model", return_value=None)
    def test_predict_captcha_without_available_model(self, load_model_mock):
        result = captcha_predictor.predict_captcha("dummy.png")
        self.assertEqual(result, "Error: Model not loaded")
        load_model_mock.assert_called_once_with(model_path=None)

    @patch(
        "captcha_predictor.api.predict._extract_digits",
        return_value=["d1", "d2", "d3", "d4", "d5"],
    )
    def test_segment_captcha_uses_segmentation_pipeline(self, extract_digits_mock):
        result = captcha_predictor.segment_captcha(b"raw-bytes")
        self.assertEqual(result, ["d1", "d2", "d3", "d4", "d5"])
        extract_digits_mock.assert_called_once_with(b"raw-bytes")


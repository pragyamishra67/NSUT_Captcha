# Usage

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from captcha_predictor import load_model, predict_captcha

model = load_model("model/final_captcha_model.h5")
result = predict_captcha("captchas/example.png", model=model)
print(result)
```

## API Surface

- `load_model(model_path: str | Path | None = None)`
  Loads a TensorFlow model. If `model_path` is omitted, the default configured path is used.
- `segment_captcha(image)`
  Runs preprocessing + segmentation and returns digit tiles.
- `predict_captcha(image, model=None, model_path=None)`
  Predicts the final 5-digit CAPTCHA string from image input.

Supported image input types:
- file path (`str`)
- `pathlib.Path`
- raw `bytes`
- file-like stream (`.read()`)

## Optional HTTP API

This package does not ship a built-in REST server.  
If needed, wrap `predict_captcha()` inside your own FastAPI/Flask endpoint.

## Configuration

Default constants are in:

- `captcha_predictor/config/settings.py`

Key paths/constants:

- `MODEL_PATH`
- `TEMP_DATASET_DIR`
- `IMS_BASE_URL`, `IMS_LOGIN_URL`

## Troubleshooting

- `Error: Model not loaded`
  Verify the model file exists and path is correct.
- `Error: Segmentation Failed`
  The image did not segment into 5 digit regions; verify image quality/format.
- Import errors for OpenCV/TensorFlow
  Install package dependencies and ensure your Python environment is active.


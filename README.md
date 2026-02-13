# captcha_predictor

Installable Python package for predicting NSUT-style 5-digit CAPTCHA images.

## Install

```bash
pip install -e .
```

## Public API

```python
from captcha_predictor import load_model, segment_captcha, predict_captcha

model = load_model('model/final_captcha_model.h5')
segments = segment_captcha('captchas/example.png')
value = predict_captcha('captchas/example.png', model=model)
```

## Project Layout

- `captcha_predictor/` package source
- `tests/` basic package-level tests
- `docs/USAGE.md` integration instructions
- `pyproject.toml` packaging metadata

For full usage details, see `docs/USAGE.md`.

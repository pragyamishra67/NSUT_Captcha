# CAPTCHArd - NSUT CAPTCHA Solver

A comprehensive computer vision solution for predicting NSUT-style 5-digit CAPTCHA images. This project combines robust image preprocessing, character segmentation, and CNN-based classification to achieve high-accuracy CAPTCHA solving.

## Features

- **Live CAPTCHA Solving**: Fetch and solve CAPTCHAs from the NSUT IMS portal in real-time
- **Installable Python Package**: Use as a library in your own projects via `captcha_predictor`
- **Streamlit Web Interface**: User-friendly UI for inference and model training
- **Custom Model Training**: Train your own models with manual hyperparameter tuning or Bayesian optimization
- **Flexible Input Support**: Accept images as file paths, bytes, or file-like streams
- **Async API Support**: Async endpoints for integration with web frameworks

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Public API Reference](#public-api-reference)
- [Web Interface](#web-interface)
- [Training Your Own Model](#training-your-own-model)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)
- [License](#license)

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Install as Package (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-repo/NSUT_Captcha.git
cd NSUT_Captcha

# Install in development mode
pip install -e .
```

### Install Dependencies Only

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Basic Prediction (Python API)

```python
from captcha_predictor import load_model, predict_captcha, segment_captcha

# Load the pre-trained model
model = load_model("model/final_captcha_model.h5")

# Predict CAPTCHA from an image file
result = predict_captcha("captchas/example.png", model=model)
print(f"Predicted CAPTCHA: {result}")  # e.g., "12345"

# Or let it auto-load the default model
result = predict_captcha("captchas/example.png")
```

### 2. Process Different Input Types

```python
from captcha_predictor import predict_captcha, segment_captcha
from pathlib import Path

# From file path string
result = predict_captcha("captchas/sample.png")

# From pathlib.Path
result = predict_captcha(Path("captchas/sample.png"))

# From raw bytes
with open("captchas/sample.png", "rb") as f:
    image_bytes = f.read()
result = predict_captcha(image_bytes)

# From file-like stream
with open("captchas/sample.png", "rb") as f:
    result = predict_captcha(f)
```

### 3. Segment CAPTCHA into Digits

```python
from captcha_predictor import segment_captcha

# Get individual digit images (32x32 numpy arrays)
digits = segment_captcha("captchas/example.png")
print(f"Found {len(digits)} digit segments")
```

### 4. Launch Web Interface

```bash
# Run the Streamlit app
streamlit run app.py

# Or use the module directly
python app.py
```

---

## Project Structure

```
NSUT_Captcha/
├── captcha_predictor/          # Main installable package
│   ├── __init__.py             # Public API exports
│   ├── api/
│   │   └── predict.py          # High-level prediction functions
│   ├── adapters/
│   │   ├── ims_captcha_client.py   # Fetches CAPTCHAs from IMS portal
│   │   └── model_loader.py     # Model loading utilities
│   ├── config/
│   │   └── settings.py         # Centralized configuration
│   ├── core/
│   │   └── vision.py           # Image preprocessing & segmentation
│   ├── schemas/
│   │   └── inference.py        # Data schemas/types
│   ├── services/
│   │   ├── dataset_service.py  # Dataset loading/processing
│   │   ├── inference_service.py# Inference pipeline
│   │   ├── predict_service.py  # Prediction service layer
│   │   └── training_utils.py   # Model builders & training callbacks
│   └── utils/
│       └── logging.py          # Logging configuration
│
├── nsut_captcha/               # Streamlit app package
│   ├── app/
│   │   └── main.py             # App entry point
│   ├── ui/
│   │   └── streamlit_app.py    # Streamlit interface
│   ├── adapters/               # Shared adapters
│   ├── core/                   # Core vision functions
│   ├── services/               # Training & dataset services
│   └── config/                 # App configuration
│
├── model/
│   └── final_captcha_model.h5  # Pre-trained model weights
│
├── assets/
│   └── style.css               # UI styling
│
├── tests/
│   └── test_public_api.py      # Package tests
│
├── captchas/                   # Sample CAPTCHA images
├── Preprocessed/               # Preprocessed training data
├── temp_dataset/               # Temporary dataset storage
│
├── app.py                      # Main application entry
├── model_train.py              # Standalone training script
├── preprocess.py               # Image preprocessing utilities
├── backend.py                  # Backend imports
├── pyproject.toml              # Package metadata
├── requirements.txt            # Dependencies
└── docs/
    └── USAGE.md                # Detailed usage guide
```

---

## Public API Reference

### Core Functions

#### `load_model(model_path=None)`
Load a trained TensorFlow model.

```python
from captcha_predictor import load_model

# Load from specific path
model = load_model("model/final_captcha_model.h5")

# Load from default configured path
model = load_model()
```

**Parameters:**
- `model_path` (str | Path | None): Path to `.h5` model file. Uses default if omitted.

**Returns:** Loaded Keras model instance

---

#### `predict_captcha(image, model=None, model_path=None)`
Predict the 5-digit CAPTCHA string from an image.

```python
from captcha_predictor import predict_captcha

# With pre-loaded model
result = predict_captcha("image.png", model=model)

# Auto-load default model
result = predict_captcha("image.png")

# Specify custom model path
result = predict_captcha("image.png", model_path="custom_model.h5")
```

**Parameters:**
- `image`: File path (str), Path object, raw bytes, or file-like stream
- `model`: Pre-loaded model (optional)
- `model_path`: Path to model file (optional)

**Returns:** String of predicted 5 digits (e.g., `"48291"`) or error message

---

#### `segment_captcha(image)`
Preprocess and segment a CAPTCHA into individual digit tiles.

```python
from captcha_predictor import segment_captcha

digits = segment_captcha("captcha.png")
# Returns list of 5 numpy arrays (32x32 each)
```

**Parameters:**
- `image`: File path, Path object, raw bytes, or file-like stream

**Returns:** List of digit images as numpy arrays

---

### Async API (for Web Frameworks)

```python
from captcha_predictor.api.predict import predict_captcha_endpoint

# Returns dict with prediction and latency
result = await predict_captcha_endpoint("image.png", model=model)
# {"prediction": "12345", "latency_ms": 15.3}
```

---

## Web Interface

The Streamlit web interface provides two modes:

### Live Inference Mode
- Fetch real CAPTCHAs from the NSUT IMS portal
- View original and preprocessed images
- See segmented digits
- Get instant predictions with latency metrics

### Training Studio
- Upload custom dataset (ZIP of labeled CAPTCHA images)
- Manual model design with customizable:
  - Convolutional filters
  - Dense units
  - Dropout rate
  - Learning rate
  - Number of epochs
- Bayesian hyperparameter optimization with real-time visualization
- Training progress charts and accuracy metrics
- Download trained models

**Launch the interface:**
```bash
streamlit run app.py
```

---

## Training Your Own Model

### Using the Training Studio (Recommended)

1. Prepare a ZIP file with CAPTCHA images named as their labels (e.g., `12345.png`)
2. Launch the Streamlit app: `streamlit run app.py`
3. Select "Training Studio" mode
4. Upload your dataset ZIP
5. Choose manual training or Bayesian optimization
6. Configure hyperparameters and train
7. Download the trained model

### Using the Standalone Script

```python
# Edit model_train.py configuration
DATA_DIR = "path/to/your/preprocessed/captchas"
EPOCHS = 20
BATCH_SIZE = 32

# Run training
python model_train.py
```

### Dataset Format

- Images should be PNG/JPG files
- Filename = CAPTCHA label (e.g., `48291.png` for CAPTCHA showing "48291")
- Only 5-digit numeric CAPTCHAs are supported

---

## Architecture

### Image Processing Pipeline

1. **Load Image**: Grayscale conversion
2. **Binarization**: Otsu's threshold with inverse binary
3. **Noise Removal**: Morphological opening
4. **Segmentation**: Contour detection and bounding box extraction
5. **Digit Isolation**: Smart splitting for merged digits
6. **Normalization**: Resize to 32x32 with padding

### Model Architecture

The CNN classifier for individual digits:

```
Conv2D(filters, 3x3, ReLU) → BatchNorm → MaxPool
Conv2D(filters, 3x3, ReLU) → BatchNorm → MaxPool
Flatten → Dense(units, ReLU) → Dropout → Dense(10, Softmax)
```

For full CAPTCHA recognition (CTC-based):

```
Conv2D → MaxPool → Conv2D → MaxPool → Reshape
Dense → Bidirectional LSTM → Dense(Softmax) + CTC Loss
```

---

## Configuration

Edit `captcha_predictor/config/settings.py`:

```python
# Paths
MODEL_PATH = PROJECT_ROOT / "model" / "final_captcha_model.h5"
TEMP_DATASET_DIR = PROJECT_ROOT / "temp_dataset"

# IMS Portal URLs (for live CAPTCHA fetching)
IMS_BASE_URL = "https://imsnsit.org/imsnsit/"
IMS_LOGIN_URL = "https://www.imsnsit.org/imsnsit/student_login.php"

# Tuner Settings
TUNER_DIRECTORY = "my_dir"
TUNER_PROJECT_NAME = "captcha_tuning_web"
```

---

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_public_api.py
```

---

## Troubleshooting

### `Error: Model not loaded`
- Verify the model file exists at the specified path
- Check that the path in settings.py is correct
- Ensure TensorFlow is properly installed

### `Error: Segmentation Failed`
- The image didn't segment into exactly 5 digit regions
- Check image quality and format
- Try preprocessing the image manually first

### Import errors (OpenCV/TensorFlow)
```bash
# Reinstall dependencies
pip install --force-reinstall tensorflow opencv-python-headless
```

### CAPTCHA fetch fails
- Check internet connection
- The IMS portal may be temporarily unavailable
- Session cookies may have expired (retry)

---

## Dependencies

| Package | Purpose |
|---------|---------|
| tensorflow | Deep learning framework |
| opencv-python-headless | Image processing |
| numpy | Numerical operations |
| streamlit | Web interface |
| keras-tuner | Hyperparameter optimization |
| beautifulsoup4 | HTML parsing for CAPTCHA fetching |
| requests | HTTP client |
| scikit-learn | Data splitting, metrics |
| matplotlib, seaborn, plotly | Visualization |

Install all:
```bash
pip install -r requirements.txt
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Acknowledgments

- TensorFlow and Keras teams for the deep learning framework
- OpenCV for computer vision utilities
- Streamlit for the interactive web framework
- NSUT for the CAPTCHA challenge inspiration

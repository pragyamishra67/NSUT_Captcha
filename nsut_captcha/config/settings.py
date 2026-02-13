"""Centralized project settings."""

from pathlib import Path


PROJECT_ROOT = Path(".")
ASSETS_DIR = PROJECT_ROOT / "assets"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_FILE_NAME = "final_captcha_model.h5"
MODEL_PATH = MODEL_DIR / MODEL_FILE_NAME

TEMP_DATASET_DIR = PROJECT_ROOT / "temp_dataset"
TUNER_DIRECTORY = "my_dir"
TUNER_PROJECT_NAME = "captcha_tuning_web"

IMS_BASE_URL = "https://imsnsit.org/imsnsit/"
IMS_LEGACY_LOGIN_URL = "https://www.imsnsit.org/imsnsit/student_login110.php"
IMS_LOGIN_URL = "https://www.imsnsit.org/imsnsit/student_login.php"

DEFAULT_REQUEST_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"


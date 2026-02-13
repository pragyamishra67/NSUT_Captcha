import os
import zipfile

import numpy as np

from nsut_captcha.config.settings import TEMP_DATASET_DIR
from nsut_captcha.core.vision import (
    preprocess_captcha_v2,
    segment_characters_robust,
)


def load_uploaded_dataset(uploaded_file):
    if TEMP_DATASET_DIR.exists():
        for root, dirs, files in os.walk(str(TEMP_DATASET_DIR), topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                os.rmdir(os.path.join(root, d))

    with zipfile.ZipFile(uploaded_file, "r") as z:
        z.extractall(str(TEMP_DATASET_DIR))

    data, labels = [], []
    valid_count = 0

    for root, _, files in os.walk(str(TEMP_DATASET_DIR)):
        for f in files:
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            label_text = os.path.splitext(f)[0]
            if len(label_text) != 5 or not label_text.isdigit():
                continue

            path = os.path.join(root, f)
            _, cleaned = preprocess_captcha_v2(path)
            digits = segment_characters_robust(cleaned)

            if len(digits) == 5:
                valid_count += 1
                for i, d in enumerate(digits):
                    img = d / 255.0
                    img = np.expand_dims(img, axis=-1)
                    data.append(img)
                    labels.append(int(label_text[i]))

    return np.array(data), np.array(labels), valid_count


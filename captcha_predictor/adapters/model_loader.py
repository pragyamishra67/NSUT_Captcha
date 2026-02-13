from pathlib import Path
from typing import Optional, Union

from captcha_predictor.config.settings import MODEL_PATH


def load_pretrained_model(model_path: Optional[Union[str, Path]] = None):
    """Load a TensorFlow model from disk without import-time side effects."""
    import tensorflow as tf

    resolved_path = Path(model_path) if model_path else MODEL_PATH
    if resolved_path.exists():
        try:
            model = tf.keras.models.load_model(str(resolved_path))
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}") from e
    return None


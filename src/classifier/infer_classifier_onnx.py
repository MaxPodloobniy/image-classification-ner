"""
Image classification inference using ONNX Runtime (production path).

Uses PIL + numpy for preprocessing — no TensorFlow required at serve time.
Public API mirrors src.classifier.infer_classifier so callers can swap
backends transparently.
"""
import json
import logging
import os

import numpy as np
import onnxruntime as ort
from PIL import Image

logger = logging.getLogger(__name__)


def preprocess_image(image_path: str, target_size: tuple[int, int]) -> np.ndarray:
    """Load and preprocess image for ONNX inference.

    Args:
        image_path: Path to image file.
        target_size: (height, width) tuple matching the model's expected input.

    Returns:
        Float32 array of shape (1, H, W, 3) normalised to [0, 1].
    """
    img = Image.open(image_path).convert("RGB").resize((target_size[1], target_size[0]))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def classify_image(
    image_path: str,
    model_path: str,
    session: ort.InferenceSession | None = None,
) -> tuple[str, float]:
    """Classify an image and return the predicted class and confidence.

    Args:
        image_path: Path to the image file.
        model_path: Path to the .onnx model file.
        session: Optional pre-loaded InferenceSession (for reuse in web app).
                 When provided, the model-path existence check is skipped.

    Returns:
        Tuple of (predicted_class_name, confidence_score).

    Raises:
        FileNotFoundError: If image, model, or classes JSON is missing.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if session is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

    classes_path = model_path.replace(".onnx", "_classes.json")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Class names file not found: {classes_path}")

    if session is None:
        session = ort.InferenceSession(model_path)
        logger.info(f"Loaded ONNX session from {model_path}")

    with open(classes_path) as f:
        class_indices = json.load(f)

    # Derive target size from model metadata — avoids hard-coding dimensions
    input_meta = session.get_inputs()[0]
    _, h, w, _ = input_meta.shape  # NHWC layout

    img_array = preprocess_image(image_path, (h, w))
    predictions = session.run(None, {input_meta.name: img_array})[0]

    # Value-based mapping avoids the key-insertion-order pitfall
    idx_to_class = {v: k for k, v in class_indices.items()}
    predicted_class = idx_to_class[int(np.argmax(predictions))]
    confidence = float(np.max(predictions))

    logger.info(f"Predicted: {predicted_class} ({confidence:.2%})")
    return predicted_class, confidence

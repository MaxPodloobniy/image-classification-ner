"""Unit tests for src.classifier.infer_classifier."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.classifier.infer_classifier import classify_image, preprocess_image


# ─── classify_image: error paths ──────────────────────────────────────────────

def test_classify_image_raises_if_image_missing(classes_json, tmp_path):
    model_path, _ = classes_json
    missing_image = tmp_path / "does_not_exist.png"
    with pytest.raises(FileNotFoundError, match="Image not found"):
        classify_image(str(missing_image), model_path)


def test_classify_image_raises_if_model_missing(tiny_image, tmp_path):
    missing_model = tmp_path / "nope.keras"
    with pytest.raises(FileNotFoundError, match="Model not found"):
        classify_image(tiny_image, str(missing_model))


def test_classify_image_raises_if_classes_json_missing(tiny_image, tmp_path):
    # .keras file exists but the matching *_classes.json does not
    model_path = tmp_path / "solo.keras"
    model_path.write_bytes(b"fake")
    with pytest.raises(FileNotFoundError, match="Class names file not found"):
        classify_image(tiny_image, str(model_path))


# ─── classify_image: happy path with mocked model ─────────────────────────────

def test_classify_image_returns_class_and_confidence(mocker, tiny_image, classes_json):
    model_path, _ = classes_json

    # Stub the Keras model: input_shape = (None, 10, 10, 3), predict returns
    # a softmax-like vector where class index 0 (tiger) wins with ~0.9
    fake_model = MagicMock()
    fake_model.input_shape = (None, 10, 10, 3)
    fake_model.predict.return_value = np.array([[0.9, 0.07, 0.03]])
    mocker.patch("src.classifier.infer_classifier.load_model", return_value=fake_model)

    predicted, confidence = classify_image(tiny_image, model_path)

    assert predicted == "tiger"
    assert confidence == pytest.approx(0.9, abs=1e-6)
    fake_model.predict.assert_called_once()


# ─── preprocess_image: shape + normalization ──────────────────────────────────

def test_preprocess_image_shape_is_batch_hwc(tiny_image):
    arr = preprocess_image(tiny_image, target_size=(10, 10))
    # Expected: (batch=1, H=10, W=10, C=3)
    assert arr.shape == (1, 10, 10, 3)


def test_preprocess_image_normalizes_to_unit_range(tiny_image):
    arr = preprocess_image(tiny_image, target_size=(10, 10))
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0

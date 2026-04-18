"""Unit tests for src.classifier.infer_classifier_onnx."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.classifier.infer_classifier_onnx import classify_image, preprocess_image


def _make_session(h: int = 10, w: int = 10, predictions=None):
    """Build a minimal ort.InferenceSession mock."""
    if predictions is None:
        predictions = np.array([[0.9, 0.07, 0.03]], dtype=np.float32)

    input_meta = MagicMock()
    input_meta.name = "input"
    input_meta.shape = [1, h, w, 3]

    session = MagicMock()
    session.get_inputs.return_value = [input_meta]
    session.run.return_value = [predictions]
    return session


# ─── classify_image: error paths ──────────────────────────────────────────────

def test_classify_image_raises_if_image_missing(classes_json, tmp_path):
    model_path, _ = classes_json
    # replace .keras extension that conftest used → .onnx
    onnx_path = model_path.replace(".keras", ".onnx")
    missing_image = tmp_path / "nope.png"
    with pytest.raises(FileNotFoundError, match="Image not found"):
        classify_image(str(missing_image), onnx_path)


def test_classify_image_raises_if_model_missing(tiny_image, tmp_path):
    missing = tmp_path / "nope.onnx"
    with pytest.raises(FileNotFoundError, match="Model not found"):
        classify_image(tiny_image, str(missing))


def test_classify_image_raises_if_classes_json_missing(tiny_image, tmp_path):
    model_path = tmp_path / "solo.onnx"
    model_path.write_bytes(b"fake")
    with pytest.raises(FileNotFoundError, match="Class names file not found"):
        classify_image(tiny_image, str(model_path))


# ─── classify_image: happy paths ──────────────────────────────────────────────

def test_classify_image_returns_first_class(mocker, tiny_image, classes_json):
    """Index 0 winning → 'tiger'."""
    model_path, _ = classes_json
    onnx_path = model_path.replace(".keras", ".onnx")
    # classes_json writes {"tiger":0,"deer":1,"eagle":2} next to model_path;
    # we need the same JSON next to onnx_path.
    import json
    import pathlib
    pathlib.Path(onnx_path.replace(".onnx", "_classes.json")).write_text(
        json.dumps({"tiger": 0, "deer": 1, "eagle": 2})
    )
    pathlib.Path(onnx_path).write_bytes(b"fake")

    session = _make_session(predictions=np.array([[0.9, 0.07, 0.03]], dtype=np.float32))
    mocker.patch("src.classifier.infer_classifier_onnx.ort.InferenceSession", return_value=session)

    predicted, confidence = classify_image(tiny_image, onnx_path)

    assert predicted == "tiger"
    assert confidence == pytest.approx(0.9, abs=1e-5)


def test_classify_image_picks_correct_class_when_argmax_is_not_first(mocker, tiny_image, classes_json):
    """Index 2 winning → 'eagle'; ensures argmax→name mapping is correct."""
    model_path, _ = classes_json
    onnx_path = model_path.replace(".keras", ".onnx")
    import json
    import pathlib
    pathlib.Path(onnx_path.replace(".onnx", "_classes.json")).write_text(
        json.dumps({"tiger": 0, "deer": 1, "eagle": 2})
    )
    pathlib.Path(onnx_path).write_bytes(b"fake")

    session = _make_session(predictions=np.array([[0.05, 0.1, 0.85]], dtype=np.float32))
    mocker.patch("src.classifier.infer_classifier_onnx.ort.InferenceSession", return_value=session)

    predicted, confidence = classify_image(tiny_image, onnx_path)

    assert predicted == "eagle"
    assert confidence == pytest.approx(0.85, abs=1e-5)


def test_classify_image_uses_model_input_shape_for_preprocessing(mocker, tiny_image, classes_json):
    """input_meta.shape (N,H,W,C) is used to determine preprocessing target size."""
    model_path, _ = classes_json
    onnx_path = model_path.replace(".keras", ".onnx")
    import json
    import pathlib
    pathlib.Path(onnx_path.replace(".onnx", "_classes.json")).write_text(
        json.dumps({"tiger": 0, "deer": 1, "eagle": 2})
    )
    pathlib.Path(onnx_path).write_bytes(b"fake")

    session = _make_session(h=32, w=32)
    mocker.patch("src.classifier.infer_classifier_onnx.ort.InferenceSession", return_value=session)

    mock_pre = mocker.patch(
        "src.classifier.infer_classifier_onnx.preprocess_image",
        return_value=np.zeros((1, 32, 32, 3), dtype=np.float32),
    )

    classify_image(tiny_image, onnx_path)
    mock_pre.assert_called_once_with(tiny_image, (32, 32))


def test_classify_image_reuses_provided_session(mocker, tiny_image, classes_json):
    """When session is passed, ort.InferenceSession is NOT called again."""
    model_path, _ = classes_json
    onnx_path = model_path.replace(".keras", ".onnx")
    import json
    import pathlib
    pathlib.Path(onnx_path.replace(".onnx", "_classes.json")).write_text(
        json.dumps({"tiger": 0, "deer": 1, "eagle": 2})
    )

    session = _make_session()
    mock_ort = mocker.patch("src.classifier.infer_classifier_onnx.ort.InferenceSession")

    classify_image(tiny_image, onnx_path, session=session)

    mock_ort.assert_not_called()


# ─── preprocess_image: shape + normalization ──────────────────────────────────

def test_preprocess_image_shape_is_batch_hwc(tiny_image):
    arr = preprocess_image(tiny_image, target_size=(10, 10))
    assert arr.shape == (1, 10, 10, 3)


def test_preprocess_image_normalizes_to_unit_range(tiny_image):
    arr = preprocess_image(tiny_image, target_size=(10, 10))
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


def test_preprocess_image_returns_float32(tiny_image):
    arr = preprocess_image(tiny_image, target_size=(10, 10))
    assert arr.dtype == np.float32

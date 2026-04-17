"""Shared pytest fixtures."""
import json

import pytest
from PIL import Image


@pytest.fixture
def tiny_image(tmp_path):
    """Create a 10x10 RGB PNG on disk and return its path."""
    path = tmp_path / "tiny.png"
    Image.new("RGB", (10, 10), color=(128, 64, 32)).save(path)
    return str(path)


@pytest.fixture
def classes_json(tmp_path):
    """Write a classes JSON and return (keras_path, classes_path).

    The classifier derives the classes path from the model path by swapping
    '.keras' → '_classes.json', so we mirror that convention here.
    """
    model_path = tmp_path / "model.keras"
    model_path.write_bytes(b"fake-keras-bytes")
    classes_path = tmp_path / "model_classes.json"
    classes_path.write_text(json.dumps({"tiger": 0, "deer": 1, "eagle": 2}))
    return str(model_path), str(classes_path)


@pytest.fixture
def ner_dataset_file(tmp_path):
    """Write a small NER dataset matching data/ner/train.json structure."""
    path = tmp_path / "ner_sample.json"
    payload = [
        {"text": "I saw a tiger today.", "entities": [[8, 13, "ANIMAL"]]},
        {"text": "The deer ran fast.", "entities": [[4, 8, "ANIMAL"]]},
        {"text": "No animals here.", "entities": []},
    ]
    path.write_text(json.dumps(payload))
    return str(path)

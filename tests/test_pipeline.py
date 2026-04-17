"""Unit tests for src.pipeline."""
from argparse import Namespace

import pytest

from src.pipeline import main

_ARGS = Namespace(
    image_path="img.png",
    text="some text",
    ner_model_path="m/ner",
    class_model_path="m/cls.keras",
)


def _patch(mocker, predicted, animals):
    mocker.patch("src.pipeline.classify_image", return_value=(predicted, 0.95))
    mocker.patch("src.pipeline.extract_animals", return_value=animals)
    mocker.patch("src.pipeline.parse_args", return_value=_ARGS)


@pytest.mark.parametrize(
    "predicted, animals, expected_code",
    [
        ("tiger", ["tiger"], 0),                     # exact match → TRUE
        ("eagle", ["tiger"], 1),                     # no match → FALSE
        ("Tiger", ["tiger"], 0),                     # case-insensitive → TRUE
        ("deer", [], 1),                             # empty NER → FALSE
        ("tiger", ["Siberian tiger"], 1),            # substring ≠ list item → FALSE
    ],
)
def test_pipeline_verdict(mocker, predicted, animals, expected_code):
    _patch(mocker, predicted, animals)
    assert main() == expected_code

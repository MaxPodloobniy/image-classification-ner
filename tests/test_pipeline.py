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
    mock_classify = mocker.patch("src.pipeline.classify_image", return_value=(predicted, 0.95))
    mock_extract = mocker.patch("src.pipeline.extract_animals", return_value=animals)
    mocker.patch("src.pipeline.parse_args", return_value=_ARGS)
    return mock_classify, mock_extract


@pytest.mark.parametrize(
    "predicted, animals, expected_code",
    [
        ("tiger", ["tiger"], 0),                # exact match → TRUE
        ("eagle", ["tiger"], 1),                # no match → FALSE
        ("Tiger", ["tiger"], 0),                # case-insensitive → TRUE
        ("deer", [], 1),                        # empty NER → FALSE
        ("tiger", ["Siberian tiger"], 1),       # substring ≠ list item → FALSE
    ],
)
def test_pipeline_verdict(mocker, predicted, animals, expected_code):
    mock_classify, mock_extract = _patch(mocker, predicted, animals)

    assert main() == expected_code

    # assert args are wired correctly — catches swapped argument regressions
    mock_classify.assert_called_once_with(_ARGS.image_path, _ARGS.class_model_path)
    mock_extract.assert_called_once_with(_ARGS.text, _ARGS.ner_model_path)

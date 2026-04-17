"""Unit tests for src.pipeline."""
import pytest

from src.pipeline import main


def _patch(mocker, predicted, animals, args_extra=None):
    """Helper: patch classify_image, extract_animals, and parse_args."""
    mocker.patch("src.pipeline.classify_image", return_value=(predicted, 0.95))
    mocker.patch("src.pipeline.extract_animals", return_value=animals)
    ns = {
        "image_path": "img.png",
        "text": "some text",
        "ner_model_path": "m/ner",
        "class_model_path": "m/cls.keras",
    }
    if args_extra:
        ns.update(args_extra)

    from argparse import Namespace
    mocker.patch("src.pipeline.parse_args", return_value=Namespace(**ns))


def test_pipeline_returns_0_when_animal_matches(mocker):
    _patch(mocker, "tiger", ["tiger"])
    assert main() == 0


def test_pipeline_returns_1_when_no_match(mocker):
    _patch(mocker, "eagle", ["tiger"])
    assert main() == 1


def test_pipeline_is_case_insensitive(mocker):
    _patch(mocker, "Tiger", ["tiger"])
    assert main() == 0


def test_pipeline_returns_1_when_ner_extracts_nothing(mocker):
    _patch(mocker, "deer", [])
    assert main() == 1

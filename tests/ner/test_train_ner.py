"""Unit tests for src.ner.train_ner."""
from unittest.mock import MagicMock

import pytest

from src.ner.train_ner import load_data, load_or_download_model


# ─── load_data ────────────────────────────────────────────────────────────────

def test_load_data_returns_tuples_with_entities(ner_dataset_file):
    data = load_data(ner_dataset_file)
    assert len(data) == 3
    for text, annotation in data:
        assert isinstance(text, str)
        assert "entities" in annotation


def test_load_data_preserves_entity_offsets_and_label(ner_dataset_file):
    data = load_data(ner_dataset_file)
    # First sample: "I saw a tiger today." entities [[8, 13, "ANIMAL"]]
    text, annotation = data[0]
    assert text == "I saw a tiger today."
    assert annotation["entities"] == [(8, 13, "ANIMAL")]


def test_load_data_handles_empty_entities(ner_dataset_file):
    data = load_data(ner_dataset_file)
    # Third sample has no entities
    _, annotation = data[2]
    assert annotation["entities"] == []


# ─── load_or_download_model ───────────────────────────────────────────────────

def test_load_or_download_model_returns_nlp_on_success(mocker):
    fake_nlp = MagicMock()
    mocker.patch("src.ner.train_ner.spacy.load", return_value=fake_nlp)
    result = load_or_download_model("en_core_web_md")
    assert result is fake_nlp


def test_load_or_download_model_raises_runtime_error_on_os_error(mocker):
    mocker.patch("src.ner.train_ner.spacy.load", side_effect=OSError("not found"))
    with pytest.raises(RuntimeError, match="missing"):
        load_or_download_model("en_core_web_md")

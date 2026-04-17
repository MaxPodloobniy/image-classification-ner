"""Unit tests for src.ner.infer_ner."""
from unittest.mock import MagicMock

import pytest

from src.ner.infer_ner import extract_animals


# ─── error paths ──────────────────────────────────────────────────────────────

def test_extract_animals_raises_if_model_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model not found"):
        extract_animals("I saw a tiger.", str(tmp_path / "missing_model"))


@pytest.mark.parametrize("bad_input", ["", "   ", 42, None, 3.14])
def test_extract_animals_raises_on_invalid_text(bad_input, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(ValueError, match="non-empty string"):
        extract_animals(bad_input, str(model_dir))


# ─── happy path with mocked spaCy model ───────────────────────────────────────

def _make_ent(text, label):
    ent = MagicMock()
    ent.text = text
    ent.label_ = label
    return ent


def test_extract_animals_filters_only_animal_label(mocker, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_path = str(model_dir)
    text = "Tiger in London, a Deer nearby."

    fake_doc = MagicMock()
    fake_doc.ents = [
        _make_ent("Tiger", "ANIMAL"),
        _make_ent("London", "GPE"),
        _make_ent("Deer", "ANIMAL"),
    ]
    fake_nlp = MagicMock(return_value=fake_doc)
    mock_load = mocker.patch("src.ner.infer_ner.spacy.load", return_value=fake_nlp)

    result = extract_animals(text, model_path)

    assert result == ["tiger", "deer"]
    mock_load.assert_called_once_with(model_path)
    fake_nlp.assert_called_once_with(text)


def test_extract_animals_returns_empty_list_when_no_entities(mocker, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    fake_doc = MagicMock()
    fake_doc.ents = []
    fake_nlp = MagicMock(return_value=fake_doc)
    mocker.patch("src.ner.infer_ner.spacy.load", return_value=fake_nlp)

    result = extract_animals("No animals here.", str(model_dir))
    assert result == []


def test_extract_animals_lowercases_output(mocker, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    fake_doc = MagicMock()
    fake_doc.ents = [_make_ent("EAGLE", "ANIMAL")]
    fake_nlp = MagicMock(return_value=fake_doc)
    mocker.patch("src.ner.infer_ner.spacy.load", return_value=fake_nlp)

    result = extract_animals("EAGLE spotted.", str(model_dir))
    assert result == ["eagle"]

"""Unit tests for src.web.app (FastAPI endpoints)."""
import io
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.web.app import app


@pytest.fixture()
def client(mocker):
    """TestClient with models stubbed out so no real files are needed."""
    # Patch lifespan model loading
    mocker.patch("src.web.app.ort.InferenceSession", return_value=MagicMock())
    mocker.patch("src.web.app.spacy.load", return_value=MagicMock())
    with TestClient(app) as c:
        yield c


def _jpeg_bytes():
    """Return a minimal valid JPEG byte payload."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(100, 150, 200)).save(buf, format="JPEG")
    return buf.getvalue()


# ── GET / ─────────────────────────────────────────────────────

def test_index_serves_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]


# ── POST /verify ──────────────────────────────────────────────

def test_verify_returns_true_on_match(client, mocker):
    mocker.patch("src.web.app.classify_image", return_value=("tiger", 0.95))
    mocker.patch("src.web.app.extract_from_nlp", return_value=["tiger"])

    res = client.post(
        "/verify",
        data={"text": "A tiger is here."},
        files={"image": ("test.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["verdict"] is True
    assert body["predicted_class"] == "tiger"
    assert "tiger" in body["extracted_animals"]


def test_verify_returns_false_on_mismatch(client, mocker):
    mocker.patch("src.web.app.classify_image", return_value=("eagle", 0.88))
    mocker.patch("src.web.app.extract_from_nlp", return_value=["tiger"])

    res = client.post(
        "/verify",
        data={"text": "I see a tiger."},
        files={"image": ("test.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert res.status_code == 200
    assert res.json()["verdict"] is False


def test_verify_rejects_non_image_file(client):
    res = client.post(
        "/verify",
        data={"text": "some text"},
        files={"image": ("file.txt", b"hello", "text/plain")},
    )
    assert res.status_code == 400


def test_verify_requires_text_field(client):
    """Missing 'text' form field → 422 Unprocessable Entity."""
    res = client.post(
        "/verify",
        files={"image": ("test.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert res.status_code == 422

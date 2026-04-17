"""Unit tests for src.classifier.web_scraper."""
import requests
import pytest

from src.classifier.web_scraper import download_images, get_high_res_image

_OG_HTML = """
<html><head>
  <meta property="og:image" content="https://cdn.example.com/og.jpg"/>
</head><body></body></html>
"""

_IMG_CLASS_HTML = """
<html><body>
  <img class="hCL kVc L4E MIw" src="https://cdn.example.com/fallback.jpg"/>
</body></html>
"""

_EMPTY_HTML = "<html><body></body></html>"


def _mock_response(text, status=200):
    from unittest.mock import MagicMock
    r = MagicMock()
    r.text = text
    r.status_code = status
    r.raise_for_status = MagicMock()
    return r


# ─── get_high_res_image ────────────────────────────────────────────────────────

def test_get_high_res_image_returns_og_image_url(mocker):
    mocker.patch("src.classifier.web_scraper.requests.get", return_value=_mock_response(_OG_HTML))
    assert get_high_res_image("https://pin.it/1") == "https://cdn.example.com/og.jpg"


def test_get_high_res_image_falls_back_to_img_class(mocker):
    mocker.patch("src.classifier.web_scraper.requests.get", return_value=_mock_response(_IMG_CLASS_HTML))
    assert get_high_res_image("https://pin.it/2") == "https://cdn.example.com/fallback.jpg"


def test_get_high_res_image_returns_none_on_request_exception(mocker):
    mocker.patch(
        "src.classifier.web_scraper.requests.get",
        side_effect=requests.exceptions.RequestException("timeout"),
    )
    assert get_high_res_image("https://pin.it/3") is None


def test_get_high_res_image_returns_none_when_no_image_tags(mocker):
    mocker.patch("src.classifier.web_scraper.requests.get", return_value=_mock_response(_EMPTY_HTML))
    assert get_high_res_image("https://pin.it/4") is None


# ─── download_images ───────────────────────────────────────────────────────────

def test_download_images_creates_output_dir(mocker, tmp_path):
    save_path = str(tmp_path / "out")
    mocker.patch("src.classifier.web_scraper.get_high_res_image", return_value=None)
    download_images(["https://pin.it/1"], save_path)
    assert (tmp_path / "out").is_dir()


def test_download_images_counts_saved_and_failed(mocker, tmp_path):
    save_path = str(tmp_path / "imgs")

    # get_high_res_image: first pin → url, second → None (counted as failed)
    mocker.patch(
        "src.classifier.web_scraper.get_high_res_image",
        side_effect=["https://cdn.example.com/a.jpg", None],
    )

    # successful image download
    from unittest.mock import MagicMock
    img_resp = MagicMock()
    img_resp.raise_for_status = MagicMock()
    img_resp.iter_content = MagicMock(return_value=[b"data"])
    mocker.patch("src.classifier.web_scraper.requests.get", return_value=img_resp)
    mocker.patch("src.classifier.web_scraper.time.sleep")

    count, failed = download_images(["https://pin.it/1", "https://pin.it/2"], save_path)
    assert count == 1
    assert failed == 1

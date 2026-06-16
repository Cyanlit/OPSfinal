"""
Integration tests for the FastAPI endpoints in main.py.

EasyOCR is expensive to load, so get_reader() is patched with a lightweight
MagicMock for all tests.  Tests that need specific OCR output override
mock_reader.readtext.return_value directly inside the test body.
"""
import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

import main
from main import app

# ── default mock reader ───────────────────────────────────────────────────────

_DEFAULT_OCR = [
    ([[10, 10], [100, 10], [100, 30], [10, 30]], "UNOCHA",  0.99),
    ([[10, 40], [100, 40], [100, 60], [10, 60]], "TEST",    0.85),
    ([[10, 70], [100, 70], [100, 90], [10, 90]], "ignored", 0.20),  # below default threshold
]


@pytest.fixture(scope="module")
def mock_reader():
    reader = MagicMock()
    reader.readtext.return_value = list(_DEFAULT_OCR)
    return reader


@pytest.fixture(scope="module")
def client(mock_reader):
    with patch.object(main, "get_reader", return_value=mock_reader):
        with TestClient(app) as c:
            yield c


# ── image factories ──────────────────────────────────────────────────────────

def _png(h=200, w=200) -> bytes:
    _, buf = cv2.imencode(".png", np.full((h, w, 3), 200, dtype=np.uint8))
    return buf.tobytes()


def _jpg(h=200, w=200) -> bytes:
    _, buf = cv2.imencode(".jpg", np.full((h, w, 3), 200, dtype=np.uint8))
    return buf.tobytes()


# ── /health ──────────────────────────────────────────────────────────────────

class TestHealthCheck:
    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_status_field_is_healthy(self, client):
        assert client.get("/health").json()["status"] == "healthy"

    def test_timestamp_present_and_utc(self, client):
        ts = client.get("/health").json()["timestamp"]
        assert isinstance(ts, str)
        assert ts.endswith("Z")


# ── /api/v1/scan — invalid inputs ────────────────────────────────────────────

class TestScanInvalidInputs:
    def test_unsupported_extension_txt(self, client):
        resp = client.post(
            "/api/v1/scan",
            files={"file": ("doc.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "INVALID_FILE_TYPE"

    def test_unsupported_extension_gif(self, client):
        resp = client.post(
            "/api/v1/scan",
            files={"file": ("anim.gif", b"GIF89a", "image/gif")},
        )
        assert resp.status_code == 400

    def test_unsupported_extension_bmp(self, client):
        resp = client.post(
            "/api/v1/scan",
            files={"file": ("img.bmp", b"\x42\x4D", "image/bmp")},
        )
        assert resp.status_code == 400

    def test_confidence_above_1_rejected(self, client):
        resp = client.post(
            "/api/v1/scan",
            files={"file": ("img.png", _png(), "image/png")},
            data={"min_confidence": "1.5"},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "INVALID_PARAMETER"

    def test_confidence_below_0_rejected(self, client):
        resp = client.post(
            "/api/v1/scan",
            files={"file": ("img.png", _png(), "image/png")},
            data={"min_confidence": "-0.1"},
        )
        assert resp.status_code == 400

    def test_corrupted_bytes_returns_422(self, client):
        resp = client.post(
            "/api/v1/scan",
            files={"file": ("img.png", b"\x00\x01\x02\x03\xFF\xFE", "image/png")},
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "IMAGE_DECODE_FAILED"

    def test_file_too_large_rejected(self, client):
        big = b"\x00" * (10 * 1024 * 1024 + 1)
        resp = client.post(
            "/api/v1/scan",
            files={"file": ("img.png", big, "image/png")},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "FILE_TOO_LARGE"

    def test_confidence_exactly_0_accepted(self, client, mock_reader):
        mock_reader.readtext.return_value = list(_DEFAULT_OCR)
        resp = client.post(
            "/api/v1/scan",
            files={"file": ("img.png", _png(), "image/png")},
            data={"min_confidence": "0.0"},
        )
        assert resp.status_code == 200

    def test_confidence_exactly_1_accepted(self, client, mock_reader):
        mock_reader.readtext.return_value = list(_DEFAULT_OCR)
        resp = client.post(
            "/api/v1/scan",
            files={"file": ("img.png", _png(), "image/png")},
            data={"min_confidence": "1.0"},
        )
        assert resp.status_code == 200


# ── /api/v1/scan — valid requests ────────────────────────────────────────────

class TestScanValidImage:
    def test_png_returns_200(self, client, mock_reader):
        mock_reader.readtext.return_value = list(_DEFAULT_OCR)
        resp = client.post(
            "/api/v1/scan",
            files={"file": ("img.png", _png(), "image/png")},
        )
        assert resp.status_code == 200

    def test_jpeg_returns_200(self, client, mock_reader):
        mock_reader.readtext.return_value = list(_DEFAULT_OCR)
        resp = client.post(
            "/api/v1/scan",
            files={"file": ("img.jpg", _jpg(), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_jpeg_alt_extension_returns_200(self, client, mock_reader):
        mock_reader.readtext.return_value = list(_DEFAULT_OCR)
        resp = client.post(
            "/api/v1/scan",
            files={"file": ("img.jpeg", _jpg(), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_response_has_required_top_level_keys(self, client, mock_reader):
        mock_reader.readtext.return_value = list(_DEFAULT_OCR)
        body = client.post(
            "/api/v1/scan",
            files={"file": ("img.png", _png(), "image/png")},
        ).json()
        assert body["success"] is True
        assert "metadata" in body
        assert "predictions" in body
        assert "grouped" in body

    def test_metadata_contains_required_fields(self, client, mock_reader):
        mock_reader.readtext.return_value = list(_DEFAULT_OCR)
        meta = client.post(
            "/api/v1/scan",
            files={"file": ("img.png", _png(300, 400), "image/png")},
        ).json()["metadata"]
        for field in ("width", "height", "processed_dimensions", "noise_level",
                      "noise_score", "denoising"):
            assert field in meta, f"metadata missing field: {field}"

    def test_metadata_width_height_match_image(self, client, mock_reader):
        mock_reader.readtext.return_value = list(_DEFAULT_OCR)
        meta = client.post(
            "/api/v1/scan",
            files={"file": ("img.png", _png(300, 400), "image/png")},
        ).json()["metadata"]
        assert meta["width"] == 400
        assert meta["height"] == 300

    def test_noise_level_is_int_in_range(self, client, mock_reader):
        mock_reader.readtext.return_value = list(_DEFAULT_OCR)
        meta = client.post(
            "/api/v1/scan",
            files={"file": ("img.png", _png(), "image/png")},
        ).json()["metadata"]
        assert isinstance(meta["noise_level"], int)
        assert 0 <= meta["noise_level"] <= 9

    def test_denoising_label_matches_level(self, client, mock_reader):
        from main import _NOISE_LEVEL_LABELS
        mock_reader.readtext.return_value = list(_DEFAULT_OCR)
        meta = client.post(
            "/api/v1/scan",
            files={"file": ("img.png", _png(), "image/png")},
        ).json()["metadata"]
        assert meta["denoising"] == _NOISE_LEVEL_LABELS[meta["noise_level"]]

    def test_predictions_filtered_by_min_confidence(self, client, mock_reader):
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "High", 0.95),
            ([[0, 40], [100, 40], [100, 70], [0, 70]], "Low",  0.20),
        ]
        preds = client.post(
            "/api/v1/scan",
            files={"file": ("img.png", _png(), "image/png")},
            data={"min_confidence": "0.5"},
        ).json()["predictions"]
        assert all(p["confidence"] >= 0.5 for p in preds)
        texts = [p["text"] for p in preds]
        assert "High" in texts
        assert "Low" not in texts

    def test_default_min_confidence_is_0_5(self, client, mock_reader):
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "Pass", 0.90),
            ([[0, 40], [100, 40], [100, 70], [0, 70]], "Fail", 0.30),
        ]
        preds = client.post(
            "/api/v1/scan",
            files={"file": ("img.png", _png(), "image/png")},
        ).json()["predictions"]
        texts = [p["text"] for p in preds]
        assert "Pass" in texts
        assert "Fail" not in texts

    def test_prediction_bounding_box_format(self, client, mock_reader):
        mock_reader.readtext.return_value = list(_DEFAULT_OCR)
        preds = client.post(
            "/api/v1/scan",
            files={"file": ("img.png", _png(), "image/png")},
            data={"min_confidence": "0.0"},
        ).json()["predictions"]
        for pred in preds:
            bb = pred["bounding_box"]
            assert len(bb) == 4
            for pt in bb:
                assert len(pt) == 2
                assert all(isinstance(v, int) for v in pt)

    def test_grouped_has_orientation_rows_columns(self, client, mock_reader):
        mock_reader.readtext.return_value = list(_DEFAULT_OCR)
        grouped = client.post(
            "/api/v1/scan",
            files={"file": ("img.png", _png(), "image/png")},
        ).json()["grouped"]
        assert "orientation" in grouped
        assert "rows" in grouped
        assert "columns" in grouped
        assert grouped["orientation"] in ("row", "column", "mixed")

    def test_empty_predictions_when_all_below_threshold(self, client, mock_reader):
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "low", 0.10),
        ]
        body = client.post(
            "/api/v1/scan",
            files={"file": ("img.png", _png(), "image/png")},
            data={"min_confidence": "0.9"},
        ).json()
        assert body["success"] is True
        assert body["predictions"] == []

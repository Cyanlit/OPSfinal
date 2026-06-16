"""
Shared fixtures for all test modules.

Path setup: adds the project root to sys.path so `import main` works
regardless of where pytest is invoked from.
"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── synthetic image factories ────────────────────────────────────────────────

def _solid(h: int, w: int, value: int = 180) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _noisy(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _encode(img: np.ndarray, ext: str = ".png") -> bytes:
    _, buf = cv2.imencode(ext, img)
    return buf.tobytes()


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_img() -> np.ndarray:
    """Uniform 2.25 MP image (res_bump=0, no hard cap) — noise_score ≈ 0, level 0 expected."""
    return _solid(1500, 1500)


@pytest.fixture
def noisy_img() -> np.ndarray:
    """Fully random 0.64 MP image — very high noise, level ≥ 5 expected."""
    return _noisy(800, 800)


@pytest.fixture
def highres_img() -> np.ndarray:
    """Clean image above 6 MP — hard cap applies."""
    return _solid(3000, 2200)


@pytest.fixture
def lowres_img() -> np.ndarray:
    """Clean image below 0.5 MP — res_bump = +3 applies."""
    return _solid(400, 300)


@pytest.fixture
def png_bytes() -> bytes:
    """Minimal valid 200×200 PNG."""
    return _encode(_solid(200, 200))


@pytest.fixture
def jpg_bytes() -> bytes:
    """Minimal valid 200×200 JPEG."""
    return _encode(_solid(200, 200), ".jpg")


@pytest.fixture
def corrupt_bytes() -> bytes:
    """Bytes that look like an image extension but cannot be decoded."""
    return b"\x00\x01\x02\x03\xFF\xFE"

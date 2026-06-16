"""
Unit tests for image-processing utilities in main.py:
  _order_points, _four_point_transform, _estimate_noise_level,
  _DENOISE_PARAMS, _NOISE_LEVEL_LABELS
"""
import numpy as np
import cv2
import pytest

from main import (
    _order_points,
    _four_point_transform,
    _estimate_noise_level,
    _DENOISE_PARAMS,
    _NOISE_LEVEL_LABELS,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _solid(h, w, v=180):
    return np.full((h, w, 3), v, dtype=np.uint8)


def _noisy(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


# GaussianBlur(5,5) attenuates residual noise to ~60 % of the raw std.
# Divide the naive std by this factor so the measured noise_score hits the target.
_BLUR_ATTENUATION = 0.60


def _img_with_noise_score(target_score: float, h=1414, w=1500) -> np.ndarray:
    """
    Build a ~2.1 MP image (res_bump=0, no hard cap) whose measured noise_score
    is close to `target_score`.

    The score formula is std(img - GaussianBlur(img)) / mean(img) * 100.
    GaussianBlur suppresses high-frequency noise, so the residual std is only
    ~60 % of the raw noise std.  We compensate with _BLUR_ATTENUATION.
    """
    mean_brightness = 128.0
    target_std = target_score * mean_brightness / 100.0 / _BLUR_ATTENUATION
    rng = np.random.default_rng(42)
    noise = rng.normal(0, target_std, (h, w, 3))
    return np.clip(mean_brightness + noise, 0, 255).astype(np.uint8)


# ── _order_points ─────────────────────────────────────────────────────────────

class TestOrderPoints:
    def test_axis_aligned_rectangle(self):
        pts = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
        tl, tr, br, bl = _order_points(pts)
        np.testing.assert_array_equal(tl, [0, 0])
        np.testing.assert_array_equal(tr, [100, 0])
        np.testing.assert_array_equal(br, [100, 50])
        np.testing.assert_array_equal(bl, [0, 50])

    def test_scrambled_order_is_corrected(self):
        # Provide BR, TL, TR, BL — should still come out in TL TR BR BL order.
        pts = np.array([[100, 50], [0, 0], [100, 0], [0, 50]], dtype=np.float32)
        tl, tr, br, bl = _order_points(pts)
        np.testing.assert_array_almost_equal(tl, [0, 0])
        np.testing.assert_array_almost_equal(tr, [100, 0])
        np.testing.assert_array_almost_equal(br, [100, 50])
        np.testing.assert_array_almost_equal(bl, [0, 50])

    def test_output_shape_is_4x2(self):
        pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        result = _order_points(pts)
        assert result.shape == (4, 2)

    def test_output_dtype_is_float32(self):
        pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        result = _order_points(pts)
        assert result.dtype == np.float32


# ── _four_point_transform ─────────────────────────────────────────────────────

class TestFourPointTransform:
    def _rect(self, x0, y0, x1, y1):
        return np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
        )

    def test_valid_rectangle_returns_tuple(self):
        img = _solid(200, 200)
        result = _four_point_transform(img, self._rect(10, 10, 190, 190))
        assert result is not None
        warped, w, h = result
        assert warped.ndim == 3
        assert w > 0 and h > 0

    def test_output_dimensions_match_rectangle(self):
        img = _solid(300, 400)
        result = _four_point_transform(img, self._rect(10, 20, 110, 70))
        assert result is not None
        _, w, h = result
        assert w == pytest.approx(100, abs=2)
        assert h == pytest.approx(50, abs=2)

    def test_degenerate_collinear_returns_none(self):
        """All four points on the same line → zero-area → must return None."""
        img = _solid(200, 200)
        collinear = np.array(
            [[0, 0], [100, 0], [100, 0], [0, 0]], dtype=np.float32
        )
        assert _four_point_transform(img, collinear) is None

    def test_zero_width_returns_none(self):
        img = _solid(200, 200)
        zero_w = np.array(
            [[50, 10], [50, 10], [50, 90], [50, 90]], dtype=np.float32
        )
        assert _four_point_transform(img, zero_w) is None


# ── _estimate_noise_level ─────────────────────────────────────────────────────

class TestEstimateNoiseLevel:
    def test_clean_image_gives_level_zero(self, clean_img):
        level, score = _estimate_noise_level(clean_img)
        assert level == 0
        assert score < 5.0

    def test_noisy_image_gives_high_level(self, noisy_img):
        level, score = _estimate_noise_level(noisy_img)
        assert level >= 5
        assert score > 15.0

    def test_highres_hard_cap_at_five(self, highres_img):
        """Images > 6 MP must never exceed level 5."""
        # Inject heavy noise into the high-res image to force a high base level.
        rng = np.random.default_rng(7)
        noisy_hr = rng.integers(0, 255, highres_img.shape, dtype=np.uint8)
        level, _ = _estimate_noise_level(noisy_hr)
        assert level <= 5

    def test_lowres_gets_higher_level_than_same_noise_medium_res(self):
        """res_bump=3 for < 0.5 MP means a mildly noisy small image gets a higher level."""
        rng = np.random.default_rng(2)
        base_noise = rng.integers(100, 160, (1200, 1200, 3), dtype=np.uint8)
        small = cv2.resize(base_noise, (350, 250))   # ~0.09 MP
        level_med, _ = _estimate_noise_level(base_noise)
        level_small, _ = _estimate_noise_level(small)
        assert level_small >= level_med

    def test_returns_int_and_float(self, clean_img):
        level, score = _estimate_noise_level(clean_img)
        assert isinstance(level, int)
        assert isinstance(score, float)

    def test_level_always_in_0_to_9(self):
        for seed in range(5):
            img = _noisy(600, 600, seed)
            level, _ = _estimate_noise_level(img)
            assert 0 <= level <= 9, f"level {level} out of range for seed {seed}"

    def test_score_is_non_negative(self, clean_img):
        _, score = _estimate_noise_level(clean_img)
        assert score >= 0.0

    @pytest.mark.parametrize("target_score,expected_level", [
        # Use bracket-centre values (not boundary values) to avoid edge-noise flips.
        # All images are ~2.1 MP (res_bump=0, no hard cap), so level == base.
        (2.5,  0),   # centre of  < 5      → level 0
        (6.0,  1),   # centre of  5–7      → level 1
        (8.0,  2),   # centre of  7–9      → level 2
        (10.0, 3),   # centre of  9–11     → level 3
        (12.0, 4),   # centre of 11–13     → level 4
        (14.5, 5),   # centre of 13–16     → level 5
        (18.0, 6),   # centre of 16–20     → level 6
        (22.5, 7),   # centre of 20–25     → level 7
        (28.0, 8),   # centre of 25–31     → level 8
        (40.0, 9),   # well above 31       → level 9
    ])
    def test_noise_score_threshold_brackets(self, target_score, expected_level):
        """
        Verify each noise_score bracket maps to the correct level.
        Uses a ~2.1 MP image so res_bump=0 and level == base directly.
        _img_with_noise_score compensates for GaussianBlur residual attenuation
        (~0.60×) so the measured score lands near target_score.
        """
        img = _img_with_noise_score(target_score)
        level, measured = _estimate_noise_level(img)
        assert level == expected_level, (
            f"target_score={target_score}, measured={measured:.2f} "
            f"→ expected level {expected_level}, got {level}"
        )


# ── _DENOISE_PARAMS and _NOISE_LEVEL_LABELS ──────────────────────────────────

class TestDenoiseMeta:
    def test_denoise_params_covers_levels_1_to_9(self):
        for lv in range(1, 10):
            assert lv in _DENOISE_PARAMS, f"Level {lv} missing from _DENOISE_PARAMS"

    def test_denoise_params_values_are_positive(self):
        for lv, (h, hc) in _DENOISE_PARAMS.items():
            assert h > 0 and hc > 0, f"Level {lv}: h={h}, hColor={hc}"

    def test_h_values_monotonically_increase(self):
        """Higher level → more aggressive denoising → larger h."""
        h_vals = [_DENOISE_PARAMS[lv][0] for lv in range(1, 10)]
        assert h_vals == sorted(h_vals), f"h values not sorted: {h_vals}"

    def test_noise_level_labels_covers_0_to_9(self):
        for lv in range(0, 10):
            assert lv in _NOISE_LEVEL_LABELS, f"Level {lv} missing from _NOISE_LEVEL_LABELS"

    def test_noise_level_labels_are_non_empty_strings(self):
        for lv, label in _NOISE_LEVEL_LABELS.items():
            assert isinstance(label, str) and label.strip(), (
                f"Level {lv} has empty/invalid label: {label!r}"
            )

    def test_level_zero_label_is_none(self):
        assert _NOISE_LEVEL_LABELS[0] == "none"

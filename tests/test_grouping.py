"""
Unit tests for prediction-grouping utilities in main.py:
  _bbox_metrics, _detect_orientation, _cluster_axis, _group_predictions
"""
import pytest

from main import (
    _bbox_metrics,
    _detect_orientation,
    _cluster_axis,
    _group_predictions,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _pred(text: str, conf: float, x0: int, y0: int, x1: int, y1: int) -> dict:
    """Axis-aligned prediction dict."""
    return {
        "text": text,
        "confidence": conf,
        "bounding_box": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
    }


def _item(text: str, conf: float, x0: int, y0: int, x1: int, y1: int) -> dict:
    """Prediction dict enriched with _bbox_metrics."""
    return _bbox_metrics(_pred(text, conf, x0, y0, x1, y1))


# ── _bbox_metrics ─────────────────────────────────────────────────────────────

class TestBboxMetrics:
    def test_horizontal_box_dimensions(self):
        m = _bbox_metrics(_pred("hi", 0.9, 10, 20, 110, 50))
        assert m["_x_min"] == 10
        assert m["_x_max"] == 110
        assert m["_y_min"] == 20
        assert m["_y_max"] == 50
        assert m["_width"] == 100
        assert m["_height"] == 30

    def test_midpoints(self):
        m = _bbox_metrics(_pred("x", 0.8, 0, 0, 100, 40))
        assert m["_x_mid"] == pytest.approx(50.0)
        assert m["_y_mid"] == pytest.approx(20.0)

    def test_vertical_box_has_height_greater_than_width(self):
        m = _bbox_metrics(_pred("字", 0.8, 50, 10, 80, 110))
        assert m["_height"] > m["_width"]

    def test_original_fields_preserved(self):
        p = _pred("test", 0.75, 0, 0, 100, 20)
        m = _bbox_metrics(p)
        assert m["text"] == "test"
        assert m["confidence"] == 0.75
        assert "bounding_box" in m

    def test_internal_keys_are_prefixed(self):
        m = _bbox_metrics(_pred("a", 0.5, 0, 0, 10, 10))
        for key in ("_x_min", "_x_max", "_x_mid", "_y_min", "_y_max", "_y_mid",
                    "_width", "_height"):
            assert key in m


# ── _detect_orientation ───────────────────────────────────────────────────────

class TestDetectOrientation:
    def test_wide_boxes_give_row(self):
        items = [_item("a", 0.9, i * 120, 0, i * 120 + 100, 20) for i in range(5)]
        assert _detect_orientation(items) == "row"

    def test_tall_boxes_give_column(self):
        items = [_item("字", 0.9, 0, i * 120, 20, i * 120 + 100) for i in range(5)]
        assert _detect_orientation(items) == "column"

    def test_empty_list_defaults_to_row(self):
        assert _detect_orientation([]) == "row"

    def test_single_wide_item(self):
        assert _detect_orientation([_item("w", 0.9, 0, 0, 200, 20)]) == "row"

    def test_single_tall_item(self):
        assert _detect_orientation([_item("t", 0.9, 0, 0, 20, 200)]) == "column"

    def test_orientation_is_one_of_valid_values(self):
        items = [_item("x", 0.9, 0, 0, 50, 50)]  # square — could be mixed
        result = _detect_orientation(items)
        assert result in ("row", "column", "mixed")


# ── _cluster_axis ─────────────────────────────────────────────────────────────

class TestClusterAxis:
    # Shorthand: cluster by Y (rows), inner sort by X
    def _rows(self, items):
        return _cluster_axis(items, "_y_mid", "_y_min", "_y_max", "_height", "_x_min")

    # Cluster by X (columns), inner sort by Y
    def _cols(self, items):
        return _cluster_axis(items, "_x_mid", "_x_min", "_x_max", "_width", "_y_min")

    def test_two_items_same_row_merged(self):
        items = [_item("Hello", 0.9, 10, 10, 110, 40), _item("World", 0.9, 130, 10, 230, 40)]
        result = self._rows(items)
        assert len(result) == 1
        assert result[0]["text"] == "Hello World"

    def test_two_items_different_rows_stay_separate(self):
        items = [_item("Row1", 0.9, 10, 10, 110, 40), _item("Row2", 0.9, 10, 300, 110, 330)]
        result = self._rows(items)
        assert len(result) == 2

    def test_within_row_sorted_left_to_right(self):
        items = [_item("B", 0.9, 200, 10, 300, 40), _item("A", 0.9, 10, 10, 110, 40)]
        result = self._rows(items)
        assert result[0]["text"] == "A B"

    def test_large_vertical_gap_creates_new_block(self):
        items = [_item("P1", 0.9, 10, 10, 110, 40), _item("P2", 0.9, 10, 500, 110, 530)]
        result = self._rows(items)
        assert len(result) == 2
        assert result[0]["block"] != result[1]["block"]

    def test_small_vertical_gap_stays_same_block(self):
        items = [_item("L1", 0.9, 10, 10, 110, 40), _item("L2", 0.9, 10, 50, 110, 80)]
        result = self._rows(items)
        assert result[0]["block"] == result[1]["block"]

    def test_column_clustering_same_x(self):
        items = [_item("Top", 0.9, 100, 10, 130, 40), _item("Bot", 0.9, 100, 200, 130, 230)]
        result = self._cols(items)
        assert len(result) == 1
        assert result[0]["text"] == "Top Bot"

    def test_confidence_is_average(self):
        items = [_item("A", 0.8, 10, 10, 110, 40), _item("B", 0.6, 130, 10, 230, 40)]
        result = self._rows(items)
        assert result[0]["confidence"] == pytest.approx(0.7, abs=1e-4)

    def test_each_result_has_required_keys(self):
        items = [_item("x", 0.9, 0, 0, 100, 30)]
        result = self._rows(items)
        assert "block" in result[0]
        assert "text" in result[0]
        assert "confidence" in result[0]

    def test_three_rows_three_groups(self):
        items = [
            _item("A", 0.9, 10, 10, 110, 40),
            _item("B", 0.9, 10, 200, 110, 230),
            _item("C", 0.9, 10, 400, 110, 430),
        ]
        result = self._rows(items)
        assert len(result) == 3

    def test_multiple_items_per_row_all_merged(self):
        row_y = (10, 40)
        items = [_item(str(i), 0.9, i * 60, row_y[0], i * 60 + 50, row_y[1]) for i in range(4)]
        result = self._rows(items)
        assert len(result) == 1
        assert result[0]["text"] == "0 1 2 3"


# ── _group_predictions ────────────────────────────────────────────────────────

class TestGroupPredictions:
    def test_empty_returns_canonical_structure(self):
        assert _group_predictions([]) == {"orientation": "row", "rows": [], "columns": []}

    def test_output_has_all_three_keys(self):
        result = _group_predictions([_pred("x", 0.9, 0, 0, 100, 20)])
        assert set(result.keys()) == {"orientation", "rows", "columns"}

    def test_orientation_is_valid_string(self):
        result = _group_predictions([_pred("x", 0.9, 0, 0, 100, 20)])
        assert result["orientation"] in ("row", "column", "mixed")

    def test_horizontal_layout_detected_as_row(self):
        preds = [_pred(c, 0.9, i * 120, 0, i * 120 + 100, 20) for i, c in enumerate("ABC")]
        assert _group_predictions(preds)["orientation"] == "row"

    def test_vertical_layout_detected_as_column(self):
        preds = [_pred(c, 0.9, 0, i * 120, 20, i * 120 + 100) for i, c in enumerate("甲乙丙")]
        assert _group_predictions(preds)["orientation"] == "column"

    def test_same_line_items_merged_in_rows(self):
        preds = [_pred("Hello", 0.9, 10, 10, 110, 40), _pred("World", 0.9, 130, 10, 230, 40)]
        rows = _group_predictions(preds)["rows"]
        assert len(rows) == 1
        assert rows[0]["text"] == "Hello World"

    def test_same_column_items_merged_in_columns(self):
        preds = [_pred("上", 0.9, 10, 10, 30, 80), _pred("下", 0.9, 10, 100, 30, 170)]
        cols = _group_predictions(preds)["columns"]
        assert len(cols) == 1
        assert cols[0]["text"] == "上 下"

    def test_row_items_have_block_text_confidence(self):
        result = _group_predictions([_pred("hi", 0.9, 0, 0, 100, 20)])
        for row in result["rows"]:
            assert "block" in row and "text" in row and "confidence" in row

    def test_column_items_have_block_text_confidence(self):
        result = _group_predictions([_pred("hi", 0.9, 0, 0, 20, 100)])
        for col in result["columns"]:
            assert "block" in col and "text" in col and "confidence" in col

    def test_two_rows_two_row_entries(self):
        preds = [_pred("A", 0.9, 10, 10, 110, 40), _pred("B", 0.9, 10, 300, 110, 330)]
        rows = _group_predictions(preds)["rows"]
        assert len(rows) == 2

    def test_confidence_propagated_correctly(self):
        preds = [_pred("X", 0.8, 0, 0, 100, 20), _pred("Y", 0.6, 120, 0, 220, 20)]
        rows = _group_predictions(preds)["rows"]
        assert rows[0]["confidence"] == pytest.approx(0.7, abs=1e-4)

"""OCR 預測結果共用工具（GUI / LINE Bot）。"""

from __future__ import annotations


def sort_predictions_reading_order(predictions: list[dict]) -> list[dict]:
    """依標框位置排序：同一行由左至右，整體由上至下。"""
    if not predictions:
        return predictions

    items: list[tuple[float, float, float, dict]] = []
    for pred in predictions:
        bbox = pred.get("bounding_box") or []
        if len(bbox) < 1:
            items.append((0.0, 0.0, 0.0, pred))
            continue

        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        min_x, min_y = min(xs), min(ys)
        height = max(ys) - min_y
        items.append((min_y, min_x, height, pred))

    heights = [h for _, _, h, _ in items if h > 0]
    if heights:
        median_height = sorted(heights)[len(heights) // 2]
        row_tolerance = max(median_height * 0.55, 10.0)
    else:
        row_tolerance = 15.0

    def reading_key(entry: tuple[float, float, float, dict]) -> tuple[int, float]:
        min_y, min_x, _, _ = entry
        row = int(min_y // row_tolerance)
        return (row, min_x)

    items.sort(key=reading_key)
    return [entry[3] for entry in items]

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import cv2
import easyocr
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None  # type: ignore[assignment]

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    EASYOCR_MODEL_STORAGE: str = "./models"
    ANTHROPIC_API_KEY: str = ""

    # LINE Bot
    LINE_CHANNEL_SECRET: str = ""
    LINE_CHANNEL_ACCESS_TOKEN: str = ""
    OCR_SERVICE_URL: str = "http://127.0.0.1:8000"
    MIN_CONFIDENCE: float = 0.5


settings = Settings()
logging.basicConfig(level=logging.INFO)
app = FastAPI(title="OCR Service", version="1.0.0")

_reader: easyocr.Reader | None = None


def get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(
            ["ch_tra", "en"],
            model_storage_directory=settings.EASYOCR_MODEL_STORAGE,
            gpu=False,
        )
    return _reader


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def _four_point_transform(
    image: np.ndarray, rect: np.ndarray
) -> tuple[np.ndarray, int, int] | None:
    tl, tr, br, bl = rect
    max_width = max(
        int(np.linalg.norm(br - bl)),
        int(np.linalg.norm(tr - tl)),
    )
    max_height = max(
        int(np.linalg.norm(tr - br)),
        int(np.linalg.norm(tl - bl)),
    )
    # Bug 1: degenerate rect (all points nearly collinear) produces 0 or negative dst coords
    if max_width < 1 or max_height < 1:
        return None
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped, max_width, max_height


# Denoising strength per level: (h, hColor)
_DENOISE_PARAMS: dict[int, tuple[int, int]] = {
    1: (3, 3),   # minimal    — barely perceptible grain
    2: (5, 5),   # very-light — slight sensor noise
    3: (7, 7),   # light      — mild ISO grain
    4: (9, 9),   # light-med  — indoor / dim lighting
    5: (11, 11), # medium     — typical phone-camera noise
    6: (13, 13), # med-heavy  — poor lighting or high ISO
    7: (15, 15), # heavy      — very noisy / high-ISO
    8: (18, 18), # very-heavy — low-res or compressed source
    9: (21, 21), # extreme    — severely degraded image
}

_NOISE_LEVEL_LABELS = {
    0: "none",
    1: "minimal",
    2: "very-light",
    3: "light",
    4: "light-medium",
    5: "medium",
    6: "medium-heavy",
    7: "heavy",
    8: "very-heavy",
    9: "extreme",
}


def _estimate_noise_level(img: np.ndarray) -> tuple[int, float]:
    """
    Returns (level, noise_score):
      level 0–9 based on two independent signals:
      - noise_score: std of residual after subtracting a Gaussian-blurred copy,
                     normalized by mean brightness (scale-independent).
      - resolution:  penalises images with fewer pixels, which need more aggressive
                     denoising to compensate for smaller character strokes.

    Thresholds were chosen empirically on phone-camera receipts and scanned docs.
    """
    h, w = img.shape[:2]
    total_pixels = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual = gray - cv2.GaussianBlur(gray, (5, 5), 0)
    noise_score = float(np.std(residual)) / (float(np.mean(gray)) + 1e-6) * 100

    # Low-res images need a heavier hand regardless of measured noise;
    # very high-res images (>6 MP) have fine strokes that denoising can blur, so pull back one level.
    if total_pixels < 500_000:        # < 0.5 MP
        res_bump = 3
    elif total_pixels < 2_000_000:    # 0.5–2 MP
        res_bump = 1
    elif total_pixels > 6_000_000:    # > 6 MP (high-end phone / DSLR)
        res_bump = -1
    else:
        res_bump = 0

    # Benchmark on IMG_20260616_085152.jpg (12.6 MP, noise_score=3.03):
    # level-0 (no denoising) produced the cleanest OCR — all tested levels 1-9
    # introduced fragmentation or hallucination.  Root cause: the original 4-level
    # code used noise_score < 5 as the "no denoising" boundary; expanding to 10
    # levels had compressed that boundary to < 3, incorrectly triggering light
    # denoising on clean high-res images.  Restored to < 5.
    if noise_score < 5:
        base = 0          # clean — skip denoising entirely
    elif noise_score < 7:
        base = 1          # minimal grain
    elif noise_score < 9:
        base = 2          # very-light
    elif noise_score < 11:
        base = 3          # light
    elif noise_score < 13:
        base = 4          # light-medium
    elif noise_score < 16:
        base = 5          # medium
    elif noise_score < 20:
        base = 6          # medium-heavy
    elif noise_score < 25:
        base = 7          # heavy
    elif noise_score < 31:
        base = 8          # very-heavy
    else:
        base = 9          # extreme

    level = max(0, min(base + res_bump, 9))

    # Hard cap: >6 MP images must not exceed level 5 — h > 11 blurs fine strokes
    if total_pixels > 6_000_000:
        level = min(level, 5)

    return level, round(noise_score, 2)


def _deskew(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return img
    # Bug 4: large images can have millions of coords — subsample to keep minAreaRect fast
    if len(coords) > 50_000:
        idx = np.random.choice(len(coords), 50_000, replace=False)
        coords = coords[idx]
    # minAreaRect expects (x, y); np.where gives (row=y, col=x) so swap
    points = coords[:, ::-1].astype(np.float32)
    angle = cv2.minAreaRect(points)[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    if abs(angle) < 0.5:   # skip trivial corrections
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _enhance_image(img: np.ndarray) -> tuple[np.ndarray, int, float]:
    noise_level, noise_score = _estimate_noise_level(img)

    if noise_level > 0:
        h_lum, h_col = _DENOISE_PARAMS[noise_level]
        img = cv2.fastNlMeansDenoisingColored(
            img, None,
            h=h_lum, hColor=h_col,
            templateWindowSize=7, searchWindowSize=21,
        )

    # CLAHE on L channel to fix uneven lighting from phone cameras
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Unsharp mask to recover edge sharpness lost in phone optics
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    return img, noise_level, noise_score


def _bbox_metrics(pred: dict) -> dict:
    pts = pred["bounding_box"]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return {
        **pred,
        "_x_min": min(xs),
        "_x_max": max(xs),
        "_x_mid": sum(xs) / len(xs),
        "_y_min": min(ys),
        "_y_max": max(ys),
        "_y_mid": sum(ys) / len(ys),
        "_width": max(xs) - min(xs),
        "_height": max(ys) - min(ys),
    }


def _detect_orientation(items: list[dict]) -> str:
    """
    Classifies layout as 'row', 'column', or 'mixed' using two signals:

    1. Bounding-box aspect ratio — horizontal text (行) produces wide boxes
       (width > height); vertical text (列) produces tall boxes (height > width).
    2. Layout spread ratio — compare the normalised Y-spread of same-X-cluster
       against the normalised X-spread of same-Y-cluster.  The axis that produces
       more cohesive clusters dominates the layout.

    'row'    → ≥ 70 % of boxes are wide, or Y-clusters are tighter
    'column' → ≥ 70 % of boxes are tall, or X-clusters are tighter
    'mixed'  → neither signal is conclusive
    """
    if not items:
        return "row"

    # Signal 1: per-box aspect ratio vote
    h_votes = sum(1 for it in items if it["_width"] > it["_height"])
    v_votes = len(items) - h_votes
    h_ratio = h_votes / len(items)

    if h_ratio >= 0.7:
        return "row"
    if h_ratio <= 0.3:
        return "column"

    # Signal 2: cluster-spread comparison
    # Group by Y (rows) and measure X variance; group by X (cols) and measure Y variance.
    # Tighter secondary-axis variance means that axis is more structured.
    def _spread(groups: list[list[dict]], secondary_key: str) -> float:
        variances = []
        for g in groups:
            vals = [it[secondary_key] for it in g]
            mean = sum(vals) / len(vals)
            variances.append(sum((v - mean) ** 2 for v in vals) / len(vals))
        return sum(variances) / len(variances) if variances else float("inf")

    y_mids = sorted(it["_y_mid"] for it in items)
    med_h = sorted(it["_height"] for it in items)[len(items) // 2]
    row_groups: list[list[dict]] = []
    for it in sorted(items, key=lambda x: x["_y_mid"]):
        if not row_groups or it["_y_mid"] - row_groups[-1][0]["_y_mid"] > med_h * 0.6:
            row_groups.append([it])
        else:
            row_groups[-1].append(it)

    med_w = sorted(it["_width"] for it in items)[len(items) // 2]
    col_groups: list[list[dict]] = []
    for it in sorted(items, key=lambda x: x["_x_mid"]):
        if not col_groups or it["_x_mid"] - col_groups[-1][0]["_x_mid"] > med_w * 0.6:
            col_groups.append([it])
        else:
            col_groups[-1].append(it)

    row_x_spread = _spread(row_groups, "_x_mid")
    col_y_spread = _spread(col_groups, "_y_mid")

    # Rows are tighter in Y, cols are tighter in X — compare the opposite-axis spread
    if row_x_spread < col_y_spread:
        return "row"
    if col_y_spread < row_x_spread:
        return "column"
    return "mixed"


def _cluster_axis(
    items: list[dict],
    mid_key: str,
    min_key: str,
    max_key: str,
    size_key: str,
    inner_sort_key: str,
) -> list[dict]:
    """
    Generic 1-D clustering.  Groups items whose `mid_key` values are close,
    then sorts each group by `inner_sort_key`, then groups clusters into blocks
    separated by large gaps along the same axis.

    Returns a flat list of {"block", "text", "confidence"} dicts.
    """
    sizes = sorted(it[size_key] for it in items)
    median_size = sizes[len(sizes) // 2]
    group_threshold = max(median_size * 0.6, 10)
    block_gap = max(median_size * 2.0, 20)

    sorted_items = sorted(items, key=lambda it: it[mid_key])
    groups: list[list[dict]] = []
    for item in sorted_items:
        if not groups or item[mid_key] - groups[-1][0][mid_key] > group_threshold:
            groups.append([item])
        else:
            groups[-1].append(item)

    for g in groups:
        g.sort(key=lambda it: it[inner_sort_key])

    blocks: list[list[list[dict]]] = [[groups[0]]]
    for i in range(1, len(groups)):
        prev_max = max(it[max_key] for it in groups[i - 1])
        curr_min = min(it[min_key] for it in groups[i])
        if curr_min - prev_max > block_gap:
            blocks.append([groups[i]])
        else:
            blocks[-1].append(groups[i])

    result = []
    for b_idx, block in enumerate(blocks):
        for group in block:
            result.append({
                "block": b_idx,
                "text": " ".join(it["text"] for it in group),
                "confidence": round(sum(it["confidence"] for it in group) / len(group), 4),
            })
    return result


def _group_predictions(predictions: list[dict]) -> dict:
    if not predictions:
        return {"orientation": "row", "rows": [], "columns": []}

    items = [_bbox_metrics(p) for p in predictions]
    orientation = _detect_orientation(items)

    rows = _cluster_axis(items, "_y_mid", "_y_min", "_y_max", "_height", "_x_min")
    columns = _cluster_axis(items, "_x_mid", "_x_min", "_x_max", "_width", "_y_min")

    return {"orientation": orientation, "rows": rows, "columns": columns}


def preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, dict]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "IMAGE_DECODE_FAILED",
                "message": "Cannot decode image data. The file may be corrupted.",
            },
        )

    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
        if len(approx) == 4:
            contour_area = cv2.contourArea(approx)
            # Reject contours that cover less than 10% of the image — likely a logo or UI element, not a document
            if contour_area / (h * w) >= 0.10:
                rect = _order_points(approx.reshape(4, 2).astype(np.float32))
                result = _four_point_transform(img, rect)
                if result is not None:
                    img, pw, ph = result
                    enhanced, noise_level, noise_score = _enhance_image(_deskew(img))
                    return enhanced, {
                        "width": w, "height": h,
                        "processed_dimensions": f"{pw}x{ph}",
                        "noise_level": noise_level,
                        "noise_score": noise_score,
                        "denoising": _NOISE_LEVEL_LABELS[noise_level],
                    }

    # No valid quadrilateral found — fall back to original image (e.g. flat scan, screenshot)
    enhanced, noise_level, noise_score = _enhance_image(_deskew(img))
    return enhanced, {
        "width": w, "height": h,
        "processed_dimensions": f"{w}x{h}",
        "noise_level": noise_level,
        "noise_score": noise_score,
        "denoising": _NOISE_LEVEL_LABELS[noise_level],
    }


def scan_image_bytes(image_bytes: bytes, min_confidence: float = 0.5) -> dict:
    """核心 OCR 流程，供 API 與 LINE Bot 共用（避免 HTTP 自呼叫逾時）。"""
    if not (0.0 <= min_confidence <= 1.0):
        raise ValueError("INVALID_PARAMETER")
    if len(image_bytes) > MAX_FILE_SIZE:
        raise ValueError("FILE_TOO_LARGE")

    try:
        processed_img, metadata = preprocess_image(image_bytes)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        raise ValueError(detail.get("error", "SCAN_FAILED")) from exc

    try:
        results = get_reader().readtext(processed_img, adjust_contrast=0.5)
    except (RuntimeError, MemoryError) as exc:
        raise ValueError("OCR_ENGINE_TIMEOUT") from exc

    predictions = [
        {
            "text": text,
            "confidence": round(float(conf), 4),
            "bounding_box": [[int(p[0]), int(p[1])] for p in bbox],
        }
        for bbox, text, conf in results
        if conf >= min_confidence
    ]

    return {
        "success": True,
        "metadata": metadata,
        "predictions": predictions,
        "grouped": _group_predictions(predictions),
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "line_configured": line_bot_service.is_configured,
    }


@app.post("/callback")
async def line_callback(request: Request):
    """LINE Messaging API Webhook。"""
    if not line_bot_service.is_configured:
        raise HTTPException(
            status_code=500,
            detail="LINE_CHANNEL_SECRET 與 LINE_CHANNEL_ACCESS_TOKEN 未設定。",
        )

    body = (await request.body()).decode("utf-8")
    signature = request.headers.get("X-Line-Signature", "")

    try:
        line_bot_service.handle_webhook(body, signature)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return "OK"


@app.post("/api/v1/scan")
async def scan_document(
    file: Annotated[UploadFile, File()],
    min_confidence: Annotated[float, Form()] = 0.5,
):
    # Bug 3: validate min_confidence is in [0, 1]
    if not (0.0 <= min_confidence <= 1.0):
        raise HTTPException(
            status_code=400,
            detail={"error": "INVALID_PARAMETER", "message": "min_confidence must be between 0.0 and 1.0."},
        )

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_FILE_TYPE",
                "message": "File extension is not a recognized image format.",
            },
        )

    # Bug 2: pre-check Content-Length header before reading into memory
    if file.size is not None and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail={"error": "FILE_TOO_LARGE", "message": "File exceeds 10 MB limit."},
        )
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail={"error": "FILE_TOO_LARGE", "message": "File exceeds 10 MB limit."},
        )

    processed_img, metadata = preprocess_image(image_bytes)

    try:
        results = get_reader().readtext(processed_img, adjust_contrast=0.5)
    except (RuntimeError, MemoryError) as exc:
        # Bug 5: catch only engine-level failures; let programming errors propagate
        raise HTTPException(
            status_code=503,
            detail={
                "error": "OCR_ENGINE_TIMEOUT",
                "message": "PyTorch worker thread experienced resource starvation.",
            },
        ) from exc

    predictions = [
        {
            "text": text,
            "confidence": round(float(conf), 4),
            "bounding_box": [[int(p[0]), int(p[1])] for p in bbox],
        }
        for bbox, text, conf in results
        if conf >= min_confidence
    ]

    return {
        "success": True,
        "metadata": metadata,
        "predictions": predictions,
        "grouped": _group_predictions(predictions),
    }

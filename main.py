from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import cv2
import easyocr
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic_settings import BaseSettings, SettingsConfigDict

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    EASYOCR_MODEL_STORAGE: str = "./models"


settings = Settings()
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
    1: (3, 3),   # light  — mild ISO grain
    2: (7, 7),   # medium — indoor / dim lighting
    3: (12, 12), # heavy  — very noisy or low-resolution
}

_NOISE_LEVEL_LABELS = {0: "none", 1: "light", 2: "medium", 3: "heavy"}


def _estimate_noise_level(img: np.ndarray) -> tuple[int, float]:
    """
    Returns (level, noise_score):
      level 0–3 based on two independent signals:
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
        res_bump = 2
    elif total_pixels < 2_000_000:    # 0.5–2 MP
        res_bump = 1
    elif total_pixels > 6_000_000:    # > 6 MP (high-end phone / DSLR)
        res_bump = -1
    else:
        res_bump = 0

    if noise_score < 5:
        base = 0
    elif noise_score < 9:
        base = 1
    elif noise_score < 14:
        base = 2
    else:
        base = 3

    level = max(0, min(base + res_bump, 3))

    # Hard cap: >6 MP images must not exceed level 2 — h=12 blurs fine strokes
    if total_pixels > 6_000_000:
        level = min(level, 2)

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


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


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

    return {"success": True, "metadata": metadata, "predictions": predictions}

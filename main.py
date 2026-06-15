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
) -> tuple[np.ndarray, int, int]:
    tl, tr, br, bl = rect
    max_width = max(
        int(np.linalg.norm(br - bl)),
        int(np.linalg.norm(tr - tl)),
    )
    max_height = max(
        int(np.linalg.norm(tr - br)),
        int(np.linalg.norm(tl - bl)),
    )
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped, max_width, max_height


def preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, dict]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "DOCUMENT_CONTOUR_NOT_FOUND",
                "message": "Cannot decode image data.",
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
            rect = _order_points(approx.reshape(4, 2).astype(np.float32))
            img, pw, ph = _four_point_transform(img, rect)
            return img, {"width": w, "height": h, "processed_dimensions": f"{pw}x{ph}"}

    raise HTTPException(
        status_code=422,
        detail={
            "error": "DOCUMENT_CONTOUR_NOT_FOUND",
            "message": "OpenCV cannot resolve four valid corner vertices.",
        },
    )


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
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_FILE_TYPE",
                "message": "File extension is not a recognized image format.",
            },
        )

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_FILE_TYPE",
                "message": "File exceeds 10 MB limit.",
            },
        )

    processed_img, metadata = preprocess_image(image_bytes)

    try:
        results = get_reader().readtext(processed_img)
    except Exception:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "OCR_ENGINE_TIMEOUT",
                "message": "PyTorch worker thread experienced resource starvation.",
            },
        )

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

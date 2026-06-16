# Engineer A — Backend Core: Step-by-Step Guide

> **Scope:** Set up, implement, run, and deploy the FastAPI OCR service defined in `main.py`.
> All steps assume you start from a clean clone of this repository.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project File Overview](#2-project-file-overview)
3. [Set Up the Python Environment](#3-set-up-the-python-environment)
4. [Configure Environment Variables](#4-configure-environment-variables)
5. [Understand `main.py` — Section by Section](#5-understand-mainpy--section-by-section)
   - 5.1 [Settings (pydantic-settings)](#51-settings-pydantic-settings)
   - 5.2 [EasyOCR Reader (Lazy Singleton)](#52-easyocr-reader-lazy-singleton)
   - 5.3 [OpenCV Vision Pipeline](#53-opencv-vision-pipeline)
   - 5.4 [FastAPI Application & Endpoints](#54-fastapi-application--endpoints)
6. [Run the Service Locally](#6-run-the-service-locally)
7. [Test the API](#7-test-the-api)
   - 7.1 [Interactive Docs (Swagger UI)](#71-interactive-docs-swagger-ui)
   - 7.2 [Health Check with curl](#72-health-check-with-curl)
   - 7.3 [Document Scan with curl](#73-document-scan-with-curl)
   - 7.4 [Validate Error Responses](#74-validate-error-responses)
8. [Lint with Ruff](#8-lint-with-ruff)
9. [Build and Run with Docker](#9-build-and-run-with-docker)
10. [Deploy to a PaaS (Render / Railway / Zeabur)](#10-deploy-to-a-paas-render--railway--zeabur)
11. [Freeze the API Contract for Engineer B](#11-freeze-the-api-contract-for-engineer-b)

---

## 1. Prerequisites

| Tool | Minimum Version | Purpose |
|---|---|---|
| Python | 3.10 | Runtime |
| `uv` | 0.4+ | Fast virtual environment & package installer |
| Docker Desktop | 24+ | Container build & local image testing |
| `curl` (or Postman) | any | Manual API testing |
| `ruff` | 0.4+ | Linter / style enforcer |

Install `uv` if you do not already have it:

```bash
# macOS / Linux
curl -Lsf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

---

## 2. Project File Overview

After cloning the repository, the files you own as Engineer A are:

```
OPSfinal/
├── main.py              ← FastAPI application (your primary deliverable)
├── requirements.txt     ← All project dependencies (backend section is yours)
├── .env.example         ← Environment variable template
├── .env                 ← Your local config (never commit this)
└── Dockerfile           ← Multi-stage container build
```

`gui.py` and the GUI section of `requirements.txt` belong to Engineer B. You do not need to touch them.

---

## 3. Set Up the Python Environment

```bash
# 1. Create a virtual environment in .venv/
uv venv

# 2a. Activate (Linux / macOS)
source .venv/bin/activate

# 2b. Activate (Windows — PowerShell)
.venv\Scripts\activate

# 3. Install all backend dependencies
uv pip install -r requirements.txt
```

> **Note:** `easyocr` will download PyTorch (~700 MB) on first install.
> On a slow connection this takes several minutes — run it once and leave the `.venv` in place.

---

## 4. Configure Environment Variables

Copy the template and fill in values for your machine:

```bash
cp .env.example .env
```

Open `.env` and confirm these keys are set:

```env
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
EASYOCR_MODEL_STORAGE=./models   # directory where EasyOCR caches model weights
```

The `Settings` class in `main.py` reads this file automatically via `pydantic-settings`.
You do not need to export these manually — just having the `.env` file present is enough.

---

## 5. Understand `main.py` — Section by Section

### 5.1 Settings (pydantic-settings)

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    EASYOCR_MODEL_STORAGE: str = "./models"

settings = Settings()
```

- `BaseSettings` reads values from `.env` first, then falls back to the defaults.
- `extra="ignore"` means extra keys in `.env` (e.g., LINE Bot tokens) are silently skipped — the backend never errors on unknown keys.

---

### 5.2 EasyOCR Reader (Lazy Singleton)

```python
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
```

**Why lazy?**
EasyOCR loads a PyTorch model (~300 MB into RAM). Loading it at import time would crash the process if the model directory is missing, and would slow down every `uvicorn --reload` restart during development.

Loading it on the first real request means the server starts instantly, and the model is ready once the first scan arrives.

**First-time cold start:**
The first call to `GET /api/v1/scan` after a clean install will download model weights into `./models/`. Subsequent calls reuse the cached files.

---

### 5.3 OpenCV Vision Pipeline

The pipeline runs inside `preprocess_image(image_bytes)` and consists of five sequential steps:

| Step | OpenCV Call | Purpose |
|---|---|---|
| 1 — Grayscale | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` | Remove color information to reduce noise |
| 2 — Gaussian Blur | `cv2.GaussianBlur(gray, (5, 5), 0)` | Suppress high-frequency pixel noise |
| 3 — Canny Edge | `cv2.Canny(blurred, 75, 200)` | Detect document boundary edges |
| 4 — Contour Detection | `cv2.findContours(...)` | Find the largest closed quadrilateral |
| 5 — Perspective Warp | `cv2.warpPerspective(...)` | Flatten a skewed document into an orthographic view |

**Corner ordering (`_order_points`):**
Before warping, the four detected corner points must be arranged in a canonical order: top-left → top-right → bottom-right → bottom-left. This is achieved by sorting on the sum and difference of (x, y) coordinates.

**Error path:**
If no contour with exactly four vertices is found, the function raises `HTTPException(422, "DOCUMENT_CONTOUR_NOT_FOUND")`. This means the endpoint only succeeds for images that clearly show a flat, four-cornered document.

---

### 5.4 FastAPI Application & Endpoints

#### `GET /health`

```python
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
```

Used by load balancers (Render, Kubernetes) to verify the container is alive. Always returns 200.

---

#### `POST /api/v1/scan`

```python
@app.post("/api/v1/scan")
async def scan_document(
    file: Annotated[UploadFile, File()],
    min_confidence: Annotated[float, Form()] = 0.5,
):
```

**Request flow:**

```
1. Validate file extension  →  400 INVALID_FILE_TYPE if unsupported
2. Validate file size       →  400 INVALID_FILE_TYPE if > 10 MB
3. preprocess_image()       →  422 DOCUMENT_CONTOUR_NOT_FOUND if no quad contour
4. get_reader().readtext()  →  503 OCR_ENGINE_TIMEOUT if PyTorch fails
5. Filter by min_confidence
6. Return JSON
```

**Why `async`?**
`file.read()` is an I/O operation. Declaring the endpoint `async` lets Uvicorn's event loop serve other requests while waiting for the file bytes to be fully buffered — important when clients are on slow connections.

---

## 6. Run the Service Locally

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Expected output:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [xxxxx] using WatchFiles
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

`--reload` watches for file changes and restarts automatically — useful during development.
Remove it for production.

---

## 7. Test the API

### 7.1 Interactive Docs (Swagger UI)

Open your browser at:

```
http://127.0.0.1:8000/docs
```

This gives you a live, in-browser form to try both endpoints without writing any code.

---

### 7.2 Health Check with curl

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "timestamp": "2026-06-16T07:00:00Z"
}
```

---

### 7.3 Document Scan with curl

Replace `document.jpg` with a real image file on your machine that clearly shows a flat, four-cornered document (receipt, A4 sheet, business card on a table, etc.):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/scan \
  -F "file=@document.jpg" \
  -F "min_confidence=0.5"
```

Expected response shape:

```json
{
  "success": true,
  "metadata": {
    "width": 1920,
    "height": 1080,
    "processed_dimensions": "800x600"
  },
  "predictions": [
    {
      "text": "INVOICE",
      "confidence": 0.9942,
      "bounding_box": [[10, 10], [120, 10], [120, 40], [10, 40]]
    }
  ]
}
```

---

### 7.4 Validate Error Responses

**400 — wrong file type:**

```bash
curl -X POST http://127.0.0.1:8000/api/v1/scan \
  -F "file=@document.pdf"
```

Expected:

```json
{"detail": {"error": "INVALID_FILE_TYPE", "message": "File extension is not a recognized image format."}}
```

**422 — no document contour found:**

Send a photo of a plain scene with no clear rectangular document boundary.
The response will be:

```json
{"detail": {"error": "DOCUMENT_CONTOUR_NOT_FOUND", "message": "OpenCV cannot resolve four valid corner vertices."}}
```

---

## 8. Lint with Ruff

`ruff` enforces strict PEP 8 and catches common bugs. Run it before every commit.

```bash
# Install ruff (if not already installed)
uv pip install ruff

# Check main.py for violations
ruff check main.py

# Auto-fix safe violations
ruff check --fix main.py

# Check and fix the entire project
ruff check --fix .
```

To enforce ruff as a pre-commit gate, add a `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I"]
```

---

## 9. Build and Run with Docker

### Build the image

```bash
docker build -t ocr-service:latest .
```

The multi-stage `Dockerfile` works as follows:

| Stage | What it does |
|---|---|
| `builder` | Installs `uv`, filters out GUI-only packages, builds `.whl` files for all backend deps |
| final | Copies only the wheel files, installs them, adds required system libs (`libglib2.0-0`, `libgomp1`), copies `main.py` |

The final image contains **no build tools** (no `pip`, no compiler), keeping it minimal.

### Run the container

```bash
docker run -p 8000:8000 ocr-service:latest
```

Test it the same way as local:

```bash
curl http://localhost:8000/health
```

### Pass environment variables to the container

```bash
docker run -p 8000:8000 \
  -e EASYOCR_MODEL_STORAGE=/app/models \
  -v $(pwd)/models:/app/models \
  ocr-service:latest
```

Mounting `./models` as a volume avoids re-downloading the EasyOCR model weights every time the container starts.

---

## 10. Deploy to a PaaS (Render / Railway / Zeabur)

All three platforms detect a `Dockerfile` automatically and build from it on every `git push origin main`.

**Render — minimal `render.yaml`:**

```yaml
services:
  - type: web
    name: ocr-service
    runtime: docker
    plan: starter
    healthCheckPath: /health
    envVars:
      - key: EASYOCR_MODEL_STORAGE
        value: /app/models
```

**Railway / Zeabur:**

Both detect the `Dockerfile` without any config file. Set the following environment variables in the platform dashboard:

| Key | Value |
|---|---|
| `APP_HOST` | `0.0.0.0` |
| `APP_PORT` | `8000` |
| `EASYOCR_MODEL_STORAGE` | `/app/models` |

---

## 11. Freeze the API Contract for Engineer B

Once the service passes local testing, communicate the following frozen contract to Engineer B so they can begin development independently.

### Endpoint

```
POST /api/v1/scan
Content-Type: multipart/form-data
```

### Request Parameters

| Field | Type | Required | Notes |
|---|---|:---:|---|
| `file` | Binary | Yes | `.png`, `.jpg`, `.jpeg` only; max 10 MB |
| `min_confidence` | Float | No | Range `0.0–1.0`; default `0.5` |

### Success Response — `200 OK`

```json
{
  "success": true,
  "metadata": {
    "width": 1920,
    "height": 1080,
    "processed_dimensions": "800x600"
  },
  "predictions": [
    {
      "text": "INVOICE",
      "confidence": 0.9942,
      "bounding_box": [[10, 10], [120, 10], [120, 40], [10, 40]]
    }
  ]
}
```

### Error Responses

| HTTP Status | `error` Tag | Trigger |
|---|---|---|
| `400` | `INVALID_FILE_TYPE` | File extension not `.png`/`.jpg`/`.jpeg`, or file > 10 MB |
| `422` | `DOCUMENT_CONTOUR_NOT_FOUND` | OpenCV cannot detect four document corners |
| `503` | `OCR_ENGINE_TIMEOUT` | PyTorch model worker failed or timed out |

All error bodies follow the structure:

```json
{
  "detail": {
    "error": "<TAG>",
    "message": "<human-readable description>"
  }
}
```

### Local base URL (for Engineer B's `.env`)

```
OCR_SERVICE_URL=http://127.0.0.1:8000
```

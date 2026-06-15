# OPSfinal

# OPSfinal — Smart Document Scanner & OCR Service

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.110%2B-009688.svg?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-v4.9%2B-5C3EE8.svg?style=flat&logo=opencv)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, production-ready OCR microservice combining an **OpenCV computer vision pipeline** with an **EasyOCR (PyTorch) deep learning engine** for intelligent document digitization.

Built around a modular design philosophy — the OCR processing logic is encapsulated as a standalone shared backend, callable by multiple frontend interfaces:

| Client Type | Use Case |
| :--- | :--- |
| 🖥️ **Local Desktop GUI** | Drag-and-drop image upload with real-time OCR results displayed on screen |
| 🤖 **LINE Bot Module** | User sends an image in chat; OCR is triggered automatically and the extracted text is replied |
| 🌐 **Web App / Other Bots** | Any HTTP client capable of sending multipart/form-data requests |

---

## 📋 Table of Contents

1. [Modular Architecture Overview](#-modular-architecture-overview)
2. [Core Module: OCR Processing Engine](#️-core-module-ocr-processing-engine)
3. [Client Module A: Local Desktop GUI](#️-client-module-a-local-desktop-gui)
4. [Client Module B: LINE Bot Integration](#-client-module-b-line-bot-integration)
5. [API Specification](#-api-specification)
6. [Tech Stack](#-tech-stack)
7. [Development Setup](#-development-setup)
8. [CI/CD & Deployment](#-cicd--deployment)
9. [Error Handling](#️-error-handling)

---

## 🧩 Modular Architecture Overview

This project follows a **"one core, many frontends"** architecture:

```
┌─────────────────────────────────────────────────────┐
│                    Client Layer                      │
│                                                     │
│   🖥️ Local Desktop GUI        🤖 LINE Bot           │
│   (GUI / CLI)                 (Webhook Handler)     │
└───────────────────┬─────────────────┬───────────────┘
                    │  HTTP POST      │  HTTP POST
                    ▼                 ▼
┌─────────────────────────────────────────────────────┐
│             Core Layer: FastAPI OCR Service          │
│                                                     │
│   POST /api/v1/scan                                 │
│                                                     │
│   ┌──────────────────┐    ┌──────────────────────┐  │
│   │  OpenCV Pipeline │ -> │   EasyOCR Engine     │  │
│   │  Perspective Fix │    │   zh-TW / EN OCR     │  │
│   └──────────────────┘    └──────────────────────┘  │
│                                                     │
│   Returns JSON (text + confidence + bounding_box)   │
└─────────────────────────────────────────────────────┘
```

---

## ⚙️ Core Module: OCR Processing Engine

The OCR engine is the shared backend of this project. All client modules communicate with it exclusively through the HTTP API.

### Vision Processing Pipeline (OpenCV)

Each image submitted to the service is processed through the following sequential steps:

1. **Grayscale Conversion** — Strips color information to reduce computational load.
2. **Gaussian Blur** — Suppresses high-frequency sensor noise.
3. **Canny Edge Detection** — Defines the boundaries of the document.
4. **Contour Detection** (`cv2.findContours`) — Locates the largest quadrilateral contour matching a document profile.
5. **Perspective Transform** (`cv2.warpPerspective`) — Rectifies a skewed document into a flat, orthographic view.

### Text Recognition Engine (EasyOCR)

- **Bilingual Support:** Out-of-the-box recognition of mixed Traditional Chinese (`ch_tra`) and English (`en`) text.
- **Confidence Filtering:** Predictions below the configurable threshold (default ≥ 0.50) are automatically discarded.

### Design Principles

- **Fully In-Memory:** No local disk writes occur during standard processing flows, minimizing I/O overhead.
- **Headless Mode:** No GUI or display bindings required — runs cleanly inside minimal Linux containers.

---

## 🖥️ Client Module A: Local Desktop GUI

The desktop module lets users access OCR functionality directly from their machine, without going through a bot or browser.

**User Flow:**

```
User drags or selects an image file
              │
              ▼
Desktop Application (tkinter / PyQt / Electron)
              │  HTTP POST /api/v1/scan (local or remote service)
              ▼
OCR Core Service
              │  JSON response
              ▼
GUI displays extracted text and bounding box annotations
```

**Start the local service (for the GUI to connect to):**

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The GUI connects to `http://127.0.0.1:8000/api/v1/scan` by default. This can be overridden via `.env`.

---

## 🤖 Client Module B: LINE Bot Integration

The LINE Bot module allows users to send an image directly in a LINE conversation and receive the extracted text as a reply.

**User Flow:**

```
LINE user sends an image
              │  Webhook Event (image message)
              ▼
LINE Bot Webhook Handler
              │  Downloads image bytes
              │  HTTP POST /api/v1/scan (multipart/form-data)
              ▼
OCR Core Service
              │  JSON response
              ▼
LINE Bot replies with extracted text
```

**Webhook Handler Example:**

```python
from linebot.v3.messaging import TextMessage, ReplyMessageRequest

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event):
    # Download image content
    image_content = line_bot_api.get_message_content(event.message.id)
    image_bytes = b"".join(image_content.iter_content())

    # Call the OCR core service
    response = requests.post(
        "http://localhost:8000/api/v1/scan",
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        data={"min_confidence": 0.5}
    )
    result = response.json()

    # Compile extracted text and send reply
    texts = [p["text"] for p in result.get("predictions", [])]
    reply_text = "\n".join(texts) if texts else "Could not recognize any text. Please check image clarity."

    line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=reply_text)]
        )
    )
```

**Additional environment variables required for the LINE Bot module:**

```env
LINE_CHANNEL_SECRET=your_channel_secret
LINE_CHANNEL_ACCESS_TOKEN=your_access_token
OCR_SERVICE_URL=http://localhost:8000
```

---

## 🔌 API Specification

### GET /health — Health Check

Used by platform load balancers (Render, Kubernetes, etc.) to monitor container status.

**Response (200 OK):**

```json
{
  "status": "healthy",
  "timestamp": "2026-06-16T07:07:15Z"
}
```

---

### POST /api/v1/scan — Document Scan & Text Extraction

**Content-Type:** `multipart/form-data`

**Payload Parameters:**

| Parameter | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `file` | Binary (File) | ✅ | Target image (`.png` / `.jpg` / `.jpeg`). Max size: 10MB. |
| `min_confidence` | Float | ❌ | Confidence threshold. Range: `0.0`–`1.0`. Default: `0.5`. |

**Response (200 OK):**

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
    },
    {
      "text": "統一發票",
      "confidence": 0.9781,
      "bounding_box": [[150, 10], [300, 10], [300, 40], [150, 40]]
    }
  ]
}
```

Interactive API docs (development): [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🛠 Tech Stack

| Component | Technology |
| :--- | :--- |
| Runtime | Python 3.10 / 3.11 / 3.12 |
| API Layer | FastAPI + Uvicorn (ASGI) |
| Computer Vision Engine | `opencv-python-headless` |
| Deep Learning Engine | `easyocr` (PyTorch-backed) |
| Data Structures | `numpy` |
| Environment Management | `pydantic-settings` + `python-dotenv` |
| Dependency Management | `uv` |

---

## 📦 Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Cyanlit/OPSfinal.git
cd OPSfinal
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
EASYOCR_MODEL_STORAGE=/app/models

# LINE Bot (only required when using the LINE Bot module)
LINE_CHANNEL_SECRET=
LINE_CHANNEL_ACCESS_TOKEN=
```

### 3. Install Dependencies with uv

```bash
# Create virtual environment
uv venv

# Activate (Linux / macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### 4. Start the OCR Core Service

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🚀 CI/CD & Deployment

### Docker Multi-Stage Build

The production stack uses multi-stage builds to strip build tooling from the final runtime layer, keeping image size minimal.

```dockerfile
# Stage 1: Build & Cache Wheels
FROM python:3.10-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY requirements.txt .
RUN uv pip compile requirements.txt -o requirements.txt
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Final Runtime Layer
FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Continuous Integration

- **Linter:** Strict PEP 8 enforcement via `ruff`.
- **PaaS Compatibility:** Optimized for Render, Railway, and Zeabur. Stateless container builds are triggered automatically on `git push origin main`.

---

## ⚠️ Error Handling

All errors conform to the RFC 7807 Problem Details standard:

| HTTP Status | Error Tag | Trigger Condition |
| :--- | :--- | :--- |
| `400 Bad Request` | `INVALID_FILE_TYPE` | File extension is not a recognized image format. |
| `422 Unprocessable` | `DOCUMENT_CONTOUR_NOT_FOUND` | OpenCV cannot resolve four valid corner vertices. |
| `503 Service Unavailable` | `OCR_ENGINE_TIMEOUT` | PyTorch worker thread experiences resource starvation. |

---

## 📝 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

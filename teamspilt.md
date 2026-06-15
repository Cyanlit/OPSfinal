# OPSfinal — Team Development Split (2 Engineers)

> Scope: OCR Core Service + Local Desktop GUI (LINE Bot excluded)

---

## Engineer A — Backend Core

Owns the service heart: image in, structured text out.

### Responsibilities

| Area | Details |
| :--- | :--- |
| FastAPI Application | `main.py`, routing, middleware, ASGI config |
| OpenCV Pipeline | Grayscale → Gaussian Blur → Canny Edge → Contour Detection → Perspective Transform |
| EasyOCR Integration | Model loading, confidence filtering, response formatting |
| API Endpoints | `GET /health`, `POST /api/v1/scan` |
| Environment Management | `pydantic-settings`, `.env` schema |
| Containerization | Dockerfile multi-stage build, system library dependencies |
| CI/CD | `ruff` linter, PaaS deployment config (Render / Railway / Zeabur) |

### Deliverable

A locally runnable HTTP service where `POST /api/v1/scan` accepts an image and returns a structured JSON response.

---

## Engineer B — Frontend Client (Local Desktop GUI)

Wraps Engineer A's API into a visible, interactive desktop experience.

### Responsibilities

| Area | Details |
| :--- | :--- |
| Desktop Application | Window framework (tkinter / PyQt / Electron — per team choice) |
| Image Input | Drag-and-drop upload + file picker dialog |
| API Integration | Calls `POST /api/v1/scan`, handles JSON response and error states |
| Results Display | Extracted text layout and rendering |
| Bounding Box Overlay | Draws labeled boxes over the source image |
| Configuration | `.env` toggle to switch between local and remote service URL |

### Deliverable

A desktop application that, when connected to Engineer A's service, runs the full scan-and-display flow end to end.

---

## Collaboration Contract

The **only coupling point** between the two engineers is the API contract.

Before development begins, Engineer A must freeze and document the following:

### Request Schema — `POST /api/v1/scan`

```
Content-Type: multipart/form-data

file            Binary (File)   Required   .png / .jpg / .jpeg, max 10MB
min_confidence  Float           Optional   Range: 0.0–1.0, default: 0.5
```

### Response Schema — `200 OK`

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

| HTTP Status | Error Tag | Condition |
| :--- | :--- | :--- |
| `400 Bad Request` | `INVALID_FILE_TYPE` | Unsupported file format |
| `422 Unprocessable` | `DOCUMENT_CONTOUR_NOT_FOUND` | OpenCV cannot detect 4 document corners |
| `503 Service Unavailable` | `OCR_ENGINE_TIMEOUT` | PyTorch worker thread starvation |

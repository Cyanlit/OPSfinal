# OCR Service — Frozen API Contract (Engineer B Reference)

> Status: **FROZEN** — Engineer A has verified all endpoints locally.
> Engineer B may begin GUI development against this contract.

---

## Base URL

```
OCR_SERVICE_URL=http://127.0.0.1:8000
```

Add this to your `.env` file. The value changes when deployed to a PaaS — update it then.

---

## Endpoints

### `GET /health`

Used to check if the service is alive before submitting a scan.

**Response — `200 OK`**

```json
{
  "status": "healthy",
  "timestamp": "2026-06-16T00:00:00Z"
}
```

---

### `POST /api/v1/scan`

Submit an image for document detection and OCR.

**Request** — `multipart/form-data`

| Field | Type | Required | Constraints |
|---|---|:---:|---|
| `file` | Binary | Yes | `.png` / `.jpg` / `.jpeg` only; max 10 MB |
| `min_confidence` | Float | No | `0.0–1.0`; default `0.5` |

**Success Response — `200 OK`**

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

| Field | Type | Notes |
|---|---|---|
| `success` | bool | Always `true` on 200 |
| `metadata.width` | int | Original image width in pixels |
| `metadata.height` | int | Original image height in pixels |
| `metadata.processed_dimensions` | string | `"{w}x{h}"` after perspective warp |
| `predictions[].text` | string | Recognised text string |
| `predictions[].confidence` | float | `0.0–1.0`, rounded to 4 decimal places |
| `predictions[].bounding_box` | array | 4 × `[x, y]` pixel coordinates (top-left → clockwise) |

**Error Responses**

All errors follow the same envelope:

```json
{
  "detail": {
    "error": "<TAG>",
    "message": "<human-readable description>"
  }
}
```

| HTTP Status | `error` Tag | When it occurs |
|---|---|---|
| `400` | `INVALID_FILE_TYPE` | Extension not `.png`/`.jpg`/`.jpeg`, or file > 10 MB |
| `422` | `DOCUMENT_CONTOUR_NOT_FOUND` | OpenCV cannot detect four document corners |
| `503` | `OCR_ENGINE_TIMEOUT` | PyTorch model worker failed or timed out |

---

## Quick Integration Checklist (Engineer B)

- [ ] Read `OCR_SERVICE_URL` from `.env` — never hard-code the URL
- [ ] Call `GET /health` on startup; show an error state in the GUI if it returns non-200
- [ ] Send `POST /api/v1/scan` as `multipart/form-data` — **not** JSON
- [ ] Handle all three error tags (`INVALID_FILE_TYPE`, `DOCUMENT_CONTOUR_NOT_FOUND`, `OCR_ENGINE_TIMEOUT`) with user-friendly messages
- [ ] `predictions` may be an empty list `[]` if no text passes `min_confidence` — handle that case

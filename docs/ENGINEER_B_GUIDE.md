# Engineer B — Desktop GUI Client: Step-by-Step Guide

> **Scope:** Set up, implement, run, and test the Tkinter desktop application defined in `gui.py`.
> All steps assume you start from a clean clone of this repository.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project File Overview](#2-project-file-overview)
3. [Set Up the Python Environment](#3-set-up-the-python-environment)
4. [Configure Environment Variables](#4-configure-environment-variables)
5. [Understand `gui.py` — Section by Section](#5-understand-guipy--section-by-section)
   - 5.1 [Configuration & Constants](#51-configuration--constants)
   - 5.2 [Drag-and-Drop Setup (`tkinterdnd2`)](#52-drag-and-drop-setup-tkinterdnd2)
   - 5.3 [UI Layout — Three Regions](#53-ui-layout--three-regions)
   - 5.4 [Image Loading & Preview](#54-image-loading--preview)
   - 5.5 [Scan Flow — Threading Pattern](#55-scan-flow--threading-pattern)
   - 5.6 [Error Handling & Friendly Messages](#56-error-handling--friendly-messages)
   - 5.7 [Bounding Box Overlay](#57-bounding-box-overlay)
   - 5.8 [Grouped Results Display](#58-grouped-results-display)
6. [Understand `prediction_utils.py`](#6-understand-prediction_utilspy)
7. [Run the GUI Locally](#7-run-the-gui-locally)
8. [End-to-End Test Checklist](#8-end-to-end-test-checklist)
9. [Extend the GUI (Optional Tasks)](#9-extend-the-gui-optional-tasks)
10. [API Contract Reference (from Engineer A)](#10-api-contract-reference-from-engineer-a)

---

## 1. Prerequisites

| Tool | Minimum Version | Purpose |
|---|---|---|
| Python | 3.10 | Runtime |
| `uv` | 0.4+ | Fast virtual environment & package installer |
| Engineer A's service | running on port 8000 | Backend for OCR calls |

The GUI depends on these additional packages beyond the backend stack:

| Package | Purpose |
|---|---|
| `Pillow` | Open and display images inside Tkinter canvases |
| `requests` | HTTP calls to `POST /api/v1/scan` and `GET /health` |
| `tkinterdnd2` | Optional — enables file drag-and-drop onto the canvas |

Install `uv` if you do not already have it:

```bash
# macOS / Linux
curl -Lsf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

---

## 2. Project File Overview

After cloning the repository, the files you own as Engineer B are:

```
OPSfinal/
├── gui.py                ← Desktop application (your primary deliverable)
├── prediction_utils.py   ← Shared reading-order sort utility (owned by you, used by LINE Bot too)
├── start_gui.bat         ← Windows one-click launcher
├── requirements.txt      ← GUI section is yours (Pillow, requests, tkinterdnd2)
└── .env                  ← Your local config (never commit this)
```

`main.py`, `Dockerfile`, and the backend section of `requirements.txt` belong to Engineer A. You do not need to touch them.

---

## 3. Set Up the Python Environment

```bash
# 1. Create a virtual environment in .venv/
uv venv

# 2a. Activate (Linux / macOS)
source .venv/bin/activate

# 2b. Activate (Windows — PowerShell)
.venv\Scripts\activate

# 3. Install all dependencies
uv pip install -r requirements.txt
```

> **Note on `tkinterdnd2`:** This package adds native drag-and-drop support. It is optional — the GUI detects its presence at runtime and falls back gracefully. If installation fails on your platform, remove the line from `requirements.txt` and proceed; a click-to-browse file picker is always available.

---

## 4. Configure Environment Variables

Copy the template and fill in values for your machine:

```bash
cp .env.example .env
```

Open `.env` and confirm these keys are set:

```env
OCR_SERVICE_URL=http://127.0.0.1:8000   # Engineer A's local service address
MIN_CONFIDENCE=0.5                       # Default confidence threshold (0.0–1.0)
```

The GUI reads these at startup via `python-dotenv`. You can override them at runtime using the toolbar's service URL field and the confidence slider.

---

## 5. Understand `gui.py` — Section by Section

### 5.1 Configuration & Constants

```python
DEFAULT_SERVICE_URL = os.getenv("OCR_SERVICE_URL", "http://127.0.0.1:8000").strip()
DEFAULT_MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.5"))
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg"}
SCAN_TIMEOUT = 120
```

- `DEFAULT_SERVICE_URL` is read from `.env` once at import time and used to pre-fill the toolbar's URL entry. The user can change it per-session without restarting.
- `SCAN_TIMEOUT = 120` seconds: EasyOCR on a CPU can take 30–90 s on first run. This timeout is intentionally generous.

`ERROR_MESSAGES` maps each API error tag to a user-facing Chinese string:

```python
ERROR_MESSAGES: dict[str, str] = {
    "INVALID_FILE_TYPE":          "不支援的圖片格式，請使用 PNG 或 JPEG。",
    "FILE_TOO_LARGE":             "檔案超過 10 MB 上限，請壓縮或換一張較小的圖片。",
    "IMAGE_DECODE_FAILED":        "無法讀取圖片，檔案可能已損壞。",
    "DOCUMENT_CONTOUR_NOT_FOUND": "無法偵測文件邊界，請確認拍攝角度與光線。",
    "OCR_ENGINE_TIMEOUT":         "OCR 引擎忙碌或記憶體不足，請稍後再試。",
}
```

---

### 5.2 Drag-and-Drop Setup (`tkinterdnd2`)

```python
def create_root() -> tk.Tk:
    try:
        from tkinterdnd2 import TkinterDnD
        return TkinterDnD.Tk()
    except Exception:
        return tk.Tk()
```

`TkinterDnD.Tk()` replaces the standard `tk.Tk()` root with one that understands OS drag events. If `tkinterdnd2` is not installed, a regular `Tk` window is created and the drop target registration in `_register_drop_target()` is silently skipped.

The `_enable_drag_drop()` method sets `self._drag_drop_enabled` and saves `DND_FILES` so that `_register_drop_target()` can bind the `<<Drop>>` event:

```python
self._canvas_original.drop_target_register(self._dnd_files)
self._canvas_original.dnd_bind("<<Drop>>", self._on_drop)
```

`_on_drop` strips curly braces that Windows wraps around paths with spaces, then delegates to `_load_image()`.

---

### 5.3 UI Layout — Three Regions

`_build_ui()` calls three builders in order:

```
┌─────────────────────────────────────────────────────────────────┐
│  Toolbar (_build_toolbar)                                       │
│  ┌ Row 1 ──────────────────────────────────────────────────┐   │
│  │ 服務位址: [___________________] [測試連線]               │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌ Row 2 ──────────────────────────────────────────────────┐   │
│  │ [選擇圖片] [開始掃描]  信心門檻: ——●—— 0.50             │   │
│  │                        [複製文字] [儲存文字]             │   │
│  └─────────────────────────────────────────────────────────┘   │
├───────────────────────────┬─────────────────────────────────────┤
│  Left: 影像預覽           │  Right: 辨識結果                   │
│  (_build_main_pane)       │                                     │
│  ┌ Tab: 原圖 ──────────┐  │  [Summary label]                   │
│  │ _canvas_original    │  │  ┌ 原始結果 ────────────────────┐  │
│  └─────────────────────┘  │  │ _raw_text_box (scrolled)     │  │
│  ┌ Tab: 校正後標框 ────┐  │  └─────────────────────────────┘  │
│  │ _canvas_processed   │  │  ┌ 套用行/列結果 ───────────────┐  │
│  └─────────────────────┘  │  │ _grouped_text_box (scrolled) │  │
│                           │  └─────────────────────────────┘  │
├───────────────────────────┴─────────────────────────────────────┤
│  Status bar (_build_status_bar):  就緒  [========== ]          │
└─────────────────────────────────────────────────────────────────┘
```

All widgets are stored as instance attributes (`self._scan_btn`, `self._canvas_original`, etc.) so event handlers can update them from any method.

---

### 5.4 Image Loading & Preview

`_load_image(path)` is the single entry point for both the file picker and drag-and-drop:

1. Validates the file extension against `ALLOWED_SUFFIXES`.
2. Opens the image with `PIL.Image.open(path).convert("RGB")`.
3. Stores it in `self._original_image`, clears any stale scan state.
4. Enables the "開始掃描" button.
5. Calls `_redraw_original()` to paint the canvas.

`_fit_image_to_canvas(image, canvas)` scales the image to fit the current canvas dimensions using `Image.LANCZOS` (high-quality downscaling). The resulting `ImageTk.PhotoImage` is kept alive as `self._photo_original` — Tkinter's garbage collector would destroy an unreferenced `PhotoImage` immediately, causing a blank canvas.

`_on_original_canvas_configure` fires on window resize and redraws both canvases to keep the preview sharp at any window size.

---

### 5.5 Scan Flow — Threading Pattern

Tkinter's main loop is single-threaded. Blocking it with a `requests.post()` call would freeze the window for the entire OCR duration (potentially 60+ seconds). The fix is a daemon thread:

```python
def _start_scan(self) -> None:
    self._scan_btn.configure(state="disabled")
    self._progress.start(12)                   # start spinner
    threading.Thread(target=self._do_scan, daemon=True).start()

def _do_scan(self) -> None:
    # runs in daemon thread — no Tkinter calls allowed here
    resp = requests.post(url, files={...}, data={...}, timeout=SCAN_TIMEOUT)
    self.root.after(0, self._on_scan_done, result)  # schedule UI update on main thread
```

**Rule:** Never call Tkinter widget methods from a background thread. Use `self.root.after(0, callback, arg)` to schedule all UI updates back on the main loop. `_on_scan_done` and `_on_scan_error` both follow this pattern.

---

### 5.6 Error Handling & Friendly Messages

`_format_http_error(exc)` parses the structured error body that Engineer A's service returns:

```python
detail = response.json().get("detail", {})
tag = detail.get("error", "HTTP_ERROR")
friendly = ERROR_MESSAGES.get(tag)
```

If the tag is in `ERROR_MESSAGES`, the friendly Chinese message is shown. Otherwise the raw `error`/`message` pair is displayed. This mapping is the only place where the two engineers' work is tightly coupled — if Engineer A adds a new error tag, add it to `ERROR_MESSAGES` here.

---

### 5.7 Bounding Box Overlay

`_build_processed_image()` constructs the annotated preview after a successful scan:

```python
for index, pred in enumerate(self._predictions, start=1):
    pts = [tuple(p) for p in pred["bounding_box"]]
    draw.polygon(pts, outline="#1f8f4e", width=2)
    label = f'{index}. {pred["text"]} ({pred["confidence"]:.2f})'
    draw.rectangle([x0, label_y, ...], fill="#1f8f4e")
    draw.text((x0 + 2, label_y + 1), label, fill="#ffffff", font=font)
```

The canvas dimensions come from `metadata["processed_dimensions"]` (e.g. `"800x600"`), parsed as `pw, ph = map(int, dims.lower().split("x"))`. Each bounding box is a list of four `[x, y]` points returned by the API.

The result is a `PIL.Image` stored in `self._processed_image`. `_redraw_processed()` calls `_fit_image_to_canvas()` to render it at the current window size.

---

### 5.8 Grouped Results Display

The API response includes a `grouped` field alongside `predictions`:

```json
{
  "grouped": {
    "orientation": "row",
    "rows": [{"block": 0, "text": "INVOICE 2024-001", "confidence": 0.9621}],
    "columns": [{"block": 0, "text": "INVOICE", "confidence": 0.9942}]
  }
}
```

`_on_scan_done` reads the `orientation` key to decide which list to display in the "套用行／列結果" panel:

| `orientation` | Uses list | Label shown |
|---|---|---|
| `"row"` | `rows` | 方向：橫排（行） |
| `"column"` | `columns` | 方向：直排（列） |
| anything else | `rows` | 方向：混合（以行為主） |

The `DENOISE_LABELS` dict maps the `denoising` field in `metadata` to a Chinese label for the summary line (e.g. `"medium"` → `"中度"`).

---

## 6. Understand `prediction_utils.py`

`sort_predictions_reading_order(predictions)` re-orders the API's `predictions` list into natural reading order (top-to-bottom, left-to-right within a line). It is **not** called automatically in `gui.py` — it is available as a utility if you want to add a "sort by reading order" toggle, or if a future feature needs an ordered transcript.

The algorithm:
1. For each prediction, extract `min_x`, `min_y`, and `height` from the bounding box.
2. Compute a `row_tolerance` from the median box height.
3. Assign each prediction a row index: `row = int(min_y // row_tolerance)`.
4. Sort by `(row, min_x)`.

This file is also imported by `line_bot.py`, so changes here affect both the GUI and the LINE Bot.

---

## 7. Run the GUI Locally

### Prerequisites

Engineer A's service must be running before you launch the GUI. Start it in a separate terminal:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Or double-click **`start_server.bat`** on Windows.

### Launch the GUI

```bash
# Activate the environment first (if not already active)
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS / Linux

python gui.py
```

Or double-click **`start_gui.bat`** on Windows. The batch file activates `.venv` automatically.

### What to expect on first launch

1. The window opens at 960×620 pixels.
2. The status bar shows "就緒".
3. After ~200 ms, the GUI calls `GET /health` automatically.
   - If the backend is running: status bar updates to "服務已連線（healthy）".
   - If the backend is not running: status bar shows "服務未連線 — 無法連線到http://127.0.0.1:8000/health…".

---

## 8. End-to-End Test Checklist

Work through these scenarios after every significant change:

### A. Happy path

- [ ] Open a `.jpg` photo of a receipt or document with clear edges.
- [ ] Click "開始掃描" — spinner starts, button disables.
- [ ] Scan completes; "校正後標框" tab opens automatically.
- [ ] Bounding boxes visible over the corrected image.
- [ ] "原始結果" panel shows extracted text.
- [ ] "套用行／列結果" panel shows grouped lines.
- [ ] Summary shows correct original dimensions and result count.

### B. File validation

- [ ] Try dragging a `.gif` file — app shows "格式錯誤" dialog.
- [ ] Try dragging a file larger than 10 MB — API returns `FILE_TOO_LARGE`, app shows the Chinese error message.

### C. Service connectivity

- [ ] Stop the backend. Click "測試連線" — error dialog appears.
- [ ] Click "開始掃描" — scan fails with connection error, spinner stops, button re-enables.

### D. Confidence threshold

- [ ] Move the slider to 0.90. Scan an image with mixed-confidence results — low-confidence tokens disappear.
- [ ] Move the slider to 0.0. All detected tokens appear.

### E. Copy and save

- [ ] After a successful scan, click "複製文字" — paste into Notepad to verify only the plain text (above the `---` divider) was copied.
- [ ] Click "儲存文字", choose a `.txt` path — file is created with UTF-8 encoding.

### F. Window resize

- [ ] Drag the window to a larger size — both image canvases rescale without going blank.
- [ ] Resize the PanedWindow divider between image and results panels.

---

## 9. Extend the GUI (Optional Tasks)

These are common follow-up improvements that keep within Engineer B's scope:

### Reading-order sort toggle

Add a checkbox in the toolbar:

```python
self._sort_var = tk.BooleanVar(value=False)
ttk.Checkbutton(row2, text="閱讀順序排序", variable=self._sort_var).pack(side=tk.LEFT)
```

In `_on_scan_done`, apply the sort before building the text box:

```python
if self._sort_var.get():
    from prediction_utils import sort_predictions_reading_order
    self._predictions = sort_predictions_reading_order(self._predictions)
```

### Export annotated image

After `_build_processed_image()` returns, add a "儲存標框圖" button that calls:

```python
self._processed_image.save(path)
```

### Switch between multiple service URLs

Replace the single URL entry with a `ttk.Combobox` pre-filled from a comma-separated `OCR_SERVICE_URLS` env var. When the user selects a URL from the dropdown, update `self._url_var`.

---

## 10. API Contract Reference (from Engineer A)

This is the frozen contract Engineer B consumes. Do not modify Engineer A's service to suit the GUI — adapt the GUI to the contract.

### Endpoint

```
POST /api/v1/scan
Content-Type: multipart/form-data
```

### Request Parameters

| Field | Type | Required | Notes |
|---|:---:|:---:|---|
| `file` | Binary | Yes | `.png`, `.jpg`, `.jpeg` only; max 10 MB |
| `min_confidence` | Float | No | Range `0.0–1.0`; default `0.5` |

### Success Response — `200 OK`

```json
{
  "success": true,
  "metadata": {
    "width": 1920,
    "height": 1080,
    "processed_dimensions": "800x600",
    "noise_level": 3,
    "noise_score": 8.14,
    "denoising": "light"
  },
  "predictions": [
    {
      "text": "INVOICE",
      "confidence": 0.9942,
      "bounding_box": [[10, 10], [120, 10], [120, 40], [10, 40]]
    }
  ],
  "grouped": {
    "orientation": "row",
    "rows": [
      {"block": 0, "text": "INVOICE 2024-001", "confidence": 0.9621}
    ],
    "columns": [
      {"block": 0, "text": "INVOICE", "confidence": 0.9942}
    ]
  }
}
```

**Extended `metadata` fields** (present in all responses):

| Field | Type | Description |
|---|---|---|
| `noise_level` | int (0–9) | Denoising strength applied (0 = none) |
| `noise_score` | float | Raw noise estimate used to select the level |
| `denoising` | string | Human-readable label for `noise_level` (e.g. `"light"`, `"medium"`) |

### Error Responses

| HTTP Status | `error` Tag | Trigger |
|---|---|---|
| `400` | `INVALID_FILE_TYPE` | File extension not `.png`/`.jpg`/`.jpeg` |
| `400` | `FILE_TOO_LARGE` | File exceeds 10 MB |
| `400` | `INVALID_PARAMETER` | `min_confidence` outside 0.0–1.0 |
| `422` | `IMAGE_DECODE_FAILED` | Image bytes are corrupted or unreadable |
| `422` | `DOCUMENT_CONTOUR_NOT_FOUND` | *(legacy — service now falls back to full image)* |
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

### Health Check

```
GET /health
```

```json
{
  "status": "healthy",
  "timestamp": "2026-06-21T07:00:00Z",
  "line_configured": false
}
```

### Local base URL (for your `.env`)

```
OCR_SERVICE_URL=http://127.0.0.1:8000
```

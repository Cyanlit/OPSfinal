# OPSfinal — Smart Document Scanner & OCR Service

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.110%2B-009688.svg?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-v4.9%2B-5C3EE8.svg?style=flat&logo=opencv)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一個輕量化、可生產部署的 OCR 微服務核心，結合 **OpenCV 電腦視覺管線** 與 **EasyOCR（PyTorch）深度學習引擎**，實現智慧型文件數位化。

本服務以模組化設計為核心理念——OCR 處理邏輯獨立封裝，可作為共用後端被多種前端介面調用：

| 調用方式 | 使用情境 |
| :--- | :--- |
| 🖥️ **本地視窗應用程式** | 桌面端直接拖拉上傳、即時顯示辨識結果 |
| 🤖 **LINE Bot 模組** | 使用者傳送圖片後自動觸發掃描並回傳文字 |
| 🌐 **Web / 其他 Bot** | 任何可發送 HTTP multipart 請求的客戶端 |

---

## 📋 目錄

1. [模組架構概覽](#-模組架構概覽)
2. [核心模組：OCR 處理引擎](#-核心模組ocr-處理引擎)
3. [調用模組 A：本地視窗介面](#-調用模組-a本地視窗介面)
4. [調用模組 B：LINE Bot 整合](#-調用模組-bline-bot-整合)
5. [API 規格](#-api-規格)
6. [技術棧](#-技術棧)
7. [開發環境設置](#-開發環境設置)
8. [CI/CD 與部署策略](#-cicd-與部署策略)
9. [錯誤處理](#-錯誤處理)

---

## 🧩 模組架構概覽

本專案以「一個核心，多個前端」的模組化架構設計：

```
┌─────────────────────────────────────────────────────┐
│                  調用層（Frontend）                   │
│                                                     │
│   🖥️ 本地視窗介面          🤖 LINE Bot              │
│   (GUI / CLI)              (Webhook Handler)        │
└───────────────────┬─────────────────┬───────────────┘
                    │  HTTP POST      │  HTTP POST
                    ▼                 ▼
┌─────────────────────────────────────────────────────┐
│              核心層：FastAPI OCR Service              │
│                                                     │
│   POST /api/v1/scan                                 │
│                                                     │
│   ┌──────────────┐      ┌──────────────────────┐   │
│   │ OpenCV 管線  │ ───> │   EasyOCR 引擎       │   │
│   │ 透視校正     │      │   繁中 / 英文辨識     │   │
│   └──────────────┘      └──────────────────────┘   │
│                                                     │
│   回傳 JSON（text + confidence + bounding_box）     │
└─────────────────────────────────────────────────────┘
```

---

## ⚙️ 核心模組：OCR 處理引擎

OCR 引擎是整個專案的共用後端，所有調用模組皆透過 HTTP API 與它溝通。

### 視覺處理管線（OpenCV）

圖片進入服務後，依序執行以下步驟：

1. **灰階轉換** — 去除色彩資訊，降低運算量。
2. **高斯模糊** — 濾除高頻感測器雜訊。
3. **Canny 邊緣偵測** — 定義文件邊界。
4. **輪廓偵測**（`cv2.findContours`）— 定位最大四邊形輪廓。
5. **透視變換**（`cv2.warpPerspective`）— 將傾斜的文件校正為正視圖。

### 文字辨識引擎（EasyOCR）

- **雙語支援：** 繁體中文（`ch_tra`）與英文（`en`）混排。
- **信心度過濾：** 自動捨棄低於閾值（預設 ≥ 0.50）的辨識結果。

### 服務設計原則

- **全記憶體處理：** 標準流程中不寫入本地磁碟，最小化 I/O 開銷。
- **無頭模式：** 不依賴任何 GUI 或顯示綁定，可在最小化 Linux 容器中運行。

---

## 🖥️ 調用模組 A：本地視窗介面

本地視窗模組讓使用者在桌面端直接使用 OCR 功能，無需透過 Bot 或網頁。

**使用流程：**

```
使用者拖拉或選擇圖片
        │
        ▼
本地視窗應用程式（tkinter / PyQt / Electron）
        │  HTTP POST /api/v1/scan（連接本地或遠端服務）
        ▼
OCR 核心服務
        │  JSON 回傳
        ▼
視窗介面顯示辨識文字與 Bounding Box 標注
```

**啟動本地服務（供視窗介面連接）：**

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

視窗介面預設連接 `http://127.0.0.1:8000/api/v1/scan`，可透過 `.env` 調整。

---

## 🤖 調用模組 B：LINE Bot 整合

LINE Bot 模組讓使用者在 LINE 對話中直接傳送圖片，自動觸發 OCR 並回傳辨識結果。

**使用流程：**

```
LINE 使用者傳送圖片
        │  Webhook Event (image message)
        ▼
LINE Bot Webhook Handler
        │  下載圖片 bytes
        │  HTTP POST /api/v1/scan（multipart/form-data）
        ▼
OCR 核心服務
        │  JSON 回傳
        ▼
LINE Bot 將辨識文字回覆給使用者
```

**Webhook Handler 範例片段：**

```python
from linebot.v3.messaging import TextMessage, ReplyMessageRequest

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event):
    # 下載圖片
    image_content = line_bot_api.get_message_content(event.message.id)
    image_bytes = b"".join(image_content.iter_content())

    # 呼叫 OCR 核心服務
    response = requests.post(
        "http://localhost:8000/api/v1/scan",
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        data={"min_confidence": 0.5}
    )
    result = response.json()

    # 整理辨識文字並回覆
    texts = [p["text"] for p in result.get("predictions", [])]
    reply_text = "\n".join(texts) if texts else "無法辨識文字，請確認圖片清晰度。"

    line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=reply_text)]
        )
    )
```

**LINE Bot 所需額外環境變數：**

```env
LINE_CHANNEL_SECRET=your_channel_secret
LINE_CHANNEL_ACCESS_TOKEN=your_access_token
OCR_SERVICE_URL=http://localhost:8000
```

---

## 🔌 API 規格

### GET /health — 健康狀態確認

供負載平衡器（Render、Kubernetes 等）監控容器狀態。

**Response（200 OK）：**

```json
{
  "status": "healthy",
  "timestamp": "2026-06-16T07:07:15Z"
}
```

---

### POST /api/v1/scan — 文件掃描與文字萃取

**Content-Type：** `multipart/form-data`

**Payload 參數：**

| 參數 | 型別 | 必填 | 說明 |
| :--- | :--- | :---: | :--- |
| `file` | Binary (File) | ✅ | 目標圖片（`.png` / `.jpg` / `.jpeg`），最大 10MB。 |
| `min_confidence` | Float | ❌ | 信心度閾值，範圍 `0.0`–`1.0`，預設 `0.5`。 |

**Response（200 OK）：**

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

互動式 API 文件（開發環境）：[http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🛠 技術棧

| 項目 | 技術 |
| :--- | :--- |
| 運行環境 | Python 3.10 / 3.11 / 3.12 |
| API 層 | FastAPI + Uvicorn（ASGI） |
| 電腦視覺引擎 | `opencv-python-headless` |
| 深度學習引擎 | `easyocr`（依賴 PyTorch） |
| 資料結構 | `numpy` |
| 環境管理 | `pydantic-settings` + `python-dotenv` |
| 依賴管理 | `uv` |

---

## 📦 開發環境設置

### 1. 複製儲存庫

```bash
git clone https://github.com/Cyanlit/OPSfinal.git
cd OPSfinal
```

### 2. 環境變數設定

在根目錄建立 `.env` 檔案：

```env
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
EASYOCR_MODEL_STORAGE=/app/models

# LINE Bot（僅使用 LINE Bot 模組時需填寫）
LINE_CHANNEL_SECRET=
LINE_CHANNEL_ACCESS_TOKEN=
```

### 3. 使用 uv 安裝環境

```bash
# 建立虛擬環境
uv venv

# 啟動（Linux / macOS）
source .venv/bin/activate

# 啟動（Windows）
.venv\Scripts\activate

# 安裝依賴
uv pip install -r requirements.txt
```

### 4. 啟動 OCR 核心服務

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🚀 CI/CD 與部署策略

### Docker 多階段建構

生產環境採用多階段建構，將編譯工具從執行層剝離，確保最小化映像體積。

```dockerfile
# Stage 1：建構與快取 Wheels
FROM python:3.10-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY requirements.txt .
RUN uv pip compile requirements.txt -o requirements.txt
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# Stage 2：最終執行層
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

### 持續整合

- **Linter：** 透過 `ruff` 嚴格執行 PEP 8 規範。
- **PaaS 相容性：** 針對 Render、Railway、Zeabur 最佳化，`git push origin main` 後自動觸發無狀態容器建構。

---

## ⚠️ 錯誤處理

所有錯誤遵循 RFC 7807 Problem Details 標準格式：

| HTTP 狀態碼 | 錯誤標籤 | 觸發條件 |
| :--- | :--- | :--- |
| `400 Bad Request` | `INVALID_FILE_TYPE` | 檔案格式不在支援清單中。 |
| `422 Unprocessable` | `DOCUMENT_CONTOUR_NOT_FOUND` | OpenCV 無法解析四個有效外框頂點。 |
| `503 Service Unavailable` | `OCR_ENGINE_TIMEOUT` | PyTorch 執行執行緒發生資源競爭。 |

---

## 📝 授權條款

本專案採用 MIT License 發布。詳見 [LICENSE](LICENSE) 文件。

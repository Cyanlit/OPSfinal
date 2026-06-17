"""LINE Messaging API — 傳圖掃描 OCR 並回覆文字。"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import requests
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    MessagingApiBlob,
    PushMessageRequest,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import ImageMessageContent, MessageEvent, TextMessageContent

from prediction_utils import sort_predictions_reading_order

logger = logging.getLogger(__name__)

HELP_TEXT = (
    "📄 智慧文件掃描 OCR Bot\n\n"
    "直接傳送一張文件照片（收據、筆記、發票），我會自動校正並辨識文字。\n\n"
    "輸入「說明」查看此訊息。"
)

SCAN_TIMEOUT = 300
REPLY_CHAR_LIMIT = 4800

SCAN_ERROR_MESSAGES = {
    "FILE_TOO_LARGE": "檔案超過 10 MB 上限，請換一張較小的圖片。",
    "OCR_ENGINE_TIMEOUT": "OCR 引擎忙碌或記憶體不足，請稍後再試。",
    "INVALID_PARAMETER": "掃描參數錯誤，請聯絡管理員。",
    "IMAGE_DECODE_FAILED": "無法讀取圖片，檔案可能損壞或格式不支援。",
}


@dataclass
class LineBotConfig:
    channel_secret: str
    channel_access_token: str
    ocr_service_url: str = "http://127.0.0.1:8000"
    min_confidence: float = 0.5


@dataclass
class LineBotService:
    config: LineBotConfig
    scan_fn: Callable[[bytes, float], dict] | None = None
    _configuration: Configuration = field(init=False, repr=False)
    _handler: WebhookHandler = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._configuration = Configuration(access_token=self.config.channel_access_token)
        self._handler = WebhookHandler(self.config.channel_secret)
        self._register_handlers()

    @property
    def is_configured(self) -> bool:
        return bool(self.config.channel_secret and self.config.channel_access_token)

    def _register_handlers(self) -> None:
        @self._handler.add(MessageEvent, message=TextMessageContent)
        def handle_text(event: MessageEvent):
            text = event.message.text.strip()
            if text in {"說明", "help", "Help", "?"}:
                self._reply_text(event.reply_token, [HELP_TEXT])
                return
            self._reply_text(
                event.reply_token,
                ["請直接傳送一張文件照片，我會幫你掃描並輸出文字。\n輸入「說明」查看使用方式。"],
            )

        @self._handler.add(MessageEvent, message=ImageMessageContent)
        def handle_image(event: MessageEvent):
            user_id = getattr(event.source, "user_id", None)
            if not user_id:
                self._reply_text(event.reply_token, ["無法取得使用者 ID，請再試一次。"])
                return

            message_id = event.message.id
            try:
                self._reply_text(event.reply_token, ["📄 收到圖片，正在掃描中，請稍候…"])
            except Exception:
                logger.exception("Failed to send immediate LINE reply")

            threading.Thread(
                target=self._process_image_background,
                args=(user_id, message_id),
                daemon=True,
            ).start()
            logger.info("Queued OCR job for user=%s message=%s", user_id, message_id)

    def handle_webhook(self, body: str, signature: str) -> None:
        if not signature:
            raise ValueError("Missing X-Line-Signature header")
        try:
            self._handler.handle(body, signature)
        except InvalidSignatureError as exc:
            raise ValueError("Invalid LINE signature") from exc

    def _process_image_background(self, user_id: str, message_id: str) -> None:
        try:
            logger.info("Downloading LINE image message_id=%s", message_id)
            with ApiClient(self._configuration) as api_client:
                blob_api = MessagingApiBlob(api_client)
                image_bytes = blob_api.get_message_content(message_id)

            logger.info("Running OCR (%d bytes)", len(image_bytes))
            result = self._scan_image(image_bytes)
            reply_text = self._format_scan_reply(result)
            self._push_text(user_id, self._split_text(reply_text))
            logger.info("OCR result pushed to user=%s", user_id)
        except requests.exceptions.ConnectionError:
            logger.exception("OCR service unreachable")
            self._push_text(user_id, ["OCR 服務未啟動，請確認後端正在運行（start_line.bat）。"])
        except requests.exceptions.ReadTimeout:
            logger.exception("OCR HTTP timeout")
            self._push_text(user_id, ["掃描逾時，圖片可能太大或首次載入模型較久，請再試一次。"])
        except requests.exceptions.HTTPError as exc:
            logger.exception("OCR HTTP error")
            self._push_text(user_id, [self._format_http_error(exc)])
        except ValueError as exc:
            logger.exception("OCR scan failed")
            self._push_text(user_id, [SCAN_ERROR_MESSAGES.get(str(exc), f"掃描失敗：{exc}")])
        except Exception:
            logger.exception("Failed to process LINE image")
            self._push_text(user_id, ["處理失敗，請稍後再試或換一張較清晰的照片。"])

    def _scan_image(self, image_bytes: bytes) -> dict:
        if self.scan_fn is not None:
            return self.scan_fn(image_bytes, self.config.min_confidence)

        response = requests.post(
            f"{self.config.ocr_service_url.rstrip('/')}/api/v1/scan",
            files={"file": ("line_image.jpg", image_bytes, "image/jpeg")},
            data={"min_confidence": self.config.min_confidence},
            timeout=SCAN_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def _format_http_error(self, exc: requests.exceptions.HTTPError) -> str:
        response = exc.response
        if response is None:
            return f"掃描失敗：{exc}"
        try:
            detail = response.json().get("detail", {})
            if isinstance(detail, dict):
                return f"掃描失敗：{detail.get('message', response.text)}"
        except Exception:
            pass
        return f"掃描失敗（HTTP {response.status_code}）"

    def _format_scan_reply(self, result: dict) -> str:
        predictions = sort_predictions_reading_order(result.get("predictions", []))
        if not predictions:
            return "未辨識到文字，請換張更清晰的照片，或調低 MIN_CONFIDENCE 再試。"

        lines = [pred["text"] for pred in predictions if pred.get("text", "").strip()]
        text = "\n".join(lines)
        meta = result.get("metadata", {})
        processed = meta.get("processed_dimensions", "-")
        denoise = meta.get("denoising")
        header = f"✅ 掃描完成（處理尺寸 {processed}"
        if denoise:
            header += f"，降噪 {denoise}"
        header += "）\n\n"
        return header + text

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= REPLY_CHAR_LIMIT:
            return [text]

        chunks: list[str] = []
        current = ""
        for line in text.splitlines(keepends=True):
            if len(current) + len(line) > REPLY_CHAR_LIMIT:
                if current.strip():
                    chunks.append(current.rstrip())
                current = line
            else:
                current += line
        if current.strip():
            chunks.append(current.rstrip())
        return chunks or [text[:REPLY_CHAR_LIMIT]]

    def _reply_text(self, reply_token: str, messages: list[str]) -> None:
        with ApiClient(self._configuration) as api_client:
            line_api = MessagingApi(api_client)
            line_api.reply_message(
                ReplyMessageRequest(
                    reply_token=reply_token,
                    messages=[TextMessage(text=msg) for msg in messages],
                )
            )

    def _push_text(self, user_id: str, messages: list[str]) -> None:
        with ApiClient(self._configuration) as api_client:
            line_api = MessagingApi(api_client)
            line_api.push_message(
                PushMessageRequest(
                    to=user_id,
                    messages=[TextMessage(text=msg) for msg in messages],
                )
            )

"""
Local Desktop GUI — Engineer B module.
Connects to the OCR Core Service (Engineer A) at POST /api/v1/scan.
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont, ImageTk

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from prediction_utils import sort_predictions_reading_order

DEFAULT_SERVICE_URL = os.getenv("OCR_SERVICE_URL", "http://127.0.0.1:8000").strip()
DEFAULT_MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.5"))
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg"}
SCAN_TIMEOUT = 120

ERROR_MESSAGES: dict[str, str] = {
    "INVALID_FILE_TYPE": "不支援的圖片格式，請使用 PNG 或 JPEG。",
    "FILE_TOO_LARGE": "檔案超過 10 MB 上限，請壓縮或換一張較小的圖片。",
    "IMAGE_DECODE_FAILED": "無法讀取圖片，檔案可能已損壞。",
    "INVALID_PARAMETER": "信心門檻必須介於 0.0 到 1.0 之間。",
    "DOCUMENT_CONTOUR_NOT_FOUND": "無法偵測文件邊界，請確認拍攝角度與光線。",
    "OCR_ENGINE_TIMEOUT": "OCR 引擎忙碌或記憶體不足，請稍後再試。",
}

DENOISE_LABELS: dict[str, str] = {
    "none": "無",
    "light": "輕度",
    "medium": "中度",
    "heavy": "重度",
}


def create_root() -> tk.Tk:
    try:
        from tkinterdnd2 import TkinterDnD

        return TkinterDnD.Tk()
    except Exception:
        return tk.Tk()


class OCRApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("智慧文件掃描 — 桌面客戶端")
        self.root.geometry("960x620")
        self.root.minsize(760, 480)
        self.root.configure(bg="#f0f2f5")

        self._image_path: Path | None = None
        self._original_image: Image.Image | None = None
        self._processed_image: Image.Image | None = None
        self._photo_original: ImageTk.PhotoImage | None = None
        self._photo_processed: ImageTk.PhotoImage | None = None
        self._predictions: list[dict] = []
        self._metadata: dict = {}
        self._grouped: dict = {}
        self._drag_drop_enabled = False

        self._enable_drag_drop()
        self._build_ui()
        self.root.after(200, self._check_health_on_startup)

    def _build_ui(self) -> None:
        self._build_toolbar()
        self._build_main_pane()
        self._build_status_bar()

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.pack(fill=tk.X)

        row1 = ttk.Frame(toolbar)
        row1.pack(fill=tk.X)

        ttk.Label(row1, text="服務位址").pack(side=tk.LEFT)
        self._url_var = tk.StringVar(value=DEFAULT_SERVICE_URL)
        ttk.Entry(row1, textvariable=self._url_var, width=46).pack(side=tk.LEFT, padx=(6, 10))
        ttk.Button(row1, text="測試連線", command=lambda: self._check_health(show_dialog=True)).pack(
            side=tk.LEFT, padx=2
        )

        row2 = ttk.Frame(toolbar)
        row2.pack(fill=tk.X, pady=(8, 0))

        ttk.Button(row2, text="選擇圖片", command=self._pick_file).pack(side=tk.LEFT, padx=(0, 4))
        self._scan_btn = ttk.Button(row2, text="開始掃描", command=self._start_scan, state="disabled")
        self._scan_btn.pack(side=tk.LEFT, padx=4)

        ttk.Label(row2, text="信心門檻").pack(side=tk.LEFT, padx=(16, 4))
        self._confidence_var = tk.DoubleVar(value=DEFAULT_MIN_CONFIDENCE)
        confidence_scale = ttk.Scale(
            row2,
            from_=0.0,
            to=1.0,
            variable=self._confidence_var,
            orient=tk.HORIZONTAL,
            length=140,
            command=self._update_confidence_label,
        )
        confidence_scale.pack(side=tk.LEFT)
        self._confidence_label = ttk.Label(row2, text=f"{DEFAULT_MIN_CONFIDENCE:.2f}")
        self._confidence_label.pack(side=tk.LEFT, padx=(6, 16))

        ttk.Button(row2, text="複製文字", command=self._copy_text).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="儲存文字", command=self._save_text).pack(side=tk.LEFT, padx=2)

    def _build_main_pane(self) -> None:
        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 6))

        left = ttk.LabelFrame(pane, text="影像預覽", padding=4)
        pane.add(left, weight=3)

        self._preview_tabs = ttk.Notebook(left)
        self._preview_tabs.pack(fill=tk.BOTH, expand=True)

        original_tab = ttk.Frame(self._preview_tabs)
        processed_tab = ttk.Frame(self._preview_tabs)
        self._preview_tabs.add(original_tab, text="原圖")
        self._preview_tabs.add(processed_tab, text="校正後標框")

        self._canvas_original = tk.Canvas(original_tab, bg="#d9dde3", highlightthickness=0, cursor="hand2")
        self._canvas_original.pack(fill=tk.BOTH, expand=True)
        self._canvas_original.bind("<Configure>", self._on_original_canvas_configure)
        self._canvas_original.bind("<Button-1>", lambda _e: self._pick_file())
        self._register_drop_target()

        self._canvas_processed = tk.Canvas(processed_tab, bg="#ffffff", highlightthickness=0)
        self._canvas_processed.pack(fill=tk.BOTH, expand=True)
        self._canvas_processed.bind("<Configure>", lambda _e: self._redraw_processed())

        hint = "拖曳圖片到左側，或點擊原圖區域選擇檔案"
        if not self._drag_drop_enabled:
            hint = "點擊原圖區域選擇檔案（安裝 tkinterdnd2 可啟用拖曳）"
        self._hint_text = hint
        self._canvas_original.create_text(
            360,
            240,
            text=hint,
            fill="#666666",
            font=("Microsoft JhengHei UI", 12),
            tags="placeholder",
        )

        right = ttk.LabelFrame(pane, text="辨識結果", padding=4)
        pane.add(right, weight=2)

        self._summary_var = tk.StringVar(value="尚未掃描")
        ttk.Label(right, textvariable=self._summary_var, foreground="#444444").pack(anchor=tk.W, pady=(0, 4))

        right_pane = ttk.PanedWindow(right, orient=tk.VERTICAL)
        right_pane.pack(fill=tk.BOTH, expand=True)

        raw_frame = ttk.LabelFrame(right_pane, text="原始結果", padding=4)
        right_pane.add(raw_frame, weight=1)

        self._raw_text_box = scrolledtext.ScrolledText(
            raw_frame,
            wrap=tk.WORD,
            font=("Microsoft JhengHei UI", 11),
            state="disabled",
        )
        self._raw_text_box.pack(fill=tk.BOTH, expand=True)

        grouped_frame = ttk.LabelFrame(right_pane, text="套用行／列結果", padding=4)
        right_pane.add(grouped_frame, weight=1)

        self._grouped_orient_var = tk.StringVar(value="")
        ttk.Label(grouped_frame, textvariable=self._grouped_orient_var, foreground="#555555").pack(anchor=tk.W)

        self._grouped_text_box = scrolledtext.ScrolledText(
            grouped_frame,
            wrap=tk.WORD,
            font=("Microsoft JhengHei UI", 11),
            state="disabled",
        )
        self._grouped_text_box.pack(fill=tk.BOTH, expand=True)

    def _build_status_bar(self) -> None:
        bar = ttk.Frame(self.root, padding=(10, 6))
        bar.pack(fill=tk.X)
        self._status_var = tk.StringVar(value="就緒")
        ttk.Label(bar, textvariable=self._status_var).pack(side=tk.LEFT)
        self._progress = ttk.Progressbar(
            bar, mode="indeterminate", length=160
        )
        self._progress.pack(side=tk.LEFT, padx=(12, 0))

    def _enable_drag_drop(self) -> None:
        try:
            from tkinterdnd2 import DND_FILES

            self._drag_drop_enabled = True
            self._dnd_files = DND_FILES
        except Exception:
            self._drag_drop_enabled = False
            self._dnd_files = None

    def _register_drop_target(self) -> None:
        if not self._drag_drop_enabled or self._dnd_files is None:
            return
        try:
            self._canvas_original.drop_target_register(self._dnd_files)
            self._canvas_original.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            self._drag_drop_enabled = False

    def _update_drop_hint(self) -> None:
        if self._original_image is not None:
            return

        self._canvas_original.delete("placeholder")
        hint = (
            "拖曳圖片到左側，或點擊原圖區域選擇檔案"
            if self._drag_drop_enabled
            else "點擊原圖區域選擇檔案（安裝 tkinterdnd2 可啟用拖曳）"
        )
        cw = self._canvas_original.winfo_width() or 700
        ch = self._canvas_original.winfo_height() or 500
        self._canvas_original.create_text(
            cw // 2,
            ch // 2,
            text=hint,
            fill="#666666",
            font=("Microsoft JhengHei UI", 12),
            tags="placeholder",
        )

    def _update_confidence_label(self, _value: str = "") -> None:
        self._confidence_label.configure(text=f"{self._confidence_var.get():.2f}")

    def _parse_drop_path(self, data: str) -> Path | None:
        raw = data.strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        path = Path(raw)
        return path if path.exists() else None

    def _on_drop(self, event) -> None:
        path = self._parse_drop_path(event.data)
        if path:
            self._load_image(path)

    def _pick_file(self) -> None:
        path = filedialog.askopenfilename(
            title="選擇圖片",
            filetypes=[("圖片檔", "*.png *.jpg *.jpeg"), ("所有檔案", "*.*")],
        )
        if path:
            self._load_image(Path(path))

    def _mime_for_path(self, path: Path) -> str:
        mapping = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }
        return mapping.get(path.suffix.lower(), "application/octet-stream")

    def _load_image(self, path: Path) -> None:
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            messagebox.showerror("格式錯誤", "請選擇 PNG 或 JPEG 圖片。")
            return

        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("讀取失敗", f"無法開啟圖片：{exc}")
            return

        self._image_path = path
        self._original_image = image
        self._processed_image = None
        self._predictions = []
        self._metadata = {}
        self._grouped = {}
        self._scan_btn.configure(state="normal")
        self._status_var.set(f"已載入：{path.name}")
        self._summary_var.set(f"原圖尺寸：{image.width} x {image.height}")
        self._set_raw_text("")
        self._set_grouped_text("", "")
        self._preview_tabs.select(0)
        self._redraw_original()
        self._redraw_processed()

    def _fit_image_to_canvas(self, image: Image.Image, canvas: tk.Canvas) -> ImageTk.PhotoImage:
        cw = max(canvas.winfo_width(), 1)
        ch = max(canvas.winfo_height(), 1)
        display = image.copy()
        display.thumbnail((cw, ch), Image.LANCZOS)
        return ImageTk.PhotoImage(display)

    def _on_original_canvas_configure(self, _event: tk.Event) -> None:
        if self._original_image is None:
            self._update_drop_hint()
        self._redraw_original()

    def _redraw_original(self) -> None:
        cw = self._canvas_original.winfo_width() or 700
        ch = self._canvas_original.winfo_height() or 500
        self._canvas_original.delete("all")
        if self._original_image is None:
            self._canvas_original.create_text(
                cw // 2, ch // 2,
                text=self._hint_text,
                fill="#666666",
                font=("Microsoft JhengHei UI", 9),
                tags="placeholder",
            )
            return

        self._photo_original = self._fit_image_to_canvas(self._original_image, self._canvas_original)
        self._canvas_original.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self._photo_original)

    def _build_processed_image(self) -> Image.Image | None:
        if not self._metadata or not self._predictions:
            return None

        dims = self._metadata.get("processed_dimensions", "")
        try:
            pw, ph = map(int, dims.lower().split("x"))
        except ValueError:
            return None

        canvas = Image.new("RGB", (pw, ph), "white")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        for index, pred in enumerate(self._predictions, start=1):
            pts = [tuple(p) for p in pred["bounding_box"]]
            draw.polygon(pts, outline="#1f8f4e", width=2)
            x0 = min(p[0] for p in pts)
            y0 = min(p[1] for p in pts)
            label = f'{index}. {pred["text"]} ({pred["confidence"]:.2f})'
            label_y = max(y0 - 14, 0)
            draw.rectangle([x0, label_y, x0 + min(len(label) * 7, pw - x0), label_y + 14], fill="#1f8f4e")
            draw.text((x0 + 2, label_y + 1), label, fill="#ffffff", font=font)

        return canvas

    def _redraw_processed(self) -> None:
        self._canvas_processed.delete("all")
        if self._processed_image is None:
            cw = self._canvas_processed.winfo_width() or 700
            ch = self._canvas_processed.winfo_height() or 500
            self._canvas_processed.create_text(
                cw // 2,
                ch // 2,
                text="掃描完成後，這裡會顯示校正後的標框預覽",
                fill="#888888",
                font=("Microsoft JhengHei UI", 12),
            )
            return

        self._photo_processed = self._fit_image_to_canvas(self._processed_image, self._canvas_processed)
        cw = self._canvas_processed.winfo_width() or 700
        ch = self._canvas_processed.winfo_height() or 500
        self._canvas_processed.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self._photo_processed)

    def _service_base_url(self) -> str:
        return self._url_var.get().strip().rstrip("/")

    def _check_health_on_startup(self) -> None:
        self._check_health(show_dialog=False)

    def _check_health(self, show_dialog: bool = True) -> None:
        url = f"{self._service_base_url()}/health"
        self._status_var.set("正在測試連線…")
        threading.Thread(
            target=self._do_health_check,
            args=(url, show_dialog),
            daemon=True,
        ).start()

    def _do_health_check(self, url: str, show_dialog: bool) -> None:
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            payload = resp.json()
            status = payload.get("status", "unknown")
            self.root.after(0, self._on_health_ok, f"服務已連線（{status}）", show_dialog)
        except requests.exceptions.ConnectionError:
            self.root.after(
                0,
                self._on_health_fail,
                f"無法連線到 {url}\n請先執行 start_server.bat 或 uvicorn main:app --port 8000",
                show_dialog,
            )
        except Exception as exc:
            self.root.after(0, self._on_health_fail, str(exc), show_dialog)

    def _on_health_ok(self, message: str, show_dialog: bool) -> None:
        self._status_var.set(message)
        if show_dialog:
            messagebox.showinfo("連線測試", message)

    def _on_health_fail(self, message: str, show_dialog: bool) -> None:
        self._status_var.set(f"服務未連線 — {message.splitlines()[0]}")
        if show_dialog:
            messagebox.showerror("連線測試", message)

    def _start_scan(self) -> None:
        if not self._image_path:
            return

        self._scan_btn.configure(state="disabled")
        self._status_var.set("掃描中，首次執行可能較久，請稍候…")
        self._progress.start(12)
        threading.Thread(target=self._do_scan, daemon=True).start()

    def _do_scan(self) -> None:
        url = f"{self._service_base_url()}/api/v1/scan"
        confidence = round(self._confidence_var.get(), 2)

        try:
            with open(self._image_path, "rb") as file_handle:
                resp = requests.post(
                    url,
                    files={
                        "file": (
                            self._image_path.name,
                            file_handle,
                            self._mime_for_path(self._image_path),
                        )
                    },
                    data={"min_confidence": confidence},
                    timeout=SCAN_TIMEOUT,
                )
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.ConnectionError:
            self.root.after(
                0,
                self._on_scan_error,
                f"無法連線到：\n{url}\n\n請先啟動 OCR 服務（uvicorn main:app --port 8000）",
            )
            return
        except requests.exceptions.HTTPError as exc:
            self.root.after(0, self._on_scan_error, self._format_http_error(exc))
            return
        except Exception as exc:
            self.root.after(0, self._on_scan_error, str(exc))
            return

        self.root.after(0, self._on_scan_done, result)

    def _format_http_error(self, exc: requests.exceptions.HTTPError) -> str:
        response = exc.response
        if response is None:
            return str(exc)

        try:
            detail = response.json().get("detail", {})
            if isinstance(detail, dict):
                tag = detail.get("error", "HTTP_ERROR")
                friendly = ERROR_MESSAGES.get(tag)
                msg = detail.get("message", response.text)
                if friendly:
                    return f"{friendly}\n\n（{tag}）"
                return f"錯誤代碼：{tag}\n\n{msg}"
        except Exception:
            pass

        return f"HTTP {response.status_code}\n\n{response.text}"

    def _format_scan_summary(self, prediction_count: int) -> str:
        width = self._metadata.get("width")
        height = self._metadata.get("height")
        processed_size = self._metadata.get("processed_dimensions", "-")

        parts = [f"共 {prediction_count} 筆結果"]
        if width and height:
            parts.append(f"原圖 {width} x {height}")
        parts.append(f"處理後 {processed_size}")

        denoise_key = self._metadata.get("denoising")
        if denoise_key:
            denoise_label = DENOISE_LABELS.get(str(denoise_key), str(denoise_key))
            noise_score = self._metadata.get("noise_score")
            if noise_score is not None:
                parts.append(f"降噪 {denoise_label}（分數 {noise_score}）")
            else:
                parts.append(f"降噪 {denoise_label}")

        return " | ".join(parts)

    def _on_scan_done(self, result: dict) -> None:
        self._metadata = result.get("metadata", {})
        self._predictions = result.get("predictions", [])
        self._grouped = result.get("grouped", {})
        self._processed_image = self._build_processed_image()

        # 原始結果
        plain_lines = [pred["text"] for pred in self._predictions]
        detail_lines = [f'[{pred["confidence"]:.4f}] {pred["text"]}' for pred in self._predictions]
        plain_text = "\n".join(plain_lines)
        detail_text = "\n".join(detail_lines)

        raw_content = plain_text if plain_text else "（未辨識到文字，可試著調低信心門檻或換張更清晰的圖片）"
        self._set_raw_text(raw_content)
        if detail_lines:
            self._raw_text_box.configure(state="normal")
            self._raw_text_box.insert(tk.END, "\n\n---\n詳細結果\n---\n")
            self._raw_text_box.insert(tk.END, detail_text)
            self._raw_text_box.configure(state="disabled")

        # 套用行／列結果
        orientation = self._grouped.get("orientation", "row")
        if orientation == "column":
            entries = self._grouped.get("columns", [])
            orient_label = "方向：直排（列）"
        elif orientation == "row":
            entries = self._grouped.get("rows", [])
            orient_label = "方向：橫排（行）"
        else:
            entries = self._grouped.get("rows", [])
            orient_label = "方向：混合（以行為主）"

        grouped_text = "\n".join(e["text"] for e in entries) if entries else "（無分組結果）"
        self._set_grouped_text(grouped_text, orient_label)

        self._progress.stop()
        self._redraw_processed()
        self._preview_tabs.select(1)
        self._scan_btn.configure(state="normal")

        original_size = f'{self._metadata.get("width")} x {self._metadata.get("height")}'
        processed_size = self._metadata.get("processed_dimensions", "-")
        self._summary_var.set(
            f"共 {len(self._predictions)} 筆結果 | 原圖 {original_size} | 校正後 {processed_size}"
        )
        self._status_var.set("掃描完成")

    def _on_scan_error(self, message: str) -> None:
        self._progress.stop()
        self._scan_btn.configure(state="normal")
        self._status_var.set("掃描失敗")
        messagebox.showerror("掃描失敗", message)

    def _set_raw_text(self, content: str) -> None:
        self._raw_text_box.configure(state="normal")
        self._raw_text_box.delete("1.0", tk.END)
        self._raw_text_box.insert("1.0", content)
        self._raw_text_box.configure(state="disabled")

    def _set_grouped_text(self, content: str, orient_label: str = "") -> None:
        self._grouped_orient_var.set(orient_label)
        self._grouped_text_box.configure(state="normal")
        self._grouped_text_box.delete("1.0", tk.END)
        self._grouped_text_box.insert("1.0", content)
        self._grouped_text_box.configure(state="disabled")

    def _copy_text(self) -> None:
        content = self._raw_text_box.get("1.0", tk.END).strip()
        if not content:
            messagebox.showinfo("複製文字", "目前沒有可複製的內容。")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(content.split("\n\n---\n")[0].strip())
        self._status_var.set("已複製辨識文字")

    def _save_text(self) -> None:
        content = self._raw_text_box.get("1.0", tk.END).strip()
        if not content:
            messagebox.showinfo("儲存文字", "目前沒有可儲存的內容。")
            return

        path = filedialog.asksaveasfilename(
            title="儲存辨識結果",
            defaultextension=".txt",
            filetypes=[("文字檔", "*.txt"), ("所有檔案", "*.*")],
        )
        if not path:
            return

        Path(path).write_text(content, encoding="utf-8")
        self._status_var.set(f"已儲存：{Path(path).name}")


def main() -> None:
    root = create_root()
    OCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

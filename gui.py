"""
Local Desktop GUI — Engineer B module.
Connects to the OCR Core Service (Engineer A) at POST /api/v1/scan.
"""

import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageTk

load_dotenv()

DEFAULT_SERVICE_URL = os.getenv("OCR_SERVICE_URL", "http://127.0.0.1:8000")
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.5"))


class OCRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Document Scanner")
        self.geometry("1200x720")
        self.minsize(800, 500)
        self.configure(bg="#f5f5f5")

        self._image_path: Path | None = None
        self._original_image: Image.Image | None = None
        self._photo_ref: ImageTk.PhotoImage | None = None  # must be kept alive
        self._predictions: list[dict] = []

        self._build_ui()
        self._try_enable_drop()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Service URL:").pack(side=tk.LEFT)
        self._url_var = tk.StringVar(value=DEFAULT_SERVICE_URL)
        ttk.Entry(top, textvariable=self._url_var, width=42).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Button(top, text="Choose Image…", command=self._pick_file).pack(side=tk.LEFT, padx=2)
        self._scan_btn = ttk.Button(top, text="Scan", command=self._start_scan, state="disabled")
        self._scan_btn.pack(side=tk.LEFT, padx=2)

        self._status_var = tk.StringVar(value="Drag an image onto the canvas, or click Choose Image.")
        ttk.Label(top, textvariable=self._status_var, foreground="#555555").pack(side=tk.LEFT, padx=10)

        pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # Left: image canvas
        left_frame = ttk.LabelFrame(pane, text="Image Preview")
        pane.add(left_frame, weight=3)

        self._canvas = tk.Canvas(left_frame, bg="#cccccc", cursor="hand2", highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas.bind("<Configure>", lambda _e: self._redraw())
        self._canvas.bind("<Button-1>", lambda _e: self._pick_file())

        # Right: text results
        right_frame = ttk.LabelFrame(pane, text="Extracted Text")
        pane.add(right_frame, weight=2)

        self._text_box = scrolledtext.ScrolledText(
            right_frame, wrap=tk.WORD, font=("Consolas", 11), state="disabled"
        )
        self._text_box.pack(fill=tk.BOTH, expand=True)

    def _try_enable_drop(self):
        try:
            from tkinterdnd2 import DND_FILES  # type: ignore[import]
            self._canvas.drop_target_register(DND_FILES)
            self._canvas.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            pass  # drag-and-drop is optional; file picker always works

    # ------------------------------------------------------------------ file loading

    def _on_drop(self, event):
        path = event.data.strip().strip("{}")
        self._load_image(Path(path))

    def _pick_file(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")],
        )
        if path:
            self._load_image(Path(path))

    def _load_image(self, path: Path):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            messagebox.showerror("Unsupported Format", "Please select a PNG or JPEG image.")
            return
        self._image_path = path
        self._original_image = Image.open(path)
        self._predictions = []
        self._redraw()
        self._scan_btn.configure(state="normal")
        self._status_var.set(f"Loaded: {path.name}")
        self._set_text_content("")

    # ------------------------------------------------------------------ rendering

    def _redraw(self):
        if self._original_image is None:
            return
        cw = self._canvas.winfo_width() or 700
        ch = self._canvas.winfo_height() or 500

        display = self._original_image.copy()
        if self._predictions:
            display = self._draw_bounding_boxes(display)

        display.thumbnail((cw, ch), Image.LANCZOS)
        self._photo_ref = ImageTk.PhotoImage(display)
        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self._photo_ref)

    def _draw_bounding_boxes(self, img: Image.Image) -> Image.Image:
        draw = ImageDraw.Draw(img)
        iw, ih = img.size
        for pred in self._predictions:
            pts = [tuple(p) for p in pred["bounding_box"]]
            # Scale bounding box from original image space (already correct here)
            draw.polygon(pts, outline="#00cc44", width=2)
            x0 = min(p[0] for p in pts)
            y0 = min(p[1] for p in pts)
            label = f'{pred["text"]} ({pred["confidence"]:.2f})'
            draw.rectangle([x0, max(y0 - 16, 0), x0 + len(label) * 7, max(y0, 16)], fill="#00cc44")
            draw.text((x0 + 1, max(y0 - 15, 1)), label, fill="#ffffff")
        return img

    # ------------------------------------------------------------------ scanning

    def _start_scan(self):
        if not self._image_path:
            return
        self._scan_btn.configure(state="disabled")
        self._status_var.set("Scanning…")
        threading.Thread(target=self._do_scan, daemon=True).start()

    def _do_scan(self):
        url = f"{self._url_var.get().rstrip('/')}/api/v1/scan"
        try:
            with open(self._image_path, "rb") as fh:
                resp = requests.post(
                    url,
                    files={"file": (self._image_path.name, fh, "image/jpeg")},
                    data={"min_confidence": MIN_CONFIDENCE},
                    timeout=60,
                )
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.ConnectionError:
            self.after(0, self._on_scan_error, f"Cannot connect to:\n{url}\n\nIs the OCR service running?")
            return
        except requests.exceptions.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", {})
                tag = detail.get("error", "HTTP_ERROR")
                msg = detail.get("message", str(exc))
            except Exception:
                tag, msg = "HTTP_ERROR", str(exc)
            self.after(0, self._on_scan_error, f"{tag}\n\n{msg}")
            return
        except Exception as exc:
            self.after(0, self._on_scan_error, str(exc))
            return

        self.after(0, self._on_scan_done, result)

    def _on_scan_done(self, result: dict):
        self._predictions = result.get("predictions", [])
        meta = result.get("metadata", {})

        lines = [f'[{p["confidence"]:.4f}]  {p["text"]}' for p in self._predictions]
        self._set_text_content("\n".join(lines) if lines else "(no text recognized)")

        self._redraw()
        self._scan_btn.configure(state="normal")
        self._status_var.set(
            f"Done — {len(self._predictions)} result(s)  |  "
            f'{meta.get("width")}×{meta.get("height")} → {meta.get("processed_dimensions")}'
        )

    def _on_scan_error(self, message: str):
        self._scan_btn.configure(state="normal")
        self._status_var.set("Scan failed — see error dialog.")
        messagebox.showerror("Scan Error", message)

    def _set_text_content(self, content: str):
        self._text_box.configure(state="normal")
        self._text_box.delete("1.0", tk.END)
        self._text_box.insert("1.0", content)
        self._text_box.configure(state="disabled")


if __name__ == "__main__":
    app = OCRApp()
    app.mainloop()

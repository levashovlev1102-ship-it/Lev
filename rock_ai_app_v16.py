from __future__ import annotations

import base64
import re
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk
import httpx


# Local LM Studio server. This test build uses GLM-4.6V-Flash only.
# LFM3-VL, Qwen, and Gemma are intentionally excluded.
LM_STUDIO_URL = "http://127.0.0.1:1234"
VISION_MODEL_PREFERENCES = ("glm-4.6v",)  # GLM-4.6V-Flash only for this test build

APP_BG = "#111827"
PANEL_BG = "#1F2937"
PANEL_DARK = "#0B1220"
ACCENT = "#38BDF8"
ACCENT_HOVER = "#0EA5E9"
TEXT_MAIN = "#F9FAFB"
TEXT_MUTED = "#9CA3AF"
SUCCESS = "#34D399"
WARNING = "#FBBF24"


def _content_to_text(content) -> str:
    """Extract user-facing text from LM Studio message content, never reasoning."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    chunks.append(part.strip())
            elif isinstance(part, dict):
                # OpenAI-compatible content parts may use text or content.
                part_type = str(part.get("type", "")).lower()
                if part_type in {"text", "output_text"}:
                    value = part.get("text", part.get("content", ""))
                    if isinstance(value, str) and value.strip():
                        chunks.append(value.strip())
        return "\n".join(chunks).strip()
    return ""


def extract_lmstudio_final_text(response_data: dict) -> str:
    """Read only final message items from LM Studio's native REST response.

    Deliberately ignores `reasoning` items: internal analysis is not a user-facing
    answer and must never be shown as the mineral identification.
    """
    output = response_data.get("output", []) if isinstance(response_data, dict) else []
    if not isinstance(output, list):
        return ""
    messages: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        message_text = _content_to_text(item.get("content", ""))
        if message_text:
            messages.append(message_text)
    return "\n\n".join(messages).strip()


def extract_openai_final_text(response_data: dict) -> str:
    """Read the final assistant message from /v1/chat/completions."""
    try:
        choices = response_data.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        if not isinstance(message, dict):
            return ""
        return _content_to_text(message.get("content", ""))
    except (AttributeError, IndexError, TypeError):
        return ""


def normalize_russian_geology_terms(text: str) -> str:
    """Replace common English/transliterated jasper spellings with Russian terms.

    This is a narrow terminology safeguard, not a general translation system.
    It prevents a known failure where the model writes “джаспер” instead of
    the standard Russian geological name “яшма”.
    """
    if not isinstance(text, str) or not text:
        return text

    replacements = {
        "джасперами": "яшмами",
        "джасперах": "яшмах",
        "джаспером": "яшмой",
        "джасперу": "яшме",
        "джаспера": "яшмы",
        "джаспере": "яшме",
        "джаспер": "яшма",
        "jasper": "яшма",
    }
    for source, target in replacements.items():
        pattern = re.compile(rf"\b{re.escape(source)}\b", re.IGNORECASE)

        def replace(match: re.Match[str], replacement: str = target) -> str:
            found = match.group(0)
            return replacement.capitalize() if found[:1].isupper() else replacement

        text = pattern.sub(replace, text)
    return text


def describe_native_output(response_data: dict) -> str:
    """Return non-sensitive response metadata useful for diagnostics."""
    output = response_data.get("output", []) if isinstance(response_data, dict) else []
    types = [str(item.get("type", "unknown")) for item in output if isinstance(item, dict)] if isinstance(output, list) else []
    stats = response_data.get("stats", {}) if isinstance(response_data, dict) else {}
    total = stats.get("total_output_tokens") if isinstance(stats, dict) else None
    reasoning = stats.get("reasoning_output_tokens") if isinstance(stats, dict) else None
    details = [f"Типы элементов: {', '.join(types) or 'не указаны'}"]
    if isinstance(total, (int, float)):
        details.append(f"всего сгенерировано токенов: {int(total)}")
    if isinstance(reasoning, (int, float)):
        details.append(f"токенов рассуждения: {int(reasoning)}")
    return "; ".join(details)


class ModernButton(tk.Button):
    """Classic rectangular 3-D button with a true sunken press state.

    The shape intentionally matches the original rectangular controls.  Tk's
    native raised/sunken relief draws crisp beveled edges instead of the
    antialiased Canvas polygons that looked pixelated in the previous build.
    """

    def __init__(
        self,
        parent: tk.Widget,
        text: str,
        command,
        *,
        bg: str,
        hover_bg: str,
        pressed_bg: str,
        fg: str = TEXT_MAIN,
        font=("Segoe UI", 10, "bold"),
        height: int = 48,
        disabled: bool = False,
    ) -> None:
        self._normal_bg = bg
        self._hover_bg = hover_bg
        self._pressed_bg = pressed_bg
        self._normal_fg = fg
        self._pressed = False
        self._hovered = False

        super().__init__(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=hover_bg,
            activeforeground=fg,
            disabledforeground="#718096",
            font=font,
            height=1,  # Tk Button height is measured in text lines, not pixels.
            relief=tk.RAISED,
            overrelief=tk.RAISED,
            borderwidth=3,
            highlightthickness=0,
            padx=12,
            pady=max(7, (height - 23) // 2),
            cursor="arrow" if disabled else "hand2",
            state=tk.DISABLED if disabled else tk.NORMAL,
            takefocus=True,
        )

        # Keep the rectangular silhouette; only the bevel and surface colour
        # change while the mouse is held down.
        self.bind("<Enter>", self._on_enter, add="+")
        self.bind("<Leave>", self._on_leave, add="+")
        self.bind("<ButtonPress-1>", self._on_press, add="+")
        self.bind("<ButtonRelease-1>", self._on_release, add="+")
        self.bind("<FocusOut>", self._cancel_press, add="+")
        self.bind("<KeyPress-space>", self._on_key_press, add="+")
        self.bind("<KeyRelease-space>", self._on_key_release, add="+")

    def _is_enabled(self) -> bool:
        return str(self.cget("state")) != str(tk.DISABLED)

    def _on_enter(self, _event=None) -> None:
        self._hovered = True
        if self._is_enabled() and not self._pressed:
            self.configure(bg=self._hover_bg, relief=tk.RAISED, cursor="hand2")

    def _on_leave(self, _event=None) -> None:
        self._hovered = False
        if self._is_enabled() and not self._pressed:
            self.configure(bg=self._normal_bg, relief=tk.RAISED)

    def _on_press(self, _event=None) -> None:
        if not self._is_enabled():
            return
        self._pressed = True
        self.configure(bg=self._pressed_bg, relief=tk.SUNKEN, cursor="hand2")

    def _on_release(self, event=None) -> None:
        if not self._is_enabled():
            self._pressed = False
            return
        self._pressed = False
        inside = bool(
            event
            and 0 <= event.x < self.winfo_width()
            and 0 <= event.y < self.winfo_height()
        )
        self.configure(
            bg=self._hover_bg if inside or self._hovered else self._normal_bg,
            relief=tk.RAISED,
            cursor="hand2",
        )

    def _cancel_press(self, _event=None) -> None:
        if self._pressed:
            self._pressed = False
            self.configure(
                bg=self._hover_bg if self._hovered else self._normal_bg,
                relief=tk.RAISED,
            )

    def _on_key_press(self, _event=None) -> None:
        if self._is_enabled() and not self._pressed:
            self._pressed = True
            self.configure(bg=self._pressed_bg, relief=tk.SUNKEN)

    def _on_key_release(self, _event=None) -> None:
        if self._pressed:
            self._pressed = False
            self.configure(
                bg=self._hover_bg if self._hovered else self._normal_bg,
                relief=tk.RAISED,
            )


class RockAIApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ROCK & AI — определение образцов · v16")
        self.root.geometry("1280x720")
        self.root.minsize(1120, 630)
        # Keep the application window in a 16:9 proportion when resized.
        try:
            self.root.aspect(16, 9, 16, 9)
        except tk.TclError:
            pass
        self.root.configure(bg=APP_BG)

        self.cv = None  # OpenCV camera handle
        self.camera_running = False
        self.current_frame: np.ndarray | None = None  # BGR image
        self.current_photo: ImageTk.PhotoImage | None = None
        self.current_path: Path | None = None
        self._camera_after_id: str | None = None
        self._resize_after_id: str | None = None
        self._camera_thread: threading.Thread | None = None
        self._camera_stop_event = threading.Event()
        self._camera_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_time = 0.0
        self._last_displayed_frame_time = 0.0
        self._camera_stale_reported = False
        self._server_check_running = False
        self._server_check_after_id: str | None = None
        self.active_model_id: str | None = None
        self.camera_connecting = False
        self._camera_open_thread: threading.Thread | None = None
        self._preferred_camera_backend: int | None = None
        self._camera_open_cancel = threading.Event()
        self._camera_started_at = 0.0
        self._camera_shutdown_pending = False
        self._file_picker_pending = False
        self._camera_release_thread: threading.Thread | None = None
        self._camera_reader_stopping_thread: threading.Thread | None = None
        self._camera_open_stopping_thread: threading.Thread | None = None
        self._closing = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(350, self._check_server)

    def _build_ui(self) -> None:
        # Header
        header = tk.Frame(self.root, bg=PANEL_DARK, height=72)
        header.pack(fill="x")
        header.pack_propagate(False)

        brand = tk.Label(
            header,
            text="ROCK  &  AI",
            bg=PANEL_DARK,
            fg=TEXT_MAIN,
            font=("Segoe UI", 21, "bold"),
        )
        brand.pack(side="left", padx=(28, 12), pady=15)

        subtitle = tk.Label(
            header,
            text="ОПРЕДЕЛЕНИЕ МИНЕРАЛОВ И ГОРНЫХ ПОРОД",
            bg=PANEL_DARK,
            fg=TEXT_MUTED,
            font=("Segoe UI", 9, "bold"),
        )
        subtitle.pack(side="left", pady=22)

        self.server_label = tk.Label(
            header,
            text="● Проверка LM Studio…",
            bg=PANEL_DARK,
            fg=WARNING,
            font=("Segoe UI", 10),
        )
        self.server_label.pack(side="right", padx=28)

        # Main content: image/camera on the left, AI result on the right.
        main = tk.Frame(self.root, bg=APP_BG)
        main.pack(fill="both", expand=True, padx=22, pady=18)
        main.grid_columnconfigure(0, weight=6, uniform="content")
        main.grid_columnconfigure(1, weight=4, uniform="content")
        main.grid_rowconfigure(0, weight=1)

        left = tk.Frame(main, bg=PANEL_BG, highlightthickness=1, highlightbackground="#374151")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        left_title = tk.Label(
            left,
            text="ИЗОБРАЖЕНИЕ ОБРАЗЦА",
            bg=PANEL_BG,
            fg=TEXT_MAIN,
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        )
        left_title.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 10))

        self.preview = tk.Canvas(left, bg="#070B13", highlightthickness=0)
        self.preview.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 12))
        self.preview.bind("<Configure>", self._on_preview_resize)
        self._draw_placeholder()

        image_buttons = tk.Frame(left, bg=PANEL_BG)
        image_buttons.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 16))
        image_buttons.grid_columnconfigure((0, 1), weight=1)

        self.open_photo_button = self._make_button(image_buttons, "Открыть фото", self.open_image)
        self.open_photo_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        # One stateful camera button: it starts the camera while it is off,
        # and becomes the snapshot button while the live preview is running.
        self.camera_button = self._make_button(image_buttons, "Включить веб-камеру", self.start_camera)
        self.camera_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        right = tk.Frame(main, bg=PANEL_BG, highlightthickness=1, highlightbackground="#374151")
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right.grid_rowconfigure(3, weight=1)
        right.grid_columnconfigure(0, weight=1)

        result_title = tk.Label(
            right,
            text="РЕЗУЛЬТАТ АНАЛИЗА",
            bg=PANEL_BG,
            fg=TEXT_MAIN,
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        )
        result_title.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 8))

        self.hint_label = tk.Label(
            right,
            text="Совет: снимайте один образец при ровном освещении. После распознавания сравните ответ с полевыми проверками.",
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 9),
            anchor="w",
            justify="left",
            wraplength=390,
        )
        self.hint_label.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 10))

        # Specimen condition selector: this context is passed to the vision model.
        condition_frame = tk.Frame(right, bg=PANEL_BG)
        condition_frame.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 10))
        condition_frame.grid_columnconfigure(0, weight=1)
        condition_label = tk.Label(
            condition_frame,
            text="СОСТОЯНИЕ ОБРАЗЦА",
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        )
        condition_label.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        self.specimen_condition_options = (
            "Не указано — определить по фотографии",
            "Целый образец, сухой",
            "Свежий скол",
            "Образец смочен водой",
            "Полированная поверхность или срез",
            "Кристаллы или друза",
            "Образец в материнской породе",
            "Крошка или отдельные фрагменты",
        )
        self.specimen_condition_var = tk.StringVar(value=self.specimen_condition_options[0])
        style = ttk.Style(self.root)
        try:
            style.configure(
                "Rock.TCombobox",
                fieldbackground=PANEL_DARK,
                background=PANEL_BG,
                foreground=TEXT_MAIN,
                arrowcolor=TEXT_MAIN,
                padding=7,
            )
            style.map(
                "Rock.TCombobox",
                fieldbackground=[("readonly", PANEL_DARK)],
                foreground=[("readonly", TEXT_MAIN)],
            )
        except tk.TclError:
            pass
        self.specimen_condition_combo = ttk.Combobox(
            condition_frame,
            textvariable=self.specimen_condition_var,
            values=self.specimen_condition_options,
            state="readonly",
            style="Rock.TCombobox",
            font=("Segoe UI", 9),
        )
        self.specimen_condition_combo.grid(row=1, column=0, sticky="ew")

        result_box = tk.Frame(right, bg=PANEL_DARK)
        result_box.grid(row=3, column=0, sticky="nsew", padx=16, pady=(0, 12))
        result_box.grid_rowconfigure(0, weight=1)
        result_box.grid_columnconfigure(0, weight=1)

        self.result_text = tk.Text(
            result_box,
            bg=PANEL_DARK,
            fg=TEXT_MAIN,
            insertbackground=TEXT_MAIN,
            selectbackground="#0E7490",
            selectforeground="#F8FAFC",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            wrap="word",
            font=("Segoe UI", 11),
            padx=14,
            pady=14,
            spacing1=4,
            spacing3=7,
            spacing2=2,
        )
        self.result_text.grid(row=0, column=0, sticky="nsew")
        self.result_text.insert(
            "1.0",
            "Здесь появится ответ модели.\n\n"
            "Сначала оцените видимые признаки. Затем проверьте предположение "
            "водой, свежим сколом или царапанием стекла — если это подходит образцу.",
        )
        self.result_text.configure(state="disabled")

        self.analyze_button = self._make_button(
            right, "АНАЛИЗИРОВАТЬ ОБРАЗЕЦ", self.analyze_image, prominent=True
        )
        self.analyze_button.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 10), ipady=5)
        self.progress_label = tk.Label(
            right,
            text="Ожидание изображения",
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 9),
            anchor="w",
        )
        self.progress_label.grid(row=5, column=0, sticky="ew", padx=18, pady=(0, 15))

        footer = tk.Label(
            self.root,
            text="ROCK & AI  •  Локальный анализ через LM Studio  •  Результат модели требует проверки",
            bg=APP_BG,
            fg="#6B7280",
            font=("Segoe UI", 8),
        )
        footer.pack(fill="x", pady=(0, 8))

    def _make_button(
        self,
        parent: tk.Widget,
        text: str,
        command,
        *,
        secondary: bool = False,
        prominent: bool = False,
    ) -> ModernButton:
        if prominent:
            normal, hover, pressed = ACCENT, ACCENT_HOVER, "#0284C7"
            foreground = "#082F49"
            font = ("Segoe UI", 11, "bold")
            height = 54
        elif secondary:
            normal, hover, pressed = "#334155", "#475569", "#1E293B"
            foreground = TEXT_MAIN
            font = ("Segoe UI", 10, "bold")
            height = 48
        else:
            normal, hover, pressed = "#0F766E", "#0D9488", "#115E59"
            foreground = TEXT_MAIN
            font = ("Segoe UI", 10, "bold")
            height = 48
        return ModernButton(
            parent,
            text,
            command,
            bg=normal,
            hover_bg=hover,
            pressed_bg=pressed,
            fg=foreground,
            font=font,
            height=height,
        )

    def _draw_placeholder(self) -> None:
        self.preview.delete("all")
        self.preview.create_text(
            max(1, self.preview.winfo_width() // 2),
            max(1, self.preview.winfo_height() // 2),
            text="ЗДЕСЬ БУДЕТ ФОТОГРАФИЯ ОБРАЗЦА\n\nОткройте фото или включите веб-камеру",
            fill="#6B7280",
            font=("Segoe UI", 12),
            justify="center",
        )

    def _on_preview_resize(self, _event=None) -> None:
        if self._resize_after_id:
            try:
                self.root.after_cancel(self._resize_after_id)
            except tk.TclError:
                pass
        self._resize_after_id = self.root.after(80, self._refresh_preview)

    def _refresh_preview(self) -> None:
        self._resize_after_id = None
        if self.current_frame is None:
            self._draw_placeholder()
            return
        frame = self.current_frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width = max(1, self.preview.winfo_width())
        height = max(1, self.preview.winfo_height())
        image = ImageOps.contain(image, (width - 12, height - 12))
        self.current_photo = ImageTk.PhotoImage(image)
        self.preview.delete("all")
        self.preview.create_image(width // 2, height // 2, image=self.current_photo, anchor="center")

    def _set_result(self, text: str) -> None:
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")

    def _set_status(self, text: str, color: str = TEXT_MUTED) -> None:
        self.progress_label.configure(text=text, fg=color)

    @staticmethod
    def _choose_vision_model(model_ids: list[str]) -> str | None:
        """Choose a known vision model, preferring GLM-4.6V-Flash.

        LFM3-VL, Qwen, and Gemma are deliberately excluded from the next validation run.
        """
        normalized = [model_id for model_id in model_ids if isinstance(model_id, str)]
        for marker in VISION_MODEL_PREFERENCES:
            match = next((mid for mid in normalized if marker in mid.lower()), None)
            if match:
                return match
        return None

    def _check_server(self) -> None:
        """Check local LM Studio without blocking the Tkinter event loop."""
        if self._server_check_running or self._closing:
            return
        self._server_check_running = True

        def worker() -> None:
            status = None
            try:
                with httpx.Client(timeout=4.0, trust_env=False) as client:
                    response = client.get(f"{LM_STUDIO_URL}/v1/models")
                    response.raise_for_status()
                    payload = response.json()
                model_ids = [item.get("id", "") for item in payload.get("data", [])]
                vision_model = self._choose_vision_model(model_ids)
                if vision_model:
                    status = (True, f"● LM Studio подключён · {vision_model}", SUCCESS, vision_model)
                elif model_ids:
                    status = (
                        False,
                        "● Сервер доступен · загрузите GLM-4.6V-Flash",
                        WARNING,
                        "Сервер работает, но GLM-4.6V-Flash не загружена. Старую LFM3-VL, Qwen и Gemma не используем.",
                    )
                else:
                    status = (False, "● Сервер доступен, но список моделей пуст", WARNING, "Загрузите GLM-4.6V-Flash в LM Studio.")
            except Exception as exc:
                detail = str(exc).strip().replace("\n", " ") or type(exc).__name__
                if len(detail) > 110:
                    detail = detail[:107] + "..."
                status = (False, "● Не удалось подключиться к LM Studio", WARNING, detail)

            try:
                self.root.after(0, lambda: self._complete_server_check(status))
            except tk.TclError:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def _complete_server_check(self, status) -> None:
        self._server_check_running = False
        self._apply_server_status(status)
        if not self._closing:
            try:
                self._server_check_after_id = self.root.after(5000, self._check_server)
            except tk.TclError:
                self._server_check_after_id = None

    def _apply_server_status(self, status) -> None:
        ok, text, color, *details = status
        self.server_label.configure(text=text, fg=color)
        self.active_model_id = details[0] if ok and details else None
        if not ok and details and details[0]:
            self._set_status(details[0], WARNING)

    def _stop_camera(self) -> None:
        self.camera_running = False
        self._camera_stop_event.set()
        if self._camera_after_id:
            try:
                self.root.after_cancel(self._camera_after_id)
            except tk.TclError:
                pass
            self._camera_after_id = None
        camera = self.cv
        self.cv = None
        thread = self._camera_thread
        # Let the capture loop notice the stop event before releasing the device.
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.5)
        if camera is not None:
            try:
                camera.release()
            except Exception:
                pass
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.8)
        self._camera_thread = None
        self._camera_stop_event = threading.Event()
        with self._camera_lock:
            self._latest_frame = None
            self._latest_frame_time = 0.0
        self._last_displayed_frame_time = 0.0
        self._camera_stale_reported = False
        self.camera_button.configure(text="Включить веб-камеру", command=self.start_camera)

    @staticmethod
    def _release_camera_handle(camera) -> None:
        """Release a VideoCapture handle without blocking the Tk event loop."""
        try:
            camera.release()
        except Exception:
            pass

    def _begin_camera_shutdown(
        self,
        *,
        open_file_dialog: bool,
        status_message: str,
    ) -> None:
        """Fully disable the webcam on a worker, then optionally open file picker.

        The UI thread never waits for VideoCapture.read(), join(), or release().
        A slow Windows camera driver therefore cannot freeze the application while
        the user is trying to choose a photo.
        """
        if self._camera_shutdown_pending:
            return

        has_camera_work = bool(
            self.camera_running
            or self.camera_connecting
            or self.cv is not None
            or (self._camera_thread is not None and self._camera_thread.is_alive())
        )
        if not has_camera_work:
            self.camera_running = False
            self.camera_connecting = False
            self.camera_button.configure(
                text="Включить веб-камеру", state="normal", command=self.start_camera
            )
            if open_file_dialog:
                self._show_image_dialog()
            else:
                self._set_status(status_message, SUCCESS)
            return

        self._camera_shutdown_pending = True
        self._file_picker_pending = open_file_dialog
        self.camera_running = False
        self._camera_stop_event.set()

        if self._camera_after_id:
            try:
                self.root.after_cancel(self._camera_after_id)
            except tk.TclError:
                pass
            self._camera_after_id = None

        opening_thread = self._camera_open_thread if self.camera_connecting else None
        if self.camera_connecting:
            self._camera_open_cancel.set()
            self.camera_connecting = False

        camera = self.cv
        self.cv = None
        reader_thread = self._camera_thread
        self._camera_thread = None
        self._camera_reader_stopping_thread = reader_thread
        self._camera_open_stopping_thread = opening_thread
        self._camera_release_thread = None

        with self._camera_lock:
            self._latest_frame = None
            self._latest_frame_time = 0.0
        self._last_displayed_frame_time = 0.0
        self._camera_stale_reported = False

        self.camera_button.configure(
            text="Отключение камеры…", state="disabled", command=self.start_camera
        )
        self.open_photo_button.configure(state="disabled")
        self._set_status("Полностью отключаю веб-камеру перед открытием файлов…", WARNING)

        def shutdown_worker() -> None:
            warnings: list[str] = []

            # If the camera was still opening, ask its worker to cancel and wait
            # briefly. That worker now releases any handle it acquired itself.
            if opening_thread is not None and opening_thread.is_alive():
                opening_thread.join(timeout=3.0)
                if opening_thread.is_alive():
                    warnings.append("драйвер ещё завершает подключение")

            # Let a normal frame read exit after the stop event before releasing.
            if reader_thread is not None and reader_thread.is_alive():
                reader_thread.join(timeout=0.35)

            # Release can block in some Windows camera drivers. Run it in its own
            # daemon thread and cap how long the file picker waits for it.
            release_thread = None
            if camera is not None:
                release_thread = threading.Thread(
                    target=self._release_camera_handle, args=(camera,), daemon=True
                )
                self._camera_release_thread = release_thread
                release_thread.start()
                release_thread.join(timeout=4.0)
                if release_thread.is_alive():
                    warnings.append("драйвер камеры не подтвердил освобождение за 4 секунды")

            if reader_thread is not None and reader_thread.is_alive():
                reader_thread.join(timeout=0.75)
                if reader_thread.is_alive():
                    warnings.append("поток чтения кадра ещё завершает работу")

            warning_text = "; ".join(warnings)
            try:
                self.root.after(
                    0,
                    lambda: self._camera_shutdown_completed(
                        open_file_dialog=open_file_dialog,
                        status_message=status_message,
                        warning_text=warning_text,
                    ),
                )
            except tk.TclError:
                pass

        threading.Thread(target=shutdown_worker, daemon=True).start()

    def _camera_shutdown_completed(
        self,
        *,
        open_file_dialog: bool,
        status_message: str,
        warning_text: str,
    ) -> None:
        if self._closing:
            return
        self._camera_shutdown_pending = False
        self._file_picker_pending = False
        if self._camera_open_thread is not None and not self._camera_open_thread.is_alive():
            self._camera_open_thread = None
        self._camera_thread = None
        self._camera_stop_event = threading.Event()
        self.camera_running = False
        self.camera_connecting = False
        self.open_photo_button.configure(state="normal")
        pending_release = any(
            thread is not None and thread.is_alive()
            for thread in (
                self._camera_release_thread,
                self._camera_reader_stopping_thread,
                self._camera_open_stopping_thread,
            )
        )
        if pending_release:
            self.camera_button.configure(
                text="Завершение отключения…", state="disabled", command=self.start_camera
            )
            self.root.after(250, self._poll_camera_release_completion)
        else:
            self.camera_button.configure(
                text="Включить веб-камеру", state="normal", command=self.start_camera
            )
            self._clear_camera_shutdown_threads()

        if open_file_dialog:
            if warning_text:
                self._set_status(
                    "Приложение прекратило работу с камерой; " + warning_text + ". Открываю выбор фото.",
                    WARNING,
                )
            else:
                self._set_status("Камера отключена. Теперь можно выбрать фотографию.", SUCCESS)
            self._show_image_dialog()
        else:
            if warning_text:
                self._set_status("Камера отключена; " + warning_text, WARNING)
            else:
                self._set_status(status_message, SUCCESS)

    def _clear_camera_shutdown_threads(self) -> None:
        self._camera_release_thread = None
        self._camera_reader_stopping_thread = None
        self._camera_open_stopping_thread = None

    def _poll_camera_release_completion(self) -> None:
        if self._closing:
            return
        pending = any(
            thread is not None and thread.is_alive()
            for thread in (
                self._camera_release_thread,
                self._camera_reader_stopping_thread,
                self._camera_open_stopping_thread,
            )
        )
        if pending:
            self.root.after(250, self._poll_camera_release_completion)
            return
        self._clear_camera_shutdown_threads()
        self.camera_button.configure(
            text="Включить веб-камеру", state="normal", command=self.start_camera
        )
        self._set_status("Веб-камера полностью отключена. Её можно включить снова.", SUCCESS)

    def start_camera(self) -> None:
        if self._camera_shutdown_pending:
            return
        if (
            (self._camera_release_thread is not None and self._camera_release_thread.is_alive())
            or (self._camera_reader_stopping_thread is not None and self._camera_reader_stopping_thread.is_alive())
            or (self._camera_open_stopping_thread is not None and self._camera_open_stopping_thread.is_alive())
        ):
            self._set_status("Драйвер ещё завершает отключение камеры. Подождите немного.", WARNING)
            return
        if self.camera_running:
            # The only visible action while the camera is active is to take
            # a snapshot; capturing also shuts the camera down cleanly.
            self.capture_camera()
            return
        if self.camera_connecting:
            self._camera_open_cancel.set()
            self.camera_connecting = False
            self.camera_button.configure(text="Включить веб-камеру", state="normal", command=self.start_camera)
            self._set_status("Подключение камеры отменено", WARNING)
            return

        self.camera_connecting = True
        self._camera_open_cancel = threading.Event()
        cancel_event = self._camera_open_cancel
        self.camera_button.configure(text="Отменить подключение", state="normal", command=self.start_camera)
        self._set_status("Подключение веб-камеры… приложение остаётся отзывчивым", WARNING)

        def open_worker() -> None:
            camera = None
            errors: list[str] = []
            # Rollback of the problematic MSMF-first / three-backend probing:
            # prefer the simpler DirectShow path used by the earlier build, then
            # try Media Foundation only if DirectShow cannot open the camera.
            # Keep opening in a worker so the app window itself stays responsive.
            # Do NOT force MJPG, FOURCC, or CONVERT_RGB: that caused colour artefacts
            # with this webcam before.
            backends = [
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF, "Media Foundation"),
            ]
            # Always try DirectShow first. Do not remember MSMF as the preferred
            # backend: this machine has reported MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED
            # (0xC00D3EA2) from Media Foundation.
            for backend, backend_name in backends:
                if cancel_event.is_set() or self._closing:
                    break
                candidate = None
                try:
                    candidate = cv2.VideoCapture(0, backend)
                    if candidate.isOpened():
                        camera = candidate
                        self._preferred_camera_backend = backend
                        break
                    errors.append(backend_name)
                except Exception as exc:
                    errors.append(f"{backend_name}: {exc}")
                if candidate is not None and candidate is not camera:
                    try:
                        candidate.release()
                    except Exception:
                        pass

            if camera is not None and not cancel_event.is_set() and not self._closing:
                # Keep negotiated pixel format untouched; request a useful image size.
                try:
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                except Exception:
                    pass

            # If the user cancelled while the driver was opening the device,
            # release it here (not on Tk's main thread).
            if camera is not None and (cancel_event.is_set() or self._closing):
                try:
                    camera.release()
                except Exception:
                    pass
                camera = None

            try:
                self.root.after(0, lambda: self._camera_open_completed(camera, errors, cancel_event))
            except tk.TclError:
                if camera is not None:
                    try:
                        camera.release()
                    except Exception:
                        pass

        self._camera_open_thread = threading.Thread(target=open_worker, daemon=True)
        self._camera_open_thread.start()

    def _camera_open_completed(self, camera, errors: list[str], cancel_event: threading.Event) -> None:
        if self._closing or cancel_event is not self._camera_open_cancel or cancel_event.is_set():
            if camera is not None:
                threading.Thread(target=self._release_camera_handle, args=(camera,), daemon=True).start()
            return

        self.camera_connecting = False
        self._camera_open_thread = None
        if camera is None:
            self.camera_button.configure(text="Включить веб-камеру", state="normal", command=self.start_camera)
            self._set_status("Не удалось подключить веб-камеру. Проверьте разрешения Windows.", WARNING)
            messagebox.showerror(
                "Веб-камера не найдена",
                "Не удалось открыть веб-камеру. Закройте программы, которые могут её использовать, "
                "и проверьте разрешения Windows.",
            )
            return

        self.cv = camera
        self.camera_running = True
        self.current_path = None
        self._camera_stop_event = threading.Event()
        self._camera_started_at = time.monotonic()
        self._camera_stale_reported = False
        self.camera_button.configure(text="Сделать снимок", state="normal", command=self.capture_camera)
        self._set_status("Веб-камера подключена · нажмите «Сделать снимок», чтобы сохранить кадр", SUCCESS)
        self._camera_thread = threading.Thread(
            target=self._camera_reader_loop, args=(camera, self._camera_stop_event), daemon=True
        )
        self._camera_thread.start()
        self._poll_camera_frame()

    def _camera_reader_loop(self, camera: cv2.VideoCapture, stop_event: threading.Event) -> None:
        """Read frames off the Tkinter thread and keep a consistent BGR frame format."""
        failed_reads = 0
        while not stop_event.is_set():
            try:
                ok, frame = camera.read()
            except Exception:
                ok, frame = False, None
            if not ok or frame is None:
                failed_reads += 1
                # MSMF can report MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED. Repeating
                # read() dozens of times only floods the console with the same warning;
                # three consecutive failures are enough to stop and release the device.
                if failed_reads >= 3:
                    break
                stop_event.wait(0.10)
                continue

            try:
                # OpenCV normally returns BGR. Normalize grayscale/BGRA frames without
                # swapping colour channels a second time.
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.ndim == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                elif frame.ndim != 3 or frame.shape[2] != 3:
                    failed_reads += 1
                    if failed_reads >= 3:
                        break
                    stop_event.wait(0.10)
                    continue
            except Exception:
                failed_reads += 1
                if failed_reads >= 3:
                    break
                stop_event.wait(0.10)
                continue

            failed_reads = 0
            with self._camera_lock:
                self._latest_frame = frame.copy()
                self._latest_frame_time = time.monotonic()

        if not stop_event.is_set():
            try:
                self.root.after(0, self._camera_stream_failed)
            except tk.TclError:
                pass

    def _camera_stream_failed(self) -> None:
        if not self.camera_running:
            return
        self._begin_camera_shutdown(
            open_file_dialog=False,
            status_message=(
                "Камера перестала передавать кадры. Поток остановлен. Проверьте драйвер и включите камеру снова."
            ),
        )

    def _poll_camera_frame(self) -> None:
        if not self.camera_running:
            return
        now = time.monotonic()
        with self._camera_lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
            frame_time = self._latest_frame_time

        if frame is not None and frame_time > self._last_displayed_frame_time:
            self.current_frame = frame
            self._last_displayed_frame_time = frame_time
            self._camera_stale_reported = False
            self._refresh_preview()
            self._set_status("Веб-камера работает", SUCCESS)
        elif (
            (self._latest_frame_time and now - self._latest_frame_time > 2.5)
            or (not self._latest_frame_time and self._camera_started_at and now - self._camera_started_at > 8.0)
        ):
            if not self._camera_stale_reported:
                self._set_status(
                    "Камера подключена, но кадры не поступают. Перезапустите камеру и проверьте разрешения Windows.",
                    WARNING,
                )
                self._camera_stale_reported = True

        self._camera_after_id = self.root.after(50, self._poll_camera_frame)

    def capture_camera(self) -> None:
        """Save the latest frame, then turn the camera fully off.

        This is the same stateful button that starts the webcam when it is off.
        """
        if not self.camera_running:
            return
        with self._camera_lock:
            latest = self._latest_frame.copy() if self._latest_frame is not None else None
            frame_time = self._latest_frame_time
        if latest is None or time.monotonic() - frame_time > 2.5:
            self._set_status("Нет свежего кадра. Проверьте веб-камеру и попробуйте ещё раз.", WARNING)
            return
        self.current_frame = latest
        self.current_path = None
        self._refresh_preview()
        self._begin_camera_shutdown(
            open_file_dialog=False,
            status_message="Снимок готов к анализу · веб-камера отключена",
        )

    def open_image(self) -> None:
        """Turn the webcam completely off before opening the Windows file picker.

        The camera is NOT restarted after selecting a file or cancelling the
        dialog. The user must explicitly press "Включить веб-камеру" again.
        """
        if self._camera_shutdown_pending or self._file_picker_pending:
            return

        if self.camera_running or self.camera_connecting or self.cv is not None:
            self._begin_camera_shutdown(
                open_file_dialog=True,
                status_message="Камера отключена перед выбором фотографии",
            )
            return

        # If the camera is already off, open the native dialog immediately.
        self.camera_button.configure(
            text="Включить веб-камеру", state="normal", command=self.start_camera
        )
        self._show_image_dialog()

    def _show_image_dialog(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self.root,
            title="Выберите фотографию образца",
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("Все файлы", "*.*"),
            ],
        )
        if not filename:
            self._set_status("Выбор фотографии отменён · веб-камера остаётся выключенной")
            return
        path = Path(filename)
        try:
            # fromfile + imdecode handles non-ASCII Windows paths more reliably.
            raw = np.fromfile(str(path), dtype=np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Не удалось прочитать изображение.")
        except Exception as exc:
            messagebox.showerror("Ошибка открытия", f"Не удалось открыть фотографию:\n{exc}")
            return
        self.current_frame = frame
        self.current_path = path
        self._refresh_preview()
        self._set_status(f"Выбрано: {path.name}", SUCCESS)

    @staticmethod
    def _image_to_data_url(frame: np.ndarray) -> str:
        # Resize very large photos to reduce request size and inference time.
        h, w = frame.shape[:2]
        max_side = 1600
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise ValueError("Не удалось подготовить фотографию для отправки модели.")
        b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def analyze_image(self) -> None:
        if self.current_frame is None:
            messagebox.showinfo("Нет изображения", "Сначала откройте фотографию или сделайте снимок веб-камерой.")
            return
        if self.analyze_button.cget("state") == "disabled":
            return

        frame_copy = self.current_frame.copy()
        specimen_condition = self.specimen_condition_var.get()
        self.analyze_button.configure(state="disabled", text="АНАЛИЗ…")
        self._set_status("Передаю фотографию модели в LM Studio…", WARNING)
        self._set_result("Модель анализирует изображение. Ожидание может занять до 10 минут…")

        def worker() -> None:
            try:
                image_url = self._image_to_data_url(frame_copy)
                system_prompt = (
                    "You are a practical field assistant for identifying mineral and rock specimens from photographs. "
                    "Write only in natural Russian and use standard Russian geological names in Cyrillic. "
                    "For jasper, always write 'яшма'; never write 'джаспер' or 'jasper'. "
                    "Do not use English specimen names or transliterations in the answer. "
                    "Describe only what is visible. Never invent composition, hardness, magnetism, test results, or locality. "
                    "Choose one best identification; if the image is insufficient, say that the identification is uncertain."
                )
                condition_context = {
                    "Не указано — определить по фотографии": (
                        "Состояние образца не задано. По фотографии определи, виден ли целый образец, "
                        "скол, мокрая или полированная поверхность, кристаллы либо фрагменты. Не угадывай "
                        "условия съёмки, если это нельзя установить уверенно."
                    ),
                    "Целый образец, сухой": (
                        "Пользователь указал: на фото целый сухой образец. Оценивай его общую форму, "
                        "цвет, текстуру и видимую структуру; не делай выводов о внутреннем строении, "
                        "которое не видно."
                    ),
                    "Свежий скол": (
                        "Пользователь указал: на фото свежий скол. В первую очередь оценивай поверхность "
                        "излома: однородность, зернистость, форму излома, блеск и видимые кристаллы. "
                        "Не описывай невидимую часть образца."
                    ),
                    "Образец смочен водой": (
                        "Пользователь указал: образец смочен водой. Учитывай, что вода может усиливать "
                        "насыщенность цвета и блеск. Не принимай потемнение от воды за самостоятельный "
                        "доказательный признак минерала или породы."
                    ),
                    "Полированная поверхность или срез": (
                        "Пользователь указал: на фото полированная поверхность или срез. Оценивай "
                        "видимый рисунок, полосчатость, включения, зернистость и распределение цветов; "
                        "не путай полировочный блеск с естественным блеском минерала."
                    ),
                    "Кристаллы или друза": (
                        "Пользователь указал: на фото кристаллы или друза. Оценивай форму кристаллов, "
                        "их взаимное расположение, прозрачность и блеск. Не называй точную кристаллическую "
                        "форму, если деталей недостаточно."
                    ),
                    "Образец в материнской породе": (
                        "Пользователь указал: образец показан вместе с материнской породой. Разделяй "
                        "основную породу и заметные включения или прожилки; не определяй весь образец "
                        "по одному отдельному включению."
                    ),
                    "Крошка или отдельные фрагменты": (
                        "Пользователь указал: на фото крошка или отдельные фрагменты. Оценивай повторяющиеся "
                        "цвет и текстуру фрагментов, но учитывай, что форма целого образца по ним неизвестна."
                    ),
                }.get(specimen_condition, "Состояние образца не задано; не делай неподтверждённых предположений.")

                user_prompt = (
                    f"Условия изображения: {specimen_condition}. {condition_context}\n\n"
                    "Identify the specimen in the attached photograph. Compare it with the owner's collection: "
                    "пирит, тигрово-соколиный глаз, соколиный глаз, тигровый глаз, нефрит, янтарь, бирюза, "
                    "симбирцит, горный хрусталь, кварц, кварцит, обсидиан, снежный обсидиан, яшма, магнетит, "
                    "аметист, мрамор, серпентинит и агат. Список неполный: не подгоняй ответ под него. "
                    "Различай минералы и горные породы.\n\n"
                    "Ответь только по-русски и строго по существу. Не объясняй, почему ты выбрал это название. "
                    "Не добавляй вступление, заключение, длинные рассуждения, общие предупреждения или посторонние примечания в скобках. "
                    "Не упоминай предметы и действия, не относящиеся к выбранной проверке. "
                    "Используй ровно такую структуру:\n"
                    "**Название:** одно наиболее вероятное название.\n"
                    "**Видимые признаки:** одно короткое предложение с 2–3 признаками, которые действительно видны.\n"
                    "**Проверка:** предложи только одну наиболее полезную полевую проверку для этого образца. "
                    "Кратко укажи, что сделать и что будет означать положительный или отрицательный результат. "
                    "Если полезной проверки среди перечисленных нет, напиши: 'Нужна дополнительная проверка'.\n"
                    "**Уверенность:** низкая, средняя или высокая.\n\n"
                    "Доступные полевые проверки:\n"
                    "- Свежий скол: рассмотреть внутреннюю поверхность. Для яшмы однородная мелкозернистая структура "
                    "без заметных отдельных кристаллов поддерживает это предположение; хорошо различимые зёрна или "
                    "кристаллы делают его менее вероятным.\n"
                    "- Царапание стекла: появление устойчивой царапины указывает на твёрдость выше примерно 5,5 по Моосу; "
                    "если царапины нет, образец может быть мягче.\n"
                    "- Вода: сравнить небольшой участок до и после смачивания. Потемнение показывает изменение вида "
                    "поверхности при намокании, но само по себе не определяет породу.\n"
                    "- Искры при ударе железом: предлагать только для кремня или похожей кремнистой породы. "
                    "Появление искр согласуется с таким материалом, но не доказывает его название.\n\n"
                    "Выбирай проверку по предполагаемому образцу, не перечисляй все варианты. Каждое поле — не более "
                    "одного-двух коротких предложений. Не используй скобочные замечания. Не предлагай кислоту, нагревание "
                    "или пробу на вкус. Не утверждай, что проверка уже выполнена."
                )

                with httpx.Client(
                    timeout=httpx.Timeout(connect=5.0, read=600.0, write=60.0, pool=10.0),
                    trust_env=False,
                ) as client:
                    models_response = client.get(f"{LM_STUDIO_URL}/v1/models")
                    models_response.raise_for_status()
                    models_payload = models_response.json()
                    model_ids = [item.get("id", "") for item in models_payload.get("data", [])]
                    model_id = self._choose_vision_model(model_ids)
                    if model_id is None:
                        raise RuntimeError(
                            "Не найдена поддерживаемая vision-модель. Загрузите GLM-4.6V-Flash "
                            "в LM Studio. LFM3-VL, Qwen и Gemma не используются в этом тесте."
                        )

                    payload = {
                        "model": model_id,
                        "system_prompt": system_prompt,
                        "input": [
                            {"type": "text", "content": user_prompt},
                            {"type": "image", "data_url": image_url},
                        ],
                        "temperature": 0.1,
                        "max_output_tokens": 3000,
                        "reasoning": "off",
                        "store": False,
                    }

                    native_data = None
                    native_error = None
                    try:
                        response = client.post(
                            f"{LM_STUDIO_URL}/api/v1/chat",
                            json=payload,
                            headers={"Content-Type": "application/json", "Accept": "application/json"},
                        )
                        response.raise_for_status()
                        native_data = response.json()
                    except httpx.HTTPStatusError as exc:
                        # Some model templates do not expose the reasoning switch.
                        # In that case, fall back to LM Studio's OpenAI-compatible route.
                        if exc.response.status_code not in (400, 422):
                            raise
                        native_error = f"native API HTTP {exc.response.status_code}"

                    answer = extract_lmstudio_final_text(native_data) if native_data is not None else ""
                    used_compat_api = False
                    if not answer:
                        # Retry through Chat Completions. This route returns the final
                        # assistant message in choices[0].message.content. Reasoning
                        # metadata is intentionally ignored.
                        compat_payload = {
                            "model": model_id,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": user_prompt},
                                        {"type": "image_url", "image_url": {"url": image_url}},
                                    ],
                                },
                            ],
                            "temperature": 0.1,
                            "max_tokens": 3000,
                            "stream": False,
                        }
                        compat_response = client.post(
                            f"{LM_STUDIO_URL}/v1/chat/completions",
                            json=compat_payload,
                            headers={"Content-Type": "application/json", "Accept": "application/json"},
                        )
                        compat_response.raise_for_status()
                        compat_data = compat_response.json()
                        answer = extract_openai_final_text(compat_data)
                        used_compat_api = True

                    if answer:
                        answer = normalize_russian_geology_terms(answer)

                    if not answer:
                        details = describe_native_output(native_data) if native_data is not None else (native_error or "native API не вернул сообщение")
                        raise RuntimeError(
                            "LM Studio не вернула итоговый текст ни через основной, ни через совместимый API. "
                            f"Служебные рассуждения не показываются. Диагностика: {details}."
                        )

                mode_note = " · совместимый API" if used_compat_api else ""
                result = (True, answer, f"Анализ выполнен моделью: {model_id}{mode_note}", model_id)
            except httpx.TimeoutException:
                result = (
                    False,
                    "LM Studio не успела завершить анализ за 10 минут. Попробуйте повторить запрос.",
                    "Превышено время ожидания",
                    None,
                )
            except httpx.HTTPStatusError as exc:
                detail = f"HTTP {exc.response.status_code}: {exc.response.text[:500]}"
                result = (False, f"Ошибка LM Studio: {detail}", "Ошибка анализа", None)
            except Exception as exc:
                result = (False, f"Не удалось выполнить анализ.\n\nПодробности: {exc}", "Ошибка анализа", None)
            try:
                self.root.after(0, lambda: self._finish_analysis(result))
            except tk.TclError:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def _finish_analysis(self, result) -> None:
        ok, text, status, model_id = result
        self._set_result(text)
        self.analyze_button.configure(state="normal", text="АНАЛИЗИРОВАТЬ ОБРАЗЕЦ")
        self._set_status(status, SUCCESS if ok else WARNING)
        if ok and model_id:
            self.active_model_id = model_id
            self.hint_label.configure(
                text="Полевые проверки: вода · свежий скол · царапание стекла · визуальные эффекты. Выбирайте подходящие."
            )
        else:
            self.hint_label.configure(
                text="Итоговый ответ не получен. Проверьте журнал LM Studio и настройки шаблона модели."
            )

    def _on_close(self) -> None:
        self._closing = True
        self._camera_open_cancel.set()
        self.camera_running = False
        self._camera_stop_event.set()
        if self._camera_after_id:
            try:
                self.root.after_cancel(self._camera_after_id)
            except tk.TclError:
                pass
            self._camera_after_id = None
        camera = self.cv
        self.cv = None
        if camera is not None:
            threading.Thread(target=self._release_camera_handle, args=(camera,), daemon=True).start()
        if self._server_check_after_id:
            try:
                self.root.after_cancel(self._server_check_after_id)
            except tk.TclError:
                pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    RockAIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

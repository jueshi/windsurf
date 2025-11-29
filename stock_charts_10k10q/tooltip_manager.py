"""Centralized tooltip manager for Tkinter widgets used in StockDataGUI."""

from __future__ import annotations

import tkinter as tk
import weakref
from typing import Callable, Dict, Optional, Tuple


TooltipTextProvider = Callable[[], str]


class TooltipManager:
    """Manage hover and keyboard-triggered tooltips for Tkinter widgets."""

    def __init__(
        self,
        root: tk.Misc,
        hover_delay_ms: int = 300,
        display_duration_ms: int = 5000,
        cursor_offset: Tuple[int, int] = (12, 18),
    ) -> None:
        self.root = root
        self.hover_delay_ms = hover_delay_ms
        self.display_duration_ms = display_duration_ms
        self.cursor_offset = cursor_offset

        self._registry: "weakref.WeakKeyDictionary[tk.Misc, Dict[str, object]]" = weakref.WeakKeyDictionary()
        self._tooltip_window: Optional[tk.Toplevel] = None
        self._current_widget: Optional[tk.Misc] = None
        self._show_job: Optional[str] = None
        self._hide_job: Optional[str] = None
        self._enabled = True

        # Allow keyboard access via Shift+F1 for the focused widget
        try:
            self.root.bind_all("<Shift-F1>", self._handle_keyboard_request, add="+")
        except Exception:
            # bind_all can fail if root is destroyed during teardown
            pass

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        if not self._enabled:
            self._hide_tooltip()

    def attach(
        self,
        widget: tk.Misc,
        *,
        text: Optional[str] = None,
        text_provider: Optional[TooltipTextProvider] = None,
        tooltip_id: Optional[str] = None,
    ) -> None:
        """Attach tooltip metadata and event bindings to a widget."""

        if text is None and text_provider is None:
            raise ValueError("Either text or text_provider must be supplied for a tooltip")

        info: Dict[str, object] = {
            "text": text,
            "text_provider": text_provider,
            "id": tooltip_id,
            "last_event": None,
        }
        self._registry[widget] = info

        def _on_enter(event: tk.Event) -> None:
            self._schedule_show(widget, event)

        def _on_motion(event: tk.Event) -> None:
            self._update_motion(widget, event)

        def _on_leave(event: tk.Event) -> None:
            self._cancel_scheduled_show()
            if self._current_widget is widget:
                self._hide_tooltip()

        def _on_destroy(event: tk.Event) -> None:
            self.detach(widget)

        widget.bind("<Enter>", _on_enter, add="+")
        widget.bind("<Leave>", _on_leave, add="+")
        widget.bind("<Motion>", _on_motion, add="+")
        widget.bind("<Destroy>", _on_destroy, add="+")

    def detach(self, widget: tk.Misc) -> None:
        if widget in self._registry:
            del self._registry[widget]
        if widget is self._current_widget:
            self._hide_tooltip()

    def enumerate_tooltips(self) -> Dict[str, str]:
        """Return a mapping of tooltip IDs to static text for diagnostics."""
        listing: Dict[str, str] = {}
        for widget, info in self._registry.items():
            tooltip_id = info.get("id") or widget.winfo_name()
            listing[tooltip_id] = self._resolve_text(widget, info)
        return listing

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _schedule_show(self, widget: tk.Misc, event: tk.Event) -> None:
        if not self._enabled or widget not in self._registry:
            return
        self._registry[widget]["last_event"] = event
        self._cancel_scheduled_show()
        self._show_job = self.root.after(self.hover_delay_ms, lambda: self._show_tooltip(widget))

    def _update_motion(self, widget: tk.Misc, event: tk.Event) -> None:
        if widget not in self._registry:
            return
        self._registry[widget]["last_event"] = event
        if widget is self._current_widget and self._tooltip_window is not None:
            self._position_tooltip(widget, event)

    def _show_tooltip(self, widget: tk.Misc, *, explicit_position: Optional[Tuple[int, int]] = None) -> None:
        if not self._enabled or widget not in self._registry:
            return

        info = self._registry[widget]
        text = self._resolve_text(widget, info)
        if not text:
            return

        self._hide_tooltip()
        self._current_widget = widget

        tooltip = tk.Toplevel(self.root)
        tooltip.wm_overrideredirect(True)
        tooltip.attributes("-topmost", True)

        label = tk.Label(
            tooltip,
            text=text,
            justify="left",
            background="#fefbe7",
            foreground="#333333",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9),
            padx=8,
            pady=4,
            wraplength=360,
        )
        label.pack()

        self._tooltip_window = tooltip

        if explicit_position is not None:
            x, y = explicit_position
        else:
            event = info.get("last_event")
            x, y = self._derive_position_from_event(widget, event)

        tooltip.wm_geometry(f"+{x}+{y}")
        self._schedule_hide()

    def _schedule_hide(self) -> None:
        self._cancel_scheduled_hide()
        self._hide_job = self.root.after(self.display_duration_ms, self._hide_tooltip)

    def _hide_tooltip(self) -> None:
        self._cancel_scheduled_show()
        self._cancel_scheduled_hide()
        if self._tooltip_window is not None:
            try:
                self._tooltip_window.destroy()
            except Exception:
                pass
        self._tooltip_window = None
        self._current_widget = None

    def _cancel_scheduled_show(self) -> None:
        if self._show_job is not None:
            try:
                self.root.after_cancel(self._show_job)
            except Exception:
                pass
        self._show_job = None

    def _cancel_scheduled_hide(self) -> None:
        if self._hide_job is not None:
            try:
                self.root.after_cancel(self._hide_job)
            except Exception:
                pass
        self._hide_job = None

    def _derive_position_from_event(self, widget: tk.Misc, event: Optional[tk.Event]) -> Tuple[int, int]:
        if event is not None:
            x = event.x_root + self.cursor_offset[0]
            y = event.y_root + self.cursor_offset[1]
            return x, y
        try:
            x = widget.winfo_rootx() + self.cursor_offset[0]
            y = widget.winfo_rooty() + widget.winfo_height() + self.cursor_offset[1]
            return x, y
        except Exception:
            return 0, 0

    def _position_tooltip(self, widget: tk.Misc, event: tk.Event) -> None:
        if self._tooltip_window is None:
            return
        x, y = self._derive_position_from_event(widget, event)
        try:
            self._tooltip_window.wm_geometry(f"+{x}+{y}")
        except Exception:
            pass

    def _resolve_text(self, widget: tk.Misc, info: Dict[str, object]) -> str:
        provider = info.get("text_provider")
        if callable(provider):
            try:
                text = provider()
                if text:
                    return text
            except Exception:
                return ""
        text = info.get("text")
        return text or ""

    def _handle_keyboard_request(self, event: tk.Event) -> None:
        if not self._enabled:
            return
        widget = self.root.focus_get()
        if not widget or widget not in self._registry:
            return
        try:
            x = widget.winfo_rootx() + self.cursor_offset[0]
            y = widget.winfo_rooty() + widget.winfo_height() + self.cursor_offset[1]
        except Exception:
            x = event.x_root + self.cursor_offset[0]
            y = event.y_root + self.cursor_offset[1]
        self._show_tooltip(widget, explicit_position=(x, y))


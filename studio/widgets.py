"""Themed Tk widgets."""
from __future__ import annotations

import tkinter as tk

from studio.config import (
    ACCENT,
    ACCENT_HOVER,
    BG,
    BG_CARD,
    BG_HOVER,
    BG_INPUT,
    BG_PANEL,
    BORDER,
    FG,
    FONT,
    FG_DIM,
)

# ── Styled widgets ───────────────────────────────────────────────────────

class FlatButton(tk.Canvas):
    def __init__(
        self, parent, *, text="", command=None, bg=ACCENT, bg_hover=ACCENT_HOVER,
        fg="#ffffff", width=90, height=32, font_size=9, bold=False, radius=6,
    ) -> None:
        self._bg = bg
        self._bg_hover = bg_hover
        self._fg = fg
        self._text = text
        self._command = command
        self._radius = radius
        weight = "bold" if bold else "normal"
        self._font = (FONT, font_size, weight)
        parent_bg = BG_PANEL
        if isinstance(parent, (tk.Frame, tk.Canvas)):
            try:
                parent_bg = parent.cget("bg")
            except Exception:
                pass
        super().__init__(parent, width=width, height=height,
                         bg=parent_bg, highlightthickness=0, bd=0)
        self._draw(self._bg)
        self.bind("<Enter>", lambda _: self._draw(self._bg_hover))
        self.bind("<Leave>", lambda _: self._draw(self._bg))
        self.bind("<ButtonPress-1>", lambda _: self._click())
        self.bind("<ButtonRelease-1>", lambda _: self._draw(self._bg_hover))

    def _draw(self, color: str) -> None:
        self.delete("all")
        w = int(self.cget("width"))
        h = int(self.cget("height"))
        r = self._radius
        pts = self._rrect_pts(0, 0, w, h, r)
        self.create_polygon(pts, smooth=True, fill=color, outline="")
        self.create_text(w // 2, h // 2, text=self._text,
                         fill=self._fg, font=self._font)

    @staticmethod
    def _rrect_pts(x1, y1, x2, y2, r):
        return [
            x1+r, y1, x2-r, y1, x2, y1, x2, y1+r,
            x2, y2-r, x2, y2, x2-r, y2, x1+r, y2,
            x1, y2, x1, y2-r, x1, y1+r, x1, y1,
        ]

    def _click(self) -> None:
        if self._command:
            self._command()

    def configure_text(self, text: str) -> None:
        self._text = text
        self._draw(self._bg)

    def configure_colors(self, *, bg: str | None = None, bg_hover: str | None = None,
                         fg: str | None = None) -> None:
        if bg is not None:
            self._bg = bg
        if bg_hover is not None:
            self._bg_hover = bg_hover
        if fg is not None:
            self._fg = fg
        self._draw(self._bg)


class FlatLabel(tk.Label):
    def __init__(self, parent, **kw) -> None:
        kw.setdefault("bg", BG_PANEL)
        kw.setdefault("fg", FG)
        kw.setdefault("font", (FONT, 9))
        super().__init__(parent, **kw)


class FlatOptionMenu(tk.Frame):
    def __init__(self, parent, variable: tk.StringVar, *values, width=10) -> None:
        super().__init__(parent, bg=BG_INPUT, bd=0, highlightthickness=1,
                         highlightbackground=BORDER, highlightcolor=ACCENT)
        self._var = variable
        self._values = list(values)
        self._menu_open = False
        self._list_win: tk.Toplevel | None = None

        self._label = tk.Label(self, textvariable=variable, bg=BG_INPUT, fg=FG,
                               font=(FONT, 9), anchor="w", padx=8)
        self._label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=5)

        arrow = tk.Label(self, text="\u25BE", bg=BG_INPUT, fg=FG_DIM, font=(FONT, 10))
        arrow.pack(side=tk.RIGHT, padx=(0, 6))

        for w in (self, self._label, arrow):
            w.bind("<ButtonPress-1>", self._toggle)

    def _toggle(self, _event=None) -> None:
        if self._menu_open and self._list_win:
            self._close_menu()
            return
        self._open_menu()

    def _open_menu(self) -> None:
        if not self._values:
            return
        self._menu_open = True
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        win = tk.Toplevel(self)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.configure(bg=BG_CARD)
        win.geometry(f"+{x}+{y}")
        for val in self._values:
            lbl = tk.Label(win, text=val, bg=BG_CARD, fg=FG,
                           font=(FONT, 9), anchor="w", padx=14, pady=5, cursor="hand2")
            lbl.pack(fill=tk.X)
            lbl.bind("<Enter>", lambda e, l=lbl: l.configure(bg=BG_HOVER))
            lbl.bind("<Leave>", lambda e, l=lbl: l.configure(bg=BG_CARD))
            lbl.bind("<ButtonPress-1>", lambda e, v=val: self._pick(v))
        win.bind("<FocusOut>", lambda _: self._close_menu())
        self._list_win = win
        win.focus_set()

    def _pick(self, value: str) -> None:
        self._var.set(value)
        self._close_menu()

    def _close_menu(self) -> None:
        self._menu_open = False
        if self._list_win:
            self._list_win.destroy()
            self._list_win = None

    def update_values(self, values: list[str]) -> None:
        self._values = values


# ── Main application ─────────────────────────────────────────────────────


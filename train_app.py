#!/usr/bin/env python3
"""YOLO training UI only (Train tab)."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path

from tkinter import messagebox

from studio.config import BG
from studio.train_tab import TrainingTab


class TrainStudioApp:
    """Train tab host — loading finished weights opens instructions for Viewer."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("POI Studio — Train")
        root.geometry("1200x820")
        root.minsize(900, 600)
        root.configure(bg=BG)
        self.trainer = TrainingTab(root, self)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def load_trained_model(self, path: str) -> None:
        p = Path(path)
        msg = (
            "Training finished.\n\n"
            "Open viewer_app.py (or the unified poi_studio.py) and use Model → load weights,\n"
            f"or choose this file:\n{p.resolve()}"
        )
        messagebox.showinfo("Load weights in Viewer", msg)

    def _on_close(self) -> None:
        self.trainer._on_close()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    TrainStudioApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

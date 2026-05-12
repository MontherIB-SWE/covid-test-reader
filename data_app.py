#!/usr/bin/env python3
"""Label + dataset pipeline (Data tab only)."""
from __future__ import annotations

import tkinter as tk

from studio.config import BG
from studio.data_tab import DataManagementTab


class DataStudioApp:
    """Minimal host without Run/Train — labeling + dataset prep."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("POI Studio — Data")
        root.geometry("1500x920")
        root.minsize(1200, 700)
        root.configure(bg=BG)
        self.trainer = None  # no TrainingTab
        self.data_tab = DataManagementTab(root, self)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self) -> None:
        self.data_tab._on_close()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    DataStudioApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

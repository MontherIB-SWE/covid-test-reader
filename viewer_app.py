#!/usr/bin/env python3
"""POI Desktop Viewer only (inference / Run surface)."""
from __future__ import annotations

import tkinter as tk

from studio.config import BG
from studio.viewer import PoiDesktopViewer


def main() -> None:
    root = tk.Tk()
    root.configure(bg=BG)
    PoiDesktopViewer(root, embedded=False, bind_shortcuts=True)
    root.mainloop()


if __name__ == "__main__":
    main()

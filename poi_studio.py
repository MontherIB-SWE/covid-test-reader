#!/usr/bin/env python3
"""Unified POI Studio launcher (Run + Data + Train in one window).

For split workflows see viewer_app.py, data_app.py, train_app.py.
The previous single-file build is kept as poi_studio_monolith.py.
"""
from __future__ import annotations

from pathlib import Path


def main() -> None:
    try:
        import tkinter as tk

        from studio.shell import PoiStudio

        root = tk.Tk()
        PoiStudio(root)
        root.mainloop()
    except Exception:
        import traceback

        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "app.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print("App crashed. See logs/app.log for details.")


if __name__ == "__main__":
    main()

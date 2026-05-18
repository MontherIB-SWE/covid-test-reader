#!/usr/bin/env python3
"""POI Studio launcher (PySide6: Run + Data + Train)."""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    try:
        from studio.qt.shell import run_app
        return run_app()
    except Exception:
        import traceback
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "app.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print("App crashed. See logs/app.log for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Phase 2: Fix H2 (pipeline params), M1-M7, L1-L3."""
import pathlib

ROOT = pathlib.Path(r"C:\Users\mndr6\Downloads\archive")

def fix_file(rel_path: str, replacements: list[tuple[bytes, bytes]], label: str) -> None:
    p = ROOT / rel_path
    data = p.read_bytes()
    applied = 0
    for old, new in replacements:
        if old in data:
            data = data.replace(old, new)
            applied += 1
        else:
            print(f"  WARN not found in {label}: {old[:80]}")
    p.write_bytes(data)
    print(f"  [{label}] Applied {applied}/{len(replacements)}")

# ══════════════════════════════════════════════════════════════════════
# H2: pipeline.generate() — accept directory params instead of globals
# ══════════════════════════════════════════════════════════════════════

# 1. Add typing import for Callable (fixes M1)
fix_file("studio/pipeline.py", [
    (b"from __future__ import annotations\n",
     b"from __future__ import annotations\n\nfrom typing import Callable\n"),
], "M1 pipeline Callable import")

# 2. Change generate() signature to accept dirs as params
fix_file("studio/pipeline.py", [
    (b"def generate(\n    num_images: int = 10,\n    on_saved: Callable[[Path], None] | None = None,\n) -> None:\n    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)\n    OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)\n\n    bg_files = _get_files(BG_DIR)\n    asset_files = _get_files(ASSETS_DIR)",
     b"def generate(\n    num_images: int = 10,\n    on_saved: Callable[[Path], None] | None = None,\n    bg_dir: Path | None = None,\n    assets_dir: Path | None = None,\n    out_img_dir: Path | None = None,\n    out_lbl_dir: Path | None = None,\n) -> None:\n    _bg_dir = bg_dir if bg_dir is not None else BG_DIR\n    _assets_dir = assets_dir if assets_dir is not None else ASSETS_DIR\n    _out_img_dir = out_img_dir if out_img_dir is not None else OUT_IMG_DIR\n    _out_lbl_dir = out_lbl_dir if out_lbl_dir is not None else OUT_LBL_DIR\n    _out_img_dir.mkdir(parents=True, exist_ok=True)\n    _out_lbl_dir.mkdir(parents=True, exist_ok=True)\n\n    bg_files = _get_files(_bg_dir)\n    asset_files = _get_files(_assets_dir)"),
], "H2 pipeline.generate signature + locals")

# 3. Fix error messages to use local vars
fix_file("studio/pipeline.py", [
    (b'print(f"Error: No backgrounds found in {BG_DIR}")',
     b'print(f"Error: No backgrounds found in {_bg_dir}")'),
    (b'print(f"Error: No POI assets found in {ASSETS_DIR}")',
     b'print(f"Error: No POI assets found in {_assets_dir}")'),
], "H2 pipeline error messages")

# 4. Fix _find_start_index call to pass dir, and OUT_IMG_DIR/OUT_LBL_DIR refs in generate body
fix_file("studio/pipeline.py", [
    (b"    start_idx = _find_start_index()",
     b"    start_idx = _find_start_index(_out_img_dir)"),
    (b"        out_img_path = OUT_IMG_DIR / f\"{out_name}.jpg\"",
     b"        out_img_path = _out_img_dir / f\"{out_name}.jpg\""),
], "H2 pipeline body refs")

# 5. Fix the final OUT_LBL_DIR ref and the print at end
fix_file("studio/pipeline.py", [
    (b"        (OUT_LBL_DIR / f\"{out_name}.txt\").write_text(label_text)",
     b"        (_out_lbl_dir / f\"{out_name}.txt\").write_text(label_text)"),
    (b"    print(f\"Output saved to: {OUT_IMG_DIR.resolve()}\")",
     b"    print(f\"Output saved to: {_out_img_dir.resolve()}\")"),
], "H2 pipeline final refs")

# 6. Fix _find_start_index to accept dir param
fix_file("studio/pipeline.py", [
    (b"def _find_start_index() -> int:\n    \"\"\"Find the next available index by scanning existing output files.\"\"\"\n    if not OUT_IMG_DIR.exists():\n        return 0\n    max_idx = -1\n    for f in OUT_IMG_DIR.iterdir():",
     b"def _find_start_index(out_dir: Path | None = None) -> int:\n    \"\"\"Find the next available index by scanning existing output files.\"\"\"\n    _dir = out_dir if out_dir is not None else OUT_IMG_DIR\n    if not _dir.exists():\n        return 0\n    max_idx = -1\n    for f in _dir.iterdir():"),
], "H2 _find_start_index param")

# 7. Fix data_tab.py — pass params instead of monkey-patching
fix_file("studio/data_tab.py", [
    (b"                pipeline.BG_DIR = backgrounds_dir\n                pipeline.ASSETS_DIR = cache_assets\n                pipeline.OUT_IMG_DIR = out_img\n                pipeline.OUT_LBL_DIR = out_lbl\n                out_img.mkdir(parents=True, exist_ok=True)\n                out_lbl.mkdir(parents=True, exist_ok=True)\n\n                pipeline.generate(num_images=n, on_saved=None)",
     b"                out_img.mkdir(parents=True, exist_ok=True)\n                out_lbl.mkdir(parents=True, exist_ok=True)\n\n                pipeline.generate(\n                    num_images=n,\n                    on_saved=None,\n                    bg_dir=backgrounds_dir,\n                    assets_dir=cache_assets,\n                    out_img_dir=out_img,\n                    out_lbl_dir=out_lbl,\n                )"),
], "H2 data_tab pass params")

print("\nPhase 2a (H2) done")

"""Fix all 17 issues across the project."""
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
            print(f"  WARN not found in {label}: {old[:60]}")
    p.write_bytes(data)
    print(f"  [{label}] Applied {applied}/{len(replacements)} replacements")

# ── C1: relabel.py:658 — self._redraw() → self.render() ──
fix_file("studio/relabel.py", [
    (b"self._redraw()\r\n            self._mark_dirty()",
     b"self.render()\r\n            self._mark_dirty()"),
    (b"self._redraw()\n            self._mark_dirty()",
     b"self.render()\n            self._mark_dirty()"),
], "C1 relabel._redraw")

# ── C2+C3: viewer.py — add os and sys imports ──
fix_file("studio/viewer.py", [
    # Add os, sys to the imports block
    (b"import json\r\nimport queue\r\nimport threading\r\nimport time",
     b"import json\r\nimport os\r\nimport queue\r\nimport sys\r\nimport threading\r\nimport time"),
    (b"import json\nimport queue\nimport threading\nimport time",
     b"import json\nimport os\nimport queue\nimport sys\nimport threading\nimport time"),
], "C2+C3 viewer imports")

# ── H1: viewer.py:815 — use POI_COLORS_HEX instead of manual BGR→hex ──
fix_file("studio/viewer.py", [
    # Add POI_COLORS_HEX to imports
    (b"    POI_COLORS,\r\n    RED,",
     b"    POI_COLORS,\r\n    POI_COLORS_HEX,\r\n    RED,"),
    (b"    POI_COLORS,\n    RED,",
     b"    POI_COLORS,\n    POI_COLORS_HEX,\n    RED,"),
    # Replace the wrong hex conversion with the correct one
    (b'color = POI_COLORS[i % len(POI_COLORS)]\r\n            color_hex = "#{:02x}{:02x}{:02x}".format(*color)',
     b'color = POI_COLORS[i % len(POI_COLORS)]\r\n            color_hex = POI_COLORS_HEX[i % len(POI_COLORS_HEX)]'),
    (b'color = POI_COLORS[i % len(POI_COLORS)]\n            color_hex = "#{:02x}{:02x}{:02x}".format(*color)',
     b'color = POI_COLORS[i % len(POI_COLORS)]\n            color_hex = POI_COLORS_HEX[i % len(POI_COLORS_HEX)]'),
], "H1 viewer color hex")

# ── H3: shell.py — shutil.move → shutil.copy2 for save-to-train ──
fix_file("studio/shell.py", [
    (b"shutil.move(str(image_path), str(out_image))",
     b"shutil.copy2(str(image_path), str(out_image))"),
    (b"shutil.move(str(label_path), str(out_label))",
     b"shutil.copy2(str(label_path), str(out_label))"),
], "H3 shell move→copy2")

print("\nPhase 1 (critical + high simple fixes) done")

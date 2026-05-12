"""Fix data_tab.py monkey-patching — CRLF version."""
import pathlib
p = pathlib.Path(r"C:\Users\mndr6\Downloads\archive\studio\data_tab.py")
data = p.read_bytes()

old = (
    b"pipeline.BG_DIR = backgrounds_dir\r\n"
    b"                pipeline.ASSETS_DIR = cache_assets\r\n"
    b"                pipeline.OUT_IMG_DIR = out_img\r\n"
    b"                pipeline.OUT_LBL_DIR = out_lbl\r\n"
    b"                out_img.mkdir(parents=True, exist_ok=True)\r\n"
    b"                out_lbl.mkdir(parents=True, exist_ok=True)\r\n"
    b"\r\n"
    b"                pipeline.generate(num_images=n, on_saved=None)"
)

new = (
    b"out_img.mkdir(parents=True, exist_ok=True)\r\n"
    b"                out_lbl.mkdir(parents=True, exist_ok=True)\r\n"
    b"\r\n"
    b"                pipeline.generate(\r\n"
    b"                    num_images=n,\r\n"
    b"                    on_saved=None,\r\n"
    b"                    bg_dir=backgrounds_dir,\r\n"
    b"                    assets_dir=cache_assets,\r\n"
    b"                    out_img_dir=out_img,\r\n"
    b"                    out_lbl_dir=out_lbl,\r\n"
    b"                )"
)

if old in data:
    data = data.replace(old, new)
    p.write_bytes(data)
    print("OK: data_tab.py monkey-patching replaced with params")
else:
    # Try LF
    old_lf = old.replace(b"\r\n", b"\n")
    new_lf = new.replace(b"\r\n", b"\n")
    if old_lf in data:
        data = data.replace(old_lf, new_lf)
        p.write_bytes(data)
        print("OK: data_tab.py (LF) monkey-patching replaced with params")
    else:
        print("NOT FOUND — dumping context")
        idx = data.find(b"pipeline.BG_DIR")
        if idx >= 0:
            print(repr(data[idx-20:idx+300]))

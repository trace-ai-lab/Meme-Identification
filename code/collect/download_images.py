"""Download the meme images listed in data/memes.csv (the images themselves are not distributed).

  python download_images.py --memes ../../data/memes.csv --out images
"""
import argparse
import csv
import os
import sys
import time

import requests

csv.field_size_limit(sys.maxsize)
EXT = {"image/jpeg": "jpg", "image/png": "png", "image/gif": "gif", "image/webp": "webp"}
FIELDS = ["idx", "image_url", "status", "file", "bytes", "content_type", "width", "height"]


def size_of(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        return "", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memes", default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "memes.csv"))
    ap.add_argument("--out", default="images")
    ap.add_argument("--sleep", type=float, default=0.5, help="seconds between requests")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man_path = os.path.join(a.out, "manifest.csv")
    old = {}
    if os.path.exists(man_path):
        old = {r["idx"]: r for r in csv.DictReader(open(man_path, encoding="utf-8"))}
    rows = list(csv.DictReader(open(a.memes, encoding="utf-8-sig")))
    out = []
    for r in rows:
        idx, url = r["row_idx"], (r["image_url"] or "").strip()
        prev = old.get(idx)
        if prev and prev["status"] == "ok" and os.path.exists(prev["file"]):
            out.append(prev)
            continue
        rec = {"idx": idx, "image_url": url, "status": "", "file": "", "bytes": "", "content_type": "",
               "width": "", "height": ""}
        if not url:
            rec["status"] = "no_url"
        else:
            try:
                resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
                ctype = resp.headers.get("Content-Type", "").split(";")[0].strip()
                if resp.status_code != 200:
                    rec["status"] = f"http_{resp.status_code}"
                elif not ctype.startswith("image/"):
                    rec["status"] = f"not_an_image:{ctype}"
                else:
                    ext = EXT.get(ctype) or os.path.splitext(url)[1].lstrip(".").lower() or "img"
                    path = os.path.abspath(os.path.join(a.out, f"{int(idx):04d}.{ext}"))
                    with open(path, "wb") as f:
                        f.write(resp.content)
                    w, h = size_of(path)
                    rec.update(status="ok", file=path, bytes=len(resp.content), content_type=ctype, width=w, height=h)
            except Exception as e:
                rec["status"] = f"error:{type(e).__name__}"
            time.sleep(a.sleep)
        out.append(rec)
        print(idx, rec["status"], flush=True)
    with open(man_path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(out)
    n_ok = sum(1 for x in out if x["status"] == "ok")
    print(f"{n_ok} of {len(out)} images downloaded; manifest: {man_path}")


if __name__ == "__main__":
    main()

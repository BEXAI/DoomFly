#!/usr/bin/env python3
"""Download the pre-generated feed clips listed in a JSON manifest into assets/clips/.

    python3 src/fetch_clips.py assets/clips/manifest.json

Manifest: {"clips": [{"name": "00_dance", "url": "https://...", "prompt": "...", "model": "..."}]}
Files are saved as <name>.mp4; existing files with the expected size are kept.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = os.path.join(ROOT, "assets", "clips")


def main(manifest: str) -> None:
    with open(manifest) as f:
        m = json.load(f)
    os.makedirs(CLIPS, exist_ok=True)
    for c in m["clips"]:
        dst = os.path.join(CLIPS, c["name"] + ".mp4")
        with urllib.request.urlopen(c["url"]) as r:
            expected = int(r.headers.get("Content-Length", "0"))
            if os.path.exists(dst) and expected and os.path.getsize(dst) == expected:
                print("keep", dst); continue
            data = r.read()
        if expected and len(data) != expected:
            raise RuntimeError(f"{c['name']}: got {len(data)} bytes, expected {expected}")
        with open(dst, "wb") as f:
            f.write(data)
        print("saved", dst, len(data) // 1024, "KB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(CLIPS, "manifest.json"))

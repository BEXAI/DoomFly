#!/usr/bin/env python3
"""Download the three MaleCNS v1.0 flat-connectome files (~1.1 GB total)."""
import os, sys, urllib.request
BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FILES = [
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
]
def main(dst="data/raw"):
    os.makedirs(dst, exist_ok=True)
    for f in FILES:
        p = os.path.join(dst, f)
        if os.path.exists(p):
            print("exists", p); continue
        print("downloading", f, flush=True)
        urllib.request.urlretrieve(f"{BASE}/{f}", p + ".part")
        os.rename(p + ".part", p)
        print("done", f, os.path.getsize(p) // 2**20, "MB", flush=True)
if __name__ == "__main__":
    main(*sys.argv[1:])

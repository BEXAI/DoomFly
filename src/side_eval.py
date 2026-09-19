#!/usr/bin/env python3
"""Open-loop side test: score the decoder's `would_swipe` decisions from a --scripted episode.

    python3 src/run_episode.py --duration 30 --tag side_real --no-spikes --scripted L:15,R:15 --teacher-gap 1.2 2.0
    python3 src/side_eval.py side_real [side_shuf ...]

For each scripted block (panel P for S seconds) the decoder is correct when its would-be swipe
goes to P. Blocks are read from the meta line's `scripted` field (default "L:15,R:15").
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate(tag: str, out_dir: str) -> dict:
    lines = [json.loads(l) for l in open(os.path.join(out_dir, f"events_{tag}.jsonl"))]
    meta = lines[0].get("meta", {}) if lines and "meta" in lines[0] else {}
    ev = lines[1:] if meta else lines
    script = meta.get("scripted") or "L:15,R:15"
    blocks, t0 = [], 0.0
    for blk in script.split(","):
        p, secs = blk.split(":")
        blocks.append((t0, t0 + float(secs), p)); t0 += float(secs)
    res = {}
    for a, b, p in blocks:
        dec = [e["would_swipe"] for e in ev if e.get("would_swipe") and a <= e["t"] < b]
        res[p] = (sum(1 for d in dec if d == p), len(dec))
    ok = sum(v[0] for v in res.values()); n = sum(v[1] for v in res.values())
    print(f"{tag}: decoder side correct  " + "  ".join(f"{p}-block {v[0]}/{v[1]}" for p, v in res.items())
          + f"  overall {ok}/{n} = {ok / max(1, n):.2f}")
    return dict(tag=tag, blocks=res, correct=ok, total=n, fraction=ok / max(1, n), scripted=script)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "out"))
    a = ap.parse_args()
    results = [evaluate(t, a.out_dir) for t in a.tags]
    with open(os.path.join(a.out_dir, "side_eval_" + "_".join(a.tags) + ".json"), "w") as f:
        json.dump(results, f, indent=1)

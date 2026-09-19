#!/usr/bin/env python3
"""Summarise an episode log (out/events_<tag>.jsonl): swipe statistics, whether each
swipe went to the eye that saw the onset, DN burst / swipe timing, and a figure.

    python3 src/analyze_episode.py final [shuffled ...]   -> out/episode_<tag>.png, out/episode_<tag>.json
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(tag: str, out_dir: str):
    lines = [json.loads(l) for l in open(os.path.join(out_dir, f"events_{tag}.jsonl"))]
    if lines and "meta" in lines[0] and "step" not in lines[0]:
        meta, ev = lines[0]["meta"], lines[1:]
    else:
        meta, ev = {}, lines
    if not ev:
        raise SystemExit(f"events_{tag}.jsonl contains no control steps")
    return meta, ev


def analyse(tag: str, out_dir: str = os.path.join(ROOT, "out")) -> dict:
    meta, ev = load(tag, out_dir)
    dt = meta.get("control_dt", 0.016)
    t = np.array([e["t"] for e in ev])
    eyeL = np.array([sum(e["eye"]["L"]) for e in ev]); eyeR = np.array([sum(e["eye"]["R"]) for e in ev])
    dn = np.array([e["pops"]["dn_L"] + e["pops"]["dn_R"] for e in ev], float)
    leg = {p: np.array([e["pops"]["leg_" + p] for e in ev], float) for p in "LR"}
    tot = np.array([e["spikes"] for e in ev], float)
    n_dn = int(meta.get("n_dn", 1314))
    leg_n = meta.get("leg_pool", {"L": 68, "R": 67})
    dt_sim = float(meta.get("lif", {}).get("dt_ms", 2.0)) / 1000.0
    dn_hz = dn / n_dn / dt
    swipes = [(e["t"], e["swipe"], i) for i, e in enumerate(ev) if e["swipe"]]

    # side check: did the swipe go to the eye with more input in the preceding 256 ms (16 steps)?
    match = []
    for ts, side, i in swipes:
        w = slice(max(0, i - 16), i)
        L, R = (eyeL[w].mean(), eyeR[w].mean()) if i > 0 else (0.0, 0.0)
        match.append(L != R and (side == "L") == (L > R))   # a tie never counts as a match
    # burst -> swipe latency: for each swipe, time since DN rate last crossed 20% of its peak
    # DN response around swipes
    lags = np.arange(-63, 40)   # -63 steps = 1.0 s before the swipe
    dn_around = np.array([[dn_hz[i + l] if 0 <= i + l < len(dn_hz) else np.nan for l in lags] for _, _, i in swipes]) if swipes else np.zeros((0, lags.size))
    # chain statistics
    ts = np.array([s[0] for s in swipes])
    gaps = np.diff(ts) if len(ts) > 1 else np.array([])
    res = dict(
        tag=tag, duration=float(t[-1] + dt), n_swipes=len(swipes),
        swipes_L=sum(1 for s in swipes if s[1] == "L"), swipes_R=sum(1 for s in swipes if s[1] == "R"),
        swipe_times=[(round(s[0], 3), s[1]) for s in swipes],
        side_matches_eye_onset=int(sum(match)), side_match_fraction=float(np.mean(match)) if match else None,
        median_gap_s=float(np.median(gaps)) if gaps.size else None,
        chains_gap_lt_0p6s=int((gaps < 0.6).sum()) if gaps.size else 0,
        posts=ev[-1]["posts"], total_spikes=int(tot.sum()),
        mean_pop_rate_hz=float(tot.sum() / len(ev) / meta.get("substeps", 8) / meta.get("n_neurons", 166700) / dt_sim),
        dn_rate_hz_mean=float(dn_hz.mean()), dn_rate_hz_peak=float(dn_hz.max()),
        dn_rate_at_swipe_hz=float(np.nanmean(dn_around[:, lags == 0])) if swipes else None,
        dn_rate_1s_before_swipe_hz=float(np.nanmean(dn_around[:, (lags >= -63) & (lags < -56)])) if swipes else None,   # 1.0-0.9 s before
        leg_spikes_after_swipe={p: float(np.mean([leg[p][i:i + 30].sum() for _, _, i in swipes])) if swipes else None for p in "LR"},
        eye_mean_hz={"L": float(eyeL.mean()), "R": float(eyeR.mean())},
        meta=meta,
    )
    with open(os.path.join(out_dir, f"episode_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)

    fig, ax = plt.subplots(4, 1, figsize=(14, 9), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1, 0.5]))
    ax[0].plot(t, eyeL, color="#25c9e8", lw=0.8, label="left eye (L1+L2 drive, Hz)")
    ax[0].plot(t, eyeR, color="#ff9c3a", lw=0.8, label="right eye")
    ax[0].set_ylabel("eye drive (Hz)"); ax[0].legend(loc="upper right", fontsize=8)
    ax[1].plot(t, dn_hz, color="#5ad46a", lw=0.8, label="descending neurons (Hz / cell)")
    if meta.get("burst_hz"):
        ax[1].axhline(meta["burst_hz"], color="w", ls=":", lw=0.8, label="swipe threshold")
    ax[1].set_ylabel("DN rate"); ax[1].legend(loc="upper right", fontsize=8)
    ax[2].plot(t, leg["L"] / dt / max(1, int(leg_n["L"])), color="#25c9e8", lw=0.8, label="front-left leg MNs (Hz/cell)")
    ax[2].plot(t, leg["R"] / dt / max(1, int(leg_n["R"])), color="#ff9c3a", lw=0.8, label="front-right leg MNs")
    ax[2].set_ylabel("leg MN rate"); ax[2].legend(loc="upper right", fontsize=8)
    for ts_, side, _ in swipes:
        for a in ax[:3]:
            a.axvline(ts_, color="#25c9e8" if side == "L" else "#ff9c3a", alpha=0.35, lw=1)
        ax[3].plot([ts_], [0 if side == "L" else 1], "o", color="#25c9e8" if side == "L" else "#ff9c3a", ms=6)
    ax[3].set_yticks([0, 1]); ax[3].set_yticklabels(["swipe L", "swipe R"]); ax[3].set_ylim(-0.5, 1.5)
    ax[3].set_xlabel("time (s)")
    title = (f"{tag}: {res['n_swipes']} swipes (L {res['swipes_L']} / R {res['swipes_R']}), "
             f"side = eye that saw the onset in {res['side_matches_eye_onset']}/{res['n_swipes']}, "
             f"mean population rate {res['mean_pop_rate_hz']:.2f} Hz")
    for a in ax:
        a.set_facecolor("#111"); a.grid(alpha=0.15)
    fig.patch.set_facecolor("#181818")
    for a in ax:
        a.tick_params(colors="w"); a.yaxis.label.set_color("w"); a.xaxis.label.set_color("w")
    fig.suptitle(title, color="w")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"episode_{tag}.png"), dpi=110)
    print(json.dumps({k: v for k, v in res.items() if k not in ("swipe_times", "meta")}, indent=1))
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Summarise episode logs: swipe statistics, side check, DN/swipe timing, figure.")
    ap.add_argument("tags", nargs="*", default=["final"], help="episode tags (out/events_<tag>.jsonl)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "out"))
    a = ap.parse_args()
    for tag in a.tags:
        analyse(tag, a.out_dir)

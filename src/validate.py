#!/usr/bin/env python3
"""Validation battery for the MaleCNS LIF simulator (`src/sim.py`).

Tests
  A  Shiu benchmark: labellar "sugar" GRN proxy (LB3a-d) at 150 Hz -> MN9 (proboscis
     motor neuron).  Sugar alone / sugar+bitter (LB1a-e) / bitter alone, at several
     weight_scale values.  Also reports whether the whole network is in runaway.
  B  Runaway / silence: (i) no input -> population rate ~0.  (ii) left-eye L1+L2 at
     60 Hz for 1.5 s at several weight_scale values; picks the largest scale that is
     NOT in runaway as `recommended_weight_scale`.
  C  Shuffle control: same left-eye drive on `conn.shuffled(0)`; lateralisation index
     (L-R)/(L+R) for optic lobe / VPN / DN / front-leg MN, real vs shuffled.

Usage
  python src/validate.py --quick                 # the budgets described in the task
  python src/validate.py --tests A,B             # subset
  python src/validate.py                         # 2x longer runs

Outputs
  out/validate_results.json, out/validate_A_mn9.png, out/validate_B_lateral.png

NOTE on the sugar/bitter sets: MaleCNS v1.0 has no sugar/bitter GRN label.  `grn_sugar`
= labellar bristle types LB3a-d (78 cells) and `grn_bitter` = LB1a-e (56 cells) are
proxies (see docs/RESEARCH_DATA.md section 6).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import Brain, Connectome, LIFConfig  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "out")

# Reference palette (dataviz skill): categorical slots 1-3, ink and chrome.
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- runner
def run_phases(conn: Connectome, cfg: LIFConfig, phases: Sequence[Tuple[int, Optional[np.ndarray], float]],
               record: Dict[str, np.ndarray], seed: int = 0):
    """Run consecutive constant-drive phases on a fresh Brain.

    phases: list of (steps, drive_idx or None, rate_hz).
    Returns (counts: name -> (T,) spikes per step incl. 'all', ever_fired mask, brain).
    """
    brain = Brain(conn, cfg, seed=seed)
    total = sum(p[0] for p in phases)
    counts = {k: np.zeros(total, np.int32) for k in record}
    counts["all"] = np.zeros(total, np.int32)
    masks = {}
    for k, idx in record.items():
        m = np.zeros(conn.n, bool)
        m[idx] = True
        masks[k] = m
    ever = np.zeros(conn.n, bool)
    s = 0
    for steps, drive_idx, rate in phases:
        rate_arr = np.full(drive_idx.size, float(rate), np.float64) if drive_idx is not None else None
        for _ in range(steps):
            fired = brain.step(drive_idx, rate_arr)
            counts["all"][s] = fired.size
            ever[fired] = True
            for k, m in masks.items():
                counts[k][s] = int(m[fired].sum())
            s += 1
    return counts, ever, brain


def rate_hz(counts: np.ndarray, n_cells: int, dt_s: float, sl: slice) -> float:
    seg = counts[sl]
    if seg.size == 0 or n_cells == 0:
        return 0.0
    return float(seg.sum()) / n_cells / (seg.size * dt_s)


def lat_index(left: float, right: float) -> Optional[float]:
    tot = left + right
    return None if tot <= 0 else float((left - right) / tot)


def runaway_stats(all_counts: np.ndarray, n: int, dt_s: float, window_steps: int) -> Dict:
    """Population-rate summary over the final `window_steps`; runaway = high and/or still rising."""
    T = all_counts.size
    w = min(window_steps, T)
    last = slice(T - w, T)
    half = max(w // 2, 1)
    prev_q = slice(T - w, T - half)
    last_q = slice(T - half, T)
    r_last = rate_hz(all_counts, n, dt_s, last)
    r_prev_q = rate_hz(all_counts, n, dt_s, prev_q)
    r_last_q = rate_hz(all_counts, n, dt_s, last_q)
    rising = bool(r_last_q > 1.2 * r_prev_q and r_last_q > 0.5)
    stable = bool(abs(r_last_q - r_prev_q) <= 0.2 * max(r_prev_q, 0.1))
    return {
        "pop_rate_hz_last_window": r_last,
        "pop_rate_hz_prev_quarter": r_prev_q,
        "pop_rate_hz_last_quarter": r_last_q,
        "still_rising": rising,
        "stable": stable,
        "runaway_gt10hz": bool(r_last > 10.0 or (rising and r_last > 5.0)),
        "peak_spikes_per_step": int(all_counts.max()) if T else 0,
    }


def binned_rate(counts: np.ndarray, n_cells: int, dt_s: float, bin_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    T = counts.size - counts.size % bin_steps
    b = counts[:T].reshape(-1, bin_steps).sum(axis=1)
    t = (np.arange(b.size) + 0.5) * bin_steps * dt_s
    return t, b / max(n_cells, 1) / (bin_steps * dt_s)


def cfg_label(ws: float, adapt: float) -> str:
    return f"ws={ws:g}" + (f",adapt={adapt:g}" if adapt else "")


# --------------------------------------------------------------------------- Test A
def test_A(conn: Connectome, factor: float, configs: List[Tuple[float, float]]) -> Dict:
    dt_s = 0.002
    base_steps = int(round(100 * factor))     # 0.2 s
    drive_steps = int(round(500 * factor))    # 1.0 s
    post_steps = int(round(150 * factor))     # 0.3 s of silence after the drive (latched-runaway check)
    sugar = conn.idx("grn_sugar")
    bitter = conn.idx("grn_bitter")
    both = np.union1d(sugar, bitter)
    mn9 = conn.idx("MN9")
    mn9_l = conn.idx([10331])
    mn9_r = conn.idx([16949])
    record = {"MN9": mn9, "MN9_L": mn9_l, "MN9_R": mn9_r, "mn_all": conn.idx("mn_all"),
              "dn_all": conn.idx("dn_all"), "grn_sugar": sugar, "grn_bitter": bitter}
    conds = {"sugar": sugar, "sugar+bitter": both, "bitter": bitter}
    out: Dict[str, Any] = {"protocol": {"baseline_s": base_steps * dt_s, "drive_s": drive_steps * dt_s, "post_s": post_steps * dt_s, "drive_hz": 150.0,
                        "sets": {"grn_sugar": "LB3a-d proxy (78)", "grn_bitter": "LB1a-e proxy (56)",
                                 "MN9": [10331, 16949]}},
           "configs": {}, "traces": {}}
    base = slice(0, base_steps)
    drv = slice(base_steps, base_steps + drive_steps)
    post_end = slice(base_steps + drive_steps + post_steps // 2, base_steps + drive_steps + post_steps)  # last half of post
    for ws, adapt in configs:
        key = cfg_label(ws, adapt)
        cfg = LIFConfig(weight_scale=ws, adapt_mv=adapt)
        res: Dict[str, Any] = {"weight_scale": ws, "adapt_mv": adapt, "conditions": {}}
        traces: Dict[str, Any] = {}
        for cname, idx in conds.items():
            t0 = time.time()
            counts, ever, brain = run_phases(conn, cfg, [(base_steps, None, 0.0), (drive_steps, idx, 150.0), (post_steps, None, 0.0)], record)
            n_drive = idx.size
            ra = runaway_stats(counts["all"][:base_steps + drive_steps], conn.n, dt_s, drive_steps // 2)
            ra["pop_rate_hz_post_drive"] = rate_hz(counts["all"], conn.n, dt_s, post_end)
            ra["latched_after_drive"] = bool(ra["pop_rate_hz_post_drive"] > 1.0)
            r: Dict[str, Any] = {
                "mn9_spikes_drive": int(counts["MN9"][drv].sum()),
                "mn9_spikes_baseline": int(counts["MN9"][base].sum()),
                "mn9_rate_hz_drive": rate_hz(counts["MN9"], mn9.size, dt_s, drv),
                "mn9_rate_hz_baseline": rate_hz(counts["MN9"], mn9.size, dt_s, base),
                "mn9_L_rate_hz_drive": rate_hz(counts["MN9_L"], 1, dt_s, drv),
                "mn9_R_rate_hz_drive": rate_hz(counts["MN9_R"], 1, dt_s, drv),
                "mn_all_rate_hz_drive": rate_hz(counts["mn_all"], record["mn_all"].size, dt_s, drv),
                "dn_all_rate_hz_drive": rate_hz(counts["dn_all"], record["dn_all"].size, dt_s, drv),
                "driven_rate_hz_check": rate_hz(counts["grn_sugar"] if cname != "bitter" else counts["grn_bitter"],
                                                sugar.size if cname != "bitter" else bitter.size, dt_s, drv),
                "pop_rate_hz_drive": rate_hz(counts["all"], conn.n, dt_s, drv),
                "pop_rate_hz_drive_excluding_driven": (counts["all"][drv].sum()
                                                       - (counts["grn_sugar"][drv].sum() if cname != "bitter" else 0)
                                                       - (counts["grn_bitter"][drv].sum() if cname != "sugar" else 0)) / (conn.n - n_drive) / (drive_steps * dt_s),
                "frac_neurons_fired_once": float(ever.sum() / conn.n),
                "n_neurons_fired_once": int(ever.sum()),
                "total_spikes": int(counts["all"].sum()),
                "runaway": ra,
                "wall_s": time.time() - t0,
            }
            res["conditions"][cname] = r
            tb, rb = binned_rate(counts["MN9"], mn9.size, dt_s, 10)
            _, pb = binned_rate(counts["all"], conn.n, dt_s, 10)
            traces[cname] = {"t": tb, "mn9": rb, "pop": pb}
            log(f"A {key:18s} {cname:13s} MN9 {r['mn9_rate_hz_drive']:7.1f} Hz/cell  "
                f"(L {r['mn9_L_rate_hz_drive']:.1f} / R {r['mn9_R_rate_hz_drive']:.1f})  pop {r['pop_rate_hz_drive']:6.2f} Hz  "
                f"fired {100*r['frac_neurons_fired_once']:5.1f}%  runaway={ra['runaway_gt10hz']} post-drive {ra['pop_rate_hz_post_drive']:.2f} Hz  "
                f"{r['wall_s']:.1f}s")
        s, sb, b = (res["conditions"][k]["mn9_rate_hz_drive"] for k in ("sugar", "sugar+bitter", "bitter"))
        res["bitter_suppresses_mn9"] = bool(sb < s)
        res["mn9_ratio_sugar_bitter_over_sugar"] = (sb / s) if s > 0 else None
        res["sugar_gt_bitter"] = bool(s > b)
        res["network_runaway"] = bool(res["conditions"]["sugar"]["runaway"]["runaway_gt10hz"])
        res["network_latched_after_drive"] = bool(res["conditions"]["sugar"]["runaway"]["latched_after_drive"])
        out["configs"][key] = res
        out["traces"][key] = traces
    return out


# --------------------------------------------------------------------------- Test B
LAT_SETS = [("ol", "ol_left", "ol_right"), ("vpn", "vpn_left", "vpn_right"),
            ("dn", "dn_left", "dn_right"), ("mn_frontleg", "mn_frontleg_left", "mn_frontleg_right")]


def eye_drive_run(conn: Connectome, cfg: LIFConfig, steps: int, seed: int = 0, post_steps: int = 0) -> Dict:
    dt_s = 0.002
    eye = np.union1d(conn.idx("eye_left_L1"), conn.idx("eye_left_L2"))
    record = {}
    sizes = {}
    for _, l, r in LAT_SETS:
        for name in (l, r):
            if name in conn.sets:
                idx = conn.idx(name)
                if name.startswith("ol_"):
                    idx = np.setdiff1d(idx, eye)  # optic lobe excluding the driven L1/L2 cells
                record[name] = idx
                sizes[name] = int(idx.size)
    record["mn_all"] = conn.idx("mn_all")
    record["dn_all"] = conn.idx("dn_all")
    record["eye_driven"] = eye
    sizes["mn_all"], sizes["dn_all"], sizes["eye_driven"] = record["mn_all"].size, record["dn_all"].size, eye.size
    t0 = time.time()
    counts, ever, brain = run_phases(conn, cfg, [(steps, eye, 60.0), (post_steps, None, 0.0)], record, seed=seed)
    win = min(int(round(0.5 / dt_s)), steps)   # last 0.5 s of drive
    last = slice(steps - win, steps)
    ra = runaway_stats(counts["all"][:steps], conn.n, dt_s, win)
    post_end = slice(steps + post_steps // 2, steps + post_steps)
    ra["pop_rate_hz_post_drive"] = rate_hz(counts["all"], conn.n, dt_s, post_end) if post_steps else None
    ra["latched_after_drive"] = bool(post_steps and ra["pop_rate_hz_post_drive"] > 1.0)
    rates = {k: rate_hz(counts[k], sizes[k], dt_s, last) for k in record}
    spikes = {k: int(counts[k][last].sum()) for k in record}
    lat = {}
    for tag, l, r in LAT_SETS:
        if l in rates and r in rates:
            lat[tag] = {"left_hz": rates[l], "right_hz": rates[r], "index": lat_index(rates[l], rates[r]),
                        "left_spikes_last_window": spikes[l], "right_spikes_last_window": spikes[r]}
    res: Dict[str, Any] = {
        "steps": steps, "sim_s": steps * dt_s, "post_s": post_steps * dt_s, "window_s": win * dt_s, "drive_hz": 60.0, "n_driven": int(eye.size),
        "pop_rate_hz_last_window": ra["pop_rate_hz_last_window"],
        "pop_rate_hz_excluding_driven_last_window": (counts["all"][last].sum() - counts["eye_driven"][last].sum())
        / (conn.n - eye.size) / (win * dt_s),
        "runaway": ra,
        "rates_hz_last_window": rates,
        "spikes_last_window": spikes,
        "lateralisation": lat,
        "mn_frontleg_fires": bool(spikes.get("mn_frontleg_left", 0) + spikes.get("mn_frontleg_right", 0) > 0),
        "mn_all_fires": bool(spikes["mn_all"] > 0),
        "frac_neurons_fired_once": float(ever.sum() / conn.n),
        "total_spikes": int(counts["all"].sum()),
        "pop_trace_hz_20ms": binned_rate(counts["all"], conn.n, dt_s, 10)[1].tolist(),
        "wall_s": time.time() - t0,
    }
    return res


def test_B(conn: Connectome, factor: float, configs: List[Tuple[float, float]]) -> Dict:
    dt_s = 0.002
    silent_steps = int(round(500 * factor))
    eye_steps = int(round(750 * factor))
    post_steps = int(round(150 * factor))
    out: Dict[str, Any] = {"silence": {}, "eye_drive": {}, "criterion": "pop mean rate over last 0.5 s of drive < 5 Hz, not rising (second 0.25 s half of the window <= 1.2x the first half, or below 0.5 Hz), and not latched (pop rate > 1 Hz 0.15-0.3 s after drive off)"}
    for ws, adapt in configs:
        key = cfg_label(ws, adapt)
        cfg = LIFConfig(weight_scale=ws, adapt_mv=adapt)
        counts, ever, _ = run_phases(conn, cfg, [(silent_steps, None, 0.0)], {})
        out["silence"][key] = {"pop_rate_hz": rate_hz(counts["all"], conn.n, dt_s, slice(0, silent_steps)),
                               "total_spikes": int(counts["all"].sum()), "sim_s": silent_steps * dt_s}
        log(f"B(i) {key:18s} silence {silent_steps*dt_s:.1f}s: {int(counts['all'].sum())} spikes")
        r = eye_drive_run(conn, cfg, eye_steps, post_steps=post_steps)
        r["weight_scale"], r["adapt_mv"] = ws, adapt
        r["not_runaway"] = bool(r["pop_rate_hz_last_window"] < 5.0 and not r["runaway"]["still_rising"] and not r["runaway"]["latched_after_drive"])
        out["eye_drive"][key] = r
        lat = r["lateralisation"]
        log(f"B(ii) {key:18s} pop {r['pop_rate_hz_last_window']:6.2f} Hz (rising={r['runaway']['still_rising']}, post-drive {r['runaway']['pop_rate_hz_post_drive']:.2f} Hz) "
            f"vpn L/R {lat['vpn']['left_hz']:.2f}/{lat['vpn']['right_hz']:.2f}  ol L/R {lat['ol']['left_hz']:.2f}/{lat['ol']['right_hz']:.2f}  "
            f"dn L/R {lat['dn']['left_hz']:.2f}/{lat['dn']['right_hz']:.2f}  mnFL L/R {lat['mn_frontleg']['left_hz']:.2f}/{lat['mn_frontleg']['right_hz']:.2f}  "
            f"ok={r['not_runaway']}  {r['wall_s']:.1f}s")
    ok = [(ws, adapt) for ws, adapt in configs if out["eye_drive"][cfg_label(ws, adapt)]["not_runaway"]]
    if ok:
        best_ws = max(ws for ws, _ in ok)
        # prefer the pure-Shiu (adapt=0) variant at that scale if it passes
        cands = [(ws, a) for ws, a in ok if ws == best_ws]
        cands.sort(key=lambda x: x[1])
        rec = cands[0]
    else:
        rec = min(configs, key=lambda x: (x[0], -x[1]))
        log("WARNING: no configuration passed the runaway criterion; recommending the smallest scale tested")
    out["recommended_weight_scale"] = rec[0]
    out["recommended_adapt_mv"] = rec[1]
    out["recommended_config"] = cfg_label(*rec)
    out["all_passing_configs"] = [cfg_label(*c) for c in ok]
    return out


# --------------------------------------------------------------------------- Test C
def test_C(conn: Connectome, factor: float, ws: float, adapt: float, real_run: Optional[Dict]) -> Dict:
    eye_steps = int(round(750 * factor))
    cfg = LIFConfig(weight_scale=ws, adapt_mv=adapt)
    log("C building shuffled connectome (seed 0) ...")
    sh = conn.shuffled(0)
    post_steps = int(round(150 * factor))
    if real_run is None:
        real_run = eye_drive_run(conn, cfg, eye_steps, post_steps=post_steps)
    shuf_run = eye_drive_run(sh, cfg, eye_steps, post_steps=post_steps)
    comp = {}
    for tag, _, _ in LAT_SETS:
        if tag in real_run["lateralisation"] and tag in shuf_run["lateralisation"]:
            comp[tag] = {"real": real_run["lateralisation"][tag], "shuffled": shuf_run["lateralisation"][tag]}
    out: Dict[str, Any] = {"config": cfg_label(ws, adapt), "shuffle_seed": 0, "real": real_run, "shuffled": shuf_run, "comparison": comp,
           "real_vpn_strongly_left": bool((real_run["lateralisation"]["vpn"]["index"] or 0) > 0.5),
           "real_ol_strongly_left": bool((real_run["lateralisation"]["ol"]["index"] or 0) > 0.5),
           "shuffled_vpn_lateralised": bool(abs(shuf_run["lateralisation"]["vpn"]["index"] or 0) > 0.5),
           "shuffled_ol_lateralised": bool(abs(shuf_run["lateralisation"]["ol"]["index"] or 0) > 0.5)}
    for tag, c in comp.items():
        log(f"C {tag:12s} real L/R {c['real']['left_hz']:.3f}/{c['real']['right_hz']:.3f} LI={c['real']['index']}   "
            f"shuffled L/R {c['shuffled']['left_hz']:.3f}/{c['shuffled']['right_hz']:.3f} LI={c['shuffled']['index']}")
    log(f"C shuffled pop rate {shuf_run['pop_rate_hz_last_window']:.2f} Hz (real {real_run['pop_rate_hz_last_window']:.2f} Hz)")
    return out


# --------------------------------------------------------------------------- figures
def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK2)


def fig_A(resA: Dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(resA["configs"].keys())
    cols = {"sugar": C_BLUE, "sugar+bitter": C_ORANGE, "bitter": C_AQUA}
    fig, axes = plt.subplots(2, len(keys), figsize=(3.6 * len(keys), 5.6), sharex=True, facecolor=SURFACE, squeeze=False)
    base_s = resA["protocol"]["baseline_s"]
    for j, key in enumerate(keys):
        tr = resA["traces"][key]
        for row, (field, ylabel) in enumerate((("mn9", "MN9 rate (Hz per cell)"), ("pop", "population rate (Hz per cell)"))):
            ax = axes[row, j]
            style_ax(ax)
            ax.axvspan(base_s, base_s + resA["protocol"]["drive_s"], color="#f0efec", zorder=0, lw=0)
            for cname in ("sugar", "sugar+bitter", "bitter"):
                ax.plot(tr[cname]["t"], tr[cname][field], color=cols[cname], lw=1.6, label=cname)
            if row == 0:
                c = resA["configs"][key]["conditions"]
                ax.set_title(f"{key}\nMN9 {c['sugar']['mn9_rate_hz_drive']:.0f} / {c['sugar+bitter']['mn9_rate_hz_drive']:.0f} / "
                             f"{c['bitter']['mn9_rate_hz_drive']:.0f} Hz", fontsize=9, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8, color=INK2)
            if row == 1:
                ax.set_xlabel("time (s)", fontsize=8, color=INK2)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, fontsize=8.5, loc="upper right", ncol=3, labelcolor=INK2, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle("Test A - labellar GRN drive at 150 Hz -> MN9 (grey = drive on, then 0.3 s silence); title numbers: MN9 Hz for sugar / sugar+bitter / bitter",
                 fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_B(resC: Dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = [t for t, _, _ in LAT_SETS if t in resC["comparison"]]
    labels = {"ol": "optic lobe\n(excl. driven)", "vpn": "VPN", "dn": "DN", "mn_frontleg": "front-leg MN"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False, facecolor=SURFACE)
    for ax, which, title in zip(axes, ("real", "shuffled"), ("Real wiring", "Shuffled wiring (seed 0)")):
        style_ax(ax)
        x = np.arange(len(tags))
        L = [resC["comparison"][t][which]["left_hz"] for t in tags]
        R = [resC["comparison"][t][which]["right_hz"] for t in tags]
        w = 0.36
        ax.bar(x - w / 2 - 0.01, L, w, color=C_BLUE, label="left", zorder=2)
        ax.bar(x + w / 2 + 0.01, R, w, color=C_ORANGE, label="right", zorder=2)
        for i, t in enumerate(tags):
            li = resC["comparison"][t][which]["index"]
            top = max(L[i], R[i])
            ax.text(x[i], top, "LI n/a" if li is None else f"LI {li:+.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[t] for t in tags], fontsize=8)
        ax.set_title(f"{title} - {resC['config']} - pop {resC[which]['pop_rate_hz_last_window']:.2f} Hz", fontsize=9, color=INK, loc="left")
        ax.set_ylabel("rate, last 0.5 s (Hz per cell)", fontsize=8, color=INK2)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.margins(y=0.25)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Test B/C - left-eye L1+L2 drive at 60 Hz: left vs right response (LI = (L-R)/(L+R))", fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- summary
def summary_tables(results: Dict) -> str:
    lines = []
    if "A" in results:
        lines.append("\nTEST A  sugar(LB3a-d proxy) -> MN9, 150 Hz, 1.0 s drive  [MN9 Hz per cell during drive | baseline]")
        lines.append(f"{'config':18s} {'sugar':>10s} {'sug+bit':>10s} {'bitter':>10s} {'bit/sug':>8s} {'pop Hz':>8s} {'fired%':>7s} {'runaway':>8s} {'post Hz':>8s}")
        for key, r in results["A"]["configs"].items():
            c = r["conditions"]
            ratio = r["mn9_ratio_sugar_bitter_over_sugar"]
            lines.append(f"{key:18s} {c['sugar']['mn9_rate_hz_drive']:10.1f} {c['sugar+bitter']['mn9_rate_hz_drive']:10.1f} "
                         f"{c['bitter']['mn9_rate_hz_drive']:10.1f} {('%.2f' % ratio) if ratio is not None else 'n/a':>8s} "
                         f"{c['sugar']['pop_rate_hz_drive']:8.2f} {100*c['sugar']['frac_neurons_fired_once']:7.1f} "
                         f"{str(c['sugar']['runaway']['runaway_gt10hz']):>8s} {c['sugar']['runaway']['pop_rate_hz_post_drive']:8.2f}")
    if "B" in results:
        B = results["B"]
        lines.append("\nTEST B  (i) silence: " + ", ".join(f"{k}: {v['pop_rate_hz']:.3f} Hz" for k, v in B["silence"].items()))
        lines.append("TEST B  (ii) left-eye L1+L2 @60 Hz, 1.5 s  [Hz per cell, last 0.5 s]")
        lines.append(f"{'config':18s} {'pop':>7s} {'post':>7s} {'vpnL':>7s} {'vpnR':>7s} {'olL':>7s} {'olR':>7s} {'dnL':>7s} {'dnR':>7s} {'mnFL_L':>7s} {'mnFL_R':>7s} {'ok':>4s}")
        for key, r in B["eye_drive"].items():
            l = r["lateralisation"]
            lines.append(f"{key:18s} {r['pop_rate_hz_last_window']:7.2f} {r['runaway']['pop_rate_hz_post_drive']:7.2f} "
                         f"{l['vpn']['left_hz']:7.2f} {l['vpn']['right_hz']:7.2f} {l['ol']['left_hz']:7.2f} {l['ol']['right_hz']:7.2f} "
                         f"{l['dn']['left_hz']:7.2f} {l['dn']['right_hz']:7.2f} {l['mn_frontleg']['left_hz']:7.2f} {l['mn_frontleg']['right_hz']:7.2f} "
                         f"{'yes' if r['not_runaway'] else 'no':>4s}")
        lines.append(f"recommended_weight_scale = {B['recommended_weight_scale']}  (adapt_mv={B['recommended_adapt_mv']}); passing: {B['all_passing_configs']}")
    if "C" in results:
        C = results["C"]
        lines.append(f"\nTEST C  shuffle control @ {C['config']}   lateralisation index (L-R)/(L+R)")
        lines.append(f"{'set':12s} {'real L':>8s} {'real R':>8s} {'real LI':>8s} {'shuf L':>8s} {'shuf R':>8s} {'shuf LI':>8s}")
        for tag, c in C["comparison"].items():
            f = lambda v: "n/a" if v is None else f"{v:+.2f}"
            lines.append(f"{tag:12s} {c['real']['left_hz']:8.3f} {c['real']['right_hz']:8.3f} {f(c['real']['index']):>8s} "
                         f"{c['shuffled']['left_hz']:8.3f} {c['shuffled']['right_hz']:8.3f} {f(c['shuffled']['index']):>8s}")
    return "\n".join(lines)


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", default="A,B,C", help="comma list from A,B,C")
    ap.add_argument("--quick", action="store_true", help="use the short budgets (0.2+1.0 s for A, 1.5 s for B/C); default is 2x longer")
    ap.add_argument("--factor", type=float, default=None, help="explicit duration multiplier (overrides --quick)")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    factor = args.factor if args.factor is not None else (1.0 if args.quick else 2.0)
    tests = [t.strip().upper() for t in args.tests.split(",") if t.strip()]
    os.makedirs(args.out, exist_ok=True)

    log("loading connectome ...")
    conn = Connectome.load()
    log(f"N={conn.n:,} edges={conn.W.nnz:,}")
    results: Dict = {"config": {"tests": tests, "duration_factor": factor, "dt_ms": 2.0, "n_neurons": conn.n,
                                "n_edges": int(conn.W.nnz), "lif_defaults": LIFConfig().to_json(),
                                "set_sizes": {k: int(conn.idx(k).size) for k in
                                              ("grn_sugar", "grn_bitter", "MN9", "mn_all", "dn_all", "eye_left_L1", "eye_left_L2",
                                               "vpn_left", "vpn_right", "ol_left", "ol_right", "dn_left", "dn_right",
                                               "mn_frontleg_left", "mn_frontleg_right")}}}

    A_configs = [(1.0, 0.0), (0.5, 0.0), (0.3, 0.0), (0.15, 0.0), (0.15, 0.6)]
    B_configs = [(1.0, 0.0), (0.5, 0.0), (0.4, 0.0), (0.35, 0.0), (0.3, 0.0), (0.15, 0.0), (0.15, 0.6)]

    if "A" in tests:
        log("=== TEST A: sugar -> MN9 ===")
        resA = test_A(conn, factor, A_configs)
        traces = resA.pop("traces")
        results["A"] = resA
        fig_A({**resA, "traces": traces}, os.path.join(args.out, "validate_A_mn9.png"))
        log("wrote validate_A_mn9.png")

    rec_ws, rec_adapt = 0.15, 0.6
    real_run = None
    if "B" in tests:
        log("=== TEST B: silence + left-eye drive ===")
        resB = test_B(conn, factor, B_configs)
        results["B"] = resB
        rec_ws, rec_adapt = resB["recommended_weight_scale"], resB["recommended_adapt_mv"]
        real_run = resB["eye_drive"][resB["recommended_config"]]
        results["recommended_weight_scale"] = rec_ws
        results["recommended_adapt_mv"] = rec_adapt

    if "C" in tests:
        log(f"=== TEST C: shuffle control @ {cfg_label(rec_ws, rec_adapt)} ===")
        resC = test_C(conn, factor, rec_ws, rec_adapt, real_run)
        # drop the long traces from the copies stored under C (they live under B already)
        for k in ("real", "shuffled"):
            resC[k] = {kk: vv for kk, vv in resC[k].items() if kk != "pop_trace_hz_20ms"} if k == "real" else resC[k]
        results["C"] = resC
        fig_B(resC, os.path.join(args.out, "validate_B_lateral.png"))
        log("wrote validate_B_lateral.png")

    results["wall_time_s"] = time.time() - T0
    with open(os.path.join(args.out, "validate_results.json"), "w") as f:
        json.dump(to_jsonable(results), f, indent=1)
    print(summary_tables(results))
    log(f"done; results -> {os.path.join(args.out, 'validate_results.json')}")


if __name__ == "__main__":
    main()

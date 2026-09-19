#!/usr/bin/env python3
"""Summarise the arena experiment (docs/ARENA_DESIGN.md, "Analysis"): per-run behavioural and
neural metrics from out/arena_<cond>_s<seed>.jsonl, per-condition means with bootstrap CIs, paired
dopamine-vs-real differences, permutation tests, and four dark-style figures.

    python3 src/analyze_arena.py [--out-dir out] [--conditions real,dopamine,shuffled,random]
                                 [--seeds 0-4] [--prefix arena]

Outputs: out/arena_summary.json (nested: per_run, per_condition, comparisons, missing),
out/arena_trajectories.png, out/arena_time_on_phone.png, out/arena_distance.png,
out/arena_w_ratio.png, and a compact table on stdout.  Missing runs are skipped with a warning.

Metric definitions (per run; dt = meta.control_dt, default 16 ms):
  time_on_phone        fraction of control steps with on_phone true (body centre inside phone rect).
  within_reach         fraction of steps where at least one front-leg tip is over a panel
                       (reach.L or reach.R non-null).
  visits               number of entries onto the phone after a 0.5 s debounce: maximal runs of
                       on_phone=true are found, and two consecutive runs separated by an off-phone
                       gap shorter than 0.5 s are merged into one visit.  A run that begins on the
                       phone at t=0 counts as a visit (the contract's start pose makes this rare).
  mean_visit_s         mean duration of merged visits (start of first run to end of last run,
                       bridged gaps included); null when there are no visits.
  swipes / blocked     count of steps with swipe != null / swipe_blocked != null.
  approach_after_novel probability of an approach after a novel cut: an event is a rising edge of
                       novel.L or novel.R (false -> true) at a step i where on_phone is false and
                       i + 2 s lies inside the run; an approach means min(dist_mm[i+1 .. i+2 s])
                       < dist_mm[i] - 20 mm.  Rate = approaches / events; null if no events.
  approach_random      the same statistic at matched random times: 20 x n_events step indices
                       drawn (seeded by the run seed) uniformly from the off-phone steps that have
                       2 s of run remaining.  approach_diff = approach_after_novel - approach_random.
  time_on_phone_h1/h2  time_on_phone over t < T/2 and t >= T/2 (T = last t + dt); learning is
                       h2 - h1.
  *_hz                 mean population rate per cell for mbon / kc / pam (and every other set in
                       pops): mean per-step spike count / set size (meta.sets) / dt.
  final_w_ratio        w_ratio on the last step; reward_events = rising edges of reward;
                       reward_steps = steps with reward true.
Across seeds: mean and 95 % bootstrap CI (2000 resamples of the seed means, seeded) per condition;
dopamine - real paired by seed with the same bootstrap CI; two-sided permutation p-values on
time_on_phone: real vs random (label permutation, exact enumeration when feasible) and dopamine
vs real (paired sign-flip, exact).  With n = 5 seeds the smallest attainable p is 2/252 ~ 0.008
(unpaired) and 2/32 = 0.0625 (paired); the p-values are reported, not interpreted.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDITIONS = ["real", "dopamine", "shuffled", "random"]
COND_COLOR = {"real": "#25c9e8", "dopamine": "#ff9c3a", "shuffled": "#b48cff", "random": "#8a8f98"}
SEED_COLORS = ["#25c9e8", "#ff9c3a", "#5ad46a", "#ff5c8a", "#f3e35a", "#b48cff", "#4dd0b8", "#ff7f50"]
DEFAULT_SETS = {"kc": 4064, "mbon": 97, "pam": 316, "ppl1": 16, "dn_L": 656, "dn_R": 648,
                "leg_L": 68, "leg_R": 67}
DEFAULT_PHONE = [-79.0, -55.5, 79.0, 55.5]
DEFAULT_PANELS = {"L": [-79.0, -55.5, 0.0, 55.5], "R": [0.0, -55.5, 79.0, 55.5]}
DEBOUNCE_S = 0.5
APPROACH_WINDOW_S = 2.0
APPROACH_MM = 20.0
RANDOM_MATCH_FACTOR = 20
N_BOOT = 2000
BOOT_SEED = 20260919

RUN_METRICS = ["time_on_phone", "within_reach", "visits", "mean_visit_s", "swipes", "blocked",
               "approach_after_novel", "approach_random", "approach_diff", "n_novel_events",
               "time_on_phone_h1", "time_on_phone_h2", "learning", "mbon_hz", "kc_hz", "pam_hz",
               "final_w_ratio", "reward_events", "reward_steps", "mean_dist_mm"]


def warn(msg: str) -> None:
    print("warning:", msg, file=sys.stderr)


def parse_seeds(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def py(o: Any) -> Any:
    """Convert numpy scalars/arrays to JSON-serialisable Python objects."""
    if isinstance(o, dict):
        return {str(k): py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [py(v) for v in o]
    if isinstance(o, np.ndarray):
        return [py(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ------------------------------------------------------------------ loading -------------------
def load_run(path: str) -> Tuple[dict, List[dict]]:
    meta: dict = {}
    steps: List[dict] = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if i == 0 and "meta" in rec and "step" not in rec:
                meta = rec["meta"]
                continue
            steps.append(rec)
    return meta, steps


def set_sizes(meta: dict) -> Dict[str, int]:
    for key in ("sets", "set_sizes", "sets_sizes", "n_sets", "pops"):
        v = meta.get(key)
        if isinstance(v, dict) and v:
            return {k: int(n) for k, n in v.items() if isinstance(n, (int, float))}
    return {}


def phone_rects(meta: dict) -> Tuple[List[float], Dict[str, List[float]]]:
    ph = meta.get("phone", {}) if isinstance(meta.get("phone"), dict) else {}
    rect = ph.get("rect", DEFAULT_PHONE)
    panels = ph.get("panels", DEFAULT_PANELS)
    if isinstance(panels, dict) and set(panels) >= {"L", "R"}:
        pl, pr = panels["L"], panels["R"]
    else:
        pl, pr = DEFAULT_PANELS["L"], DEFAULT_PANELS["R"]
    return [float(v) for v in rect], {"L": [float(v) for v in pl], "R": [float(v) for v in pr]}


def room_size(meta: dict) -> Tuple[float, float]:
    room = meta.get("room", {})
    if isinstance(room, dict):
        return float(room.get("w_mm") or room.get("w") or 400.0), float(room.get("h_mm") or room.get("h") or 400.0)
    if isinstance(room, (list, tuple)) and len(room) == 2:
        return float(room[0]), float(room[1])
    return 400.0, 400.0


# ------------------------------------------------------------------ per-run metrics -----------
def visits_debounced(on: np.ndarray, dt: float, debounce_s: float = DEBOUNCE_S) -> List[Tuple[int, int]]:
    """Merged [start, end) index intervals of on-phone runs (gaps < debounce_s bridged)."""
    if on.size == 0 or not on.any():
        return []
    padded = np.concatenate([[False], on, [False]])
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    gap_steps = max(1, int(round(debounce_s / dt)))
    merged: List[Tuple[int, int]] = []
    for s, e in zip(starts, ends):
        if merged and s - merged[-1][1] < gap_steps:
            merged[-1] = (merged[-1][0], int(e))
        else:
            merged.append((int(s), int(e)))
    return merged


def rising_edges(flag: np.ndarray) -> np.ndarray:
    if flag.size == 0:
        return np.zeros(0, int)
    prev = np.concatenate([[False], flag[:-1]])
    return np.flatnonzero(flag & ~prev)


def approach_rate(idx: np.ndarray, dist: np.ndarray, win: int) -> Optional[float]:
    if idx.size == 0:
        return None
    hits = 0
    for i in idx:
        seg = dist[i + 1:i + 1 + win]
        if seg.size and seg.min() < dist[i] - APPROACH_MM:
            hits += 1
    return hits / idx.size


def run_metrics(meta: dict, steps: List[dict], cond: str, seed: int) -> dict:
    dt = float(meta.get("control_dt") or 0.016)
    n = len(steps)
    t = np.array([s.get("t", i * dt) for i, s in enumerate(steps)], float)
    T = float(t[-1] + dt) if n else 0.0
    on = np.array([bool(s.get("on_phone", False)) for s in steps], bool)
    dist = np.array([float(s.get("dist_mm", np.nan)) for s in steps], float)
    reach = np.array([bool((s.get("reach") or {}).get("L")) or bool((s.get("reach") or {}).get("R")) for s in steps], bool)
    swipe = np.array([s.get("swipe") is not None for s in steps], bool)
    blocked = np.array([s.get("swipe_blocked") is not None for s in steps], bool)
    novel = {p: np.array([bool((s.get("novel") or {}).get(p, False)) for s in steps], bool) for p in "LR"}
    reward = np.array([bool(s.get("reward", False)) for s in steps], bool)
    w_ratio = np.array([float(s.get("w_ratio", 1.0)) for s in steps], float)
    pop_names = sorted({k for s in steps for k in (s.get("pops") or {})})
    pops = {k: np.array([float((s.get("pops") or {}).get(k, 0)) for s in steps], float) for k in pop_names}
    sizes = set_sizes(meta)

    visits = visits_debounced(on, dt)
    durations = [(e - s) * dt for s, e in visits]

    win = max(1, int(round(APPROACH_WINDOW_S / dt)))
    edges = np.unique(np.concatenate([rising_edges(novel["L"]), rising_edges(novel["R"])]))
    edges = edges[(~on[edges]) & (edges + win < n)] if edges.size else edges
    eligible = np.flatnonzero(~on[:max(0, n - win)]) if n > win else np.zeros(0, int)
    rng = np.random.default_rng(1000 + seed)
    rand_idx = rng.choice(eligible, size=RANDOM_MATCH_FACTOR * edges.size, replace=True) if edges.size and eligible.size else np.zeros(0, int)
    a_nov = approach_rate(edges, dist, win)
    a_rnd = approach_rate(rand_idx, dist, win)

    half = t < T / 2
    h1 = float(on[half].mean()) if half.any() else None
    h2 = float(on[~half].mean()) if (~half).any() else None

    rates: Dict[str, Optional[float]] = {}
    for k, arr in pops.items():
        nk = sizes.get(k, DEFAULT_SETS.get(k))
        if nk is None or nk <= 0:
            warn(f"{cond} s{seed}: no set size for pop '{k}' in meta; rate per cell left null")
            rates[k] = None
        else:
            rates[k] = float(arr.mean() / nk / dt) if n else None

    res: Dict[str, Any] = dict(
        condition=cond, seed=seed, n_steps=n, duration_s=T, dt=dt,
        time_on_phone=float(on.mean()) if n else None,
        within_reach=float(reach.mean()) if n else None,
        visits=len(visits),
        mean_visit_s=float(np.mean(durations)) if durations else None,
        swipes=int(swipe.sum()), blocked=int(blocked.sum()),
        n_novel_events=int(edges.size),
        approach_after_novel=a_nov, approach_random=a_rnd,
        approach_diff=(a_nov - a_rnd) if (a_nov is not None and a_rnd is not None) else None,
        time_on_phone_h1=h1, time_on_phone_h2=h2,
        learning=(h2 - h1) if (h1 is not None and h2 is not None) else None,
        mbon_hz=rates.get("mbon"), kc_hz=rates.get("kc"), pam_hz=rates.get("pam"),
        pop_rates_hz=rates,
        final_w_ratio=float(w_ratio[-1]) if n else None,
        min_w_ratio=float(w_ratio.min()) if n else None,
        reward_events=int(rising_edges(reward).size), reward_steps=int(reward.sum()),
        mean_dist_mm=float(np.nanmean(dist)) if n and np.isfinite(dist).any() else None,
        posts=steps[-1].get("posts") if n else None,
        start=dict(x=steps[0]["pose"]["x"], y=steps[0]["pose"]["y"]) if n and "pose" in steps[0] else None,
        fixture=bool(meta.get("fixture", False)),
    )
    # traces kept for figures (not written to JSON)
    res["_trace"] = dict(t=t, x=np.array([s["pose"]["x"] for s in steps], float) if n else np.zeros(0),
                         y=np.array([s["pose"]["y"] for s in steps], float) if n else np.zeros(0),
                         dist=dist, w_ratio=w_ratio, reward_idx=rising_edges(reward), on=on)
    return res


# ------------------------------------------------------------------ statistics ----------------
def bootstrap_ci(vals: Sequence[float], rng: np.random.Generator, n_boot: int = N_BOOT) -> Tuple[Optional[float], Optional[float]]:
    x = np.asarray(vals, float)
    if x.size < 2:
        return None, None
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarise(vals: Sequence[Optional[float]], seeds: Sequence[int], rng: np.random.Generator) -> dict:
    pairs = [(s, v) for s, v in zip(seeds, vals) if v is not None and not (isinstance(v, float) and math.isnan(v))]
    x = [v for _, v in pairs]
    lo, hi = bootstrap_ci(x, rng)
    return dict(mean=float(np.mean(x)) if x else None, ci95=[lo, hi], n=len(x),
                sd=float(np.std(x, ddof=1)) if len(x) > 1 else None,
                per_seed={str(s): v for s, v in pairs})


def perm_test_unpaired(a: Sequence[float], b: Sequence[float], rng: np.random.Generator, max_exact: int = 20000, n_perm: int = 10000) -> dict:
    """Two-sided label-permutation test on the difference of means (a - b)."""
    a_, b_ = np.asarray(a, float), np.asarray(b, float)
    if a_.size == 0 or b_.size == 0:
        return dict(p=None, observed=None, n_a=int(a_.size), n_b=int(b_.size), method="none")
    pooled = np.concatenate([a_, b_])
    n_a, n_tot = a_.size, pooled.size
    obs = float(a_.mean() - b_.mean())
    n_comb = math.comb(n_tot, n_a)
    diffs: List[float] = []
    if n_comb <= max_exact:
        total = pooled.sum()
        for comb in itertools.combinations(range(n_tot), n_a):
            sa = pooled[list(comb)].sum()
            diffs.append(sa / n_a - (total - sa) / (n_tot - n_a))
        method, n_used = "exact", n_comb
    else:
        for _ in range(n_perm):
            perm = rng.permutation(pooled)
            diffs.append(perm[:n_a].mean() - perm[n_a:].mean())
        method, n_used = "monte_carlo", n_perm
    d = np.asarray(diffs)
    p = float(np.mean(np.abs(d) >= abs(obs) - 1e-12))
    return dict(p=p, observed=obs, n_a=int(n_a), n_b=int(n_tot - n_a), method=method, n_permutations=int(n_used),
                min_attainable_p=float(2.0 / n_used) if method == "exact" else None)


def perm_test_paired(d: Sequence[float], rng: np.random.Generator, max_exact: int = 20, n_perm: int = 10000) -> dict:
    """Two-sided sign-flip test on paired differences (mean of d)."""
    d_ = np.asarray(d, float)
    n = d_.size
    if n == 0:
        return dict(p=None, observed=None, n=0, method="none")
    obs = float(d_.mean())
    if n <= max_exact:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        means = (signs * d_).mean(axis=1)
        method, n_used = "exact", int(2 ** n)
    else:
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
        means = (signs * d_).mean(axis=1)
        method, n_used = "monte_carlo", n_perm
    p = float(np.mean(np.abs(means) >= abs(obs) - 1e-12))
    return dict(p=p, observed=obs, n=int(n), method=method, n_permutations=n_used,
                min_attainable_p=float(2.0 / n_used) if method == "exact" else None)


# ------------------------------------------------------------------ figures -------------------
def dark(fig: plt.Figure, axes: Sequence[plt.Axes]) -> None:
    fig.patch.set_facecolor("#181818")
    for a in axes:
        a.set_facecolor("#111")
        a.grid(alpha=0.15)
        a.tick_params(colors="w")
        a.yaxis.label.set_color("w"); a.xaxis.label.set_color("w"); a.title.set_color("w")
        for sp in a.spines.values():
            sp.set_color("#555")


def draw_phone(ax: plt.Axes, rect: List[float], panels: Dict[str, List[float]]) -> None:
    for p, r in panels.items():
        ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], facecolor="#263447", edgecolor="#6f8aa8", lw=1.0, zorder=1))
        ax.text((r[0] + r[2]) / 2, (r[1] + r[3]) / 2, p, color="#9fb3cc", ha="center", va="center", fontsize=9, zorder=2)
    ax.add_patch(Rectangle((rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1], facecolor="none", edgecolor="#c0d0e0", lw=1.4, zorder=3))


def fig_trajectories(runs: Dict[str, Dict[int, dict]], meta_any: dict, conds: List[str], path: str) -> None:
    rect, panels = phone_rects(meta_any)
    w, h = room_size(meta_any)
    ncol = 2
    nrow = max(1, math.ceil(len(conds) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 5.5 * nrow), squeeze=False)
    flat = [a for row in axes for a in row]
    for ax, cond in zip(flat, conds):
        ax.add_patch(Rectangle((-w / 2, -h / 2), w, h, facecolor="#0c0c0c", edgecolor="#777", lw=1.5, zorder=0))
        draw_phone(ax, rect, panels)
        seeds = sorted(runs.get(cond, {}))
        for j, seed in enumerate(seeds):
            tr = runs[cond][seed]["_trace"]
            c = SEED_COLORS[j % len(SEED_COLORS)]
            if tr["x"].size:
                ax.plot(tr["x"], tr["y"], color=c, lw=0.7, alpha=0.55, zorder=4, label=f"seed {seed}")
                ax.plot(tr["x"][0], tr["y"][0], "o", color=c, ms=7, mec="w", mew=0.8, zorder=6)
                ax.plot(tr["x"][-1], tr["y"][-1], "x", color=c, ms=7, mew=1.5, zorder=6)
        top = [runs[cond][s]["time_on_phone"] for s in seeds if runs[cond][s]["time_on_phone"] is not None]
        sub = f"{len(seeds)} seeds, time on phone {np.mean(top):.2f}" if top else "no runs"
        ax.set_title(f"{cond}  ({sub})", fontsize=11)
        ax.set_xlim(-w / 2 - 10, w / 2 + 10); ax.set_ylim(-h / 2 - 10, h / 2 + 10)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    for ax in flat[len(conds):]:
        ax.set_visible(False)
    dark(fig, flat[:len(conds)])
    all_seeds = sorted({s for c in conds for s in runs.get(c, {})})
    if all_seeds:
        handles = [plt.Line2D([], [], color=SEED_COLORS[j % len(SEED_COLORS)], lw=2, marker="o", mec="w", mew=0.8, ms=6, label=f"seed {s}")
                   for j, s in enumerate(all_seeds)]
        leg = fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=9, framealpha=0.3,
                         facecolor="#222", edgecolor="#444", bbox_to_anchor=(0.5, 0.955))
        for txt in leg.get_texts():
            txt.set_color("w")
    fig.suptitle("Arena trajectories: start = circle, end = cross, phone panels L/R drawn to scale", color="w")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_time_on_phone(per_cond: dict, runs: Dict[str, Dict[int, dict]], conds: List[str], path: str) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw=dict(width_ratios=[1.3, 1]))
    rng = np.random.default_rng(7)
    # left: overall time on phone
    for i, cond in enumerate(conds):
        st = per_cond.get(cond, {}).get("time_on_phone")
        if not st or st["mean"] is None:
            continue
        c = COND_COLOR.get(cond, "#ccc")
        ax0.bar(i, st["mean"], color=c, alpha=0.45, width=0.65, edgecolor=c, lw=1.2)
        lo, hi = st["ci95"]
        if lo is not None and hi is not None:
            ax0.errorbar(i, st["mean"], yerr=[[st["mean"] - lo], [hi - st["mean"]]], color="w", capsize=6, lw=1.4, zorder=5)
        for seed, v in st["per_seed"].items():
            ax0.plot(i + rng.uniform(-0.18, 0.18), v, "o", color="w", ms=5, mec=c, mew=1.2, zorder=6, alpha=0.9)
    ax0.set_xticks(range(len(conds))); ax0.set_xticklabels(conds)
    ax0.set_ylabel("fraction of time on the phone")
    ax0.set_ylim(0, 1.0)
    ax0.set_title("time on phone: mean, 95 % bootstrap CI, per-seed dots", fontsize=10)
    # right: first vs second half for real and dopamine
    xs = 0
    ticks, labels = [], []
    for cond in [c for c in ("real", "dopamine") if c in per_cond]:
        c = COND_COLOR[cond]
        for k, (key, lab) in enumerate((("time_on_phone_h1", "1st half"), ("time_on_phone_h2", "2nd half"))):
            st = per_cond[cond].get(key)
            if not st or st["mean"] is None:
                continue
            xpos = xs + k
            ax1.bar(xpos, st["mean"], color=c, alpha=0.25 if k == 0 else 0.55, width=0.7, edgecolor=c, lw=1.2)
            lo, hi = st["ci95"]
            if lo is not None and hi is not None:
                ax1.errorbar(xpos, st["mean"], yerr=[[st["mean"] - lo], [hi - st["mean"]]], color="w", capsize=5, lw=1.2, zorder=5)
            ticks.append(xpos); labels.append(f"{cond}\n{lab}")
        for seed, r in sorted(runs.get(cond, {}).items()):
            if r["time_on_phone_h1"] is not None and r["time_on_phone_h2"] is not None:
                ax1.plot([xs, xs + 1], [r["time_on_phone_h1"], r["time_on_phone_h2"]], "-o", color="w", ms=4, mec=c, mew=1.0, lw=0.7, alpha=0.7, zorder=6)
        xs += 3
    ax1.set_xticks(ticks); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylim(0, 1.0)
    ax1.set_title("learning check: first vs second half (lines = seeds)", fontsize=10)
    if not ticks:
        ax1.text(0.5, 0.5, "no real / dopamine runs", color="w", ha="center", va="center", transform=ax1.transAxes)
    dark(fig, [ax0, ax1])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_distance(runs: Dict[str, Dict[int, dict]], conds: List[str], path: str, bin_s: float = 0.5) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    any_line = False
    for cond in conds:
        series = []
        for seed in sorted(runs.get(cond, {})):
            tr = runs[cond][seed]["_trace"]
            if tr["t"].size < 2:
                continue
            dt = float(np.median(np.diff(tr["t"])))
            k = max(1, int(round(bin_s / dt)))
            nb = tr["dist"].size // k
            if nb == 0:
                continue
            series.append(tr["dist"][:nb * k].reshape(nb, k).mean(axis=1))
        if not series:
            continue
        nb = min(len(s) for s in series)
        M = np.vstack([s[:nb] for s in series])
        tt = (np.arange(nb) + 0.5) * bin_s
        mean = np.nanmean(M, axis=0)
        c = COND_COLOR.get(cond, "#ccc")
        if M.shape[0] > 1:
            sd = np.nanstd(M, axis=0, ddof=1)
            ax.fill_between(tt, mean - sd, mean + sd, color=c, alpha=0.15, lw=0)
        ax.plot(tt, mean, color=c, lw=1.6, label=f"{cond} (n={M.shape[0]})")
        any_line = True
    ax.set_xlabel("time (s)"); ax.set_ylabel("distance to phone (mm, as logged in dist_mm)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"mean distance to the phone per condition, band = ±1 SD across seeds, {bin_s:g} s bins", fontsize=10)
    if any_line:
        leg = ax.legend(loc="upper right", fontsize=8, framealpha=0.3, facecolor="#222", edgecolor="#444")
        for txt in leg.get_texts():
            txt.set_color("w")
    else:
        ax.text(0.5, 0.5, "no runs", color="w", ha="center", va="center", transform=ax.transAxes)
    dark(fig, [ax])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_w_ratio(runs: Dict[str, Dict[int, dict]], path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.2))
    dop = runs.get("dopamine", {})
    for j, seed in enumerate(sorted(dop)):
        tr = dop[seed]["_trace"]
        if tr["t"].size == 0:
            continue
        c = SEED_COLORS[j % len(SEED_COLORS)]
        ax.plot(tr["t"], tr["w_ratio"], color=c, lw=1.2, label=f"seed {seed} ({dop[seed]['reward_events']} rewards, final {tr['w_ratio'][-1]:.3f})")
        ri = tr["reward_idx"]
        if ri.size:
            ax.plot(tr["t"][ri], tr["w_ratio"][ri], "v", color=c, ms=4, alpha=0.8)
    ax.axhline(1.0, color="w", ls=":", lw=0.8)
    ax.set_xlabel("time (s)"); ax.set_ylabel("KC→MBON mean w / w0")
    ax.set_title("dopamine runs: KC→MBON weight ratio over time (triangles = reward events)", fontsize=10)
    if dop:
        leg = ax.legend(loc="lower left", fontsize=8, framealpha=0.3, facecolor="#222", edgecolor="#444")
        for txt in leg.get_texts():
            txt.set_color("w")
        lo = min(float(np.nanmin(r["_trace"]["w_ratio"])) for r in dop.values() if r["_trace"]["w_ratio"].size)
        ax.set_ylim(min(lo - 0.05, 0.9), 1.05)
    else:
        ax.text(0.5, 0.5, "no dopamine runs", color="w", ha="center", va="center", transform=ax.transAxes)
    dark(fig, [ax])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ driver --------------------
def fmt(v: Optional[float], nd: int = 2) -> str:
    return "  -  " if v is None else f"{v:.{nd}f}"


def fmt_ci(st: Optional[dict], nd: int = 2, n_total: Optional[int] = None) -> str:
    if not st or st.get("mean") is None:
        return "-"
    lo, hi = st["ci95"]
    tag = f" n={st['n']}" if n_total is not None and st["n"] != n_total else ""
    if lo is None:
        return f"{st['mean']:.{nd}f}{tag}"
    return f"{st['mean']:.{nd}f} [{lo:.{nd}f},{hi:.{nd}f}]{tag}"


def analyse(out_dir: str, conds: List[str], seeds: List[int], prefix: str) -> dict:
    runs: Dict[str, Dict[int, dict]] = {c: {} for c in conds}
    missing: List[str] = []
    meta_any: dict = {}
    for cond in conds:
        for seed in seeds:
            path = os.path.join(out_dir, f"{prefix}_{cond}_s{seed}.jsonl")
            if not os.path.exists(path):
                warn(f"missing run {path}; skipped")
                missing.append(os.path.basename(path))
                continue
            try:
                meta, steps = load_run(path)
            except (json.JSONDecodeError, OSError) as e:
                warn(f"could not read {path} ({e}); skipped")
                missing.append(os.path.basename(path))
                continue
            if not steps:
                warn(f"{path} has no control steps; skipped")
                missing.append(os.path.basename(path))
                continue
            if not meta_any:
                meta_any = meta
            if meta.get("fixture"):
                warn(f"{os.path.basename(path)} is a synthetic fixture, not a simulation")
            runs[cond][seed] = run_metrics(meta, steps, cond, seed)

    rng = np.random.default_rng(BOOT_SEED)
    per_condition: Dict[str, Any] = {}
    for cond in conds:
        rs = runs[cond]
        ss = sorted(rs)
        per_condition[cond] = {"n_runs": len(ss), "seeds": ss}
        for m in RUN_METRICS:
            per_condition[cond][m] = summarise([rs[s][m] for s in ss], ss, rng)

    comparisons: Dict[str, Any] = {}
    if "dopamine" in runs and "real" in runs:
        common = sorted(set(runs["dopamine"]) & set(runs["real"]))
        paired: Dict[str, Any] = {"seeds": common, "n": len(common)}
        for m in RUN_METRICS:
            d = [(s, runs["dopamine"][s][m] - runs["real"][s][m]) for s in common
                 if runs["dopamine"][s][m] is not None and runs["real"][s][m] is not None]
            paired[m] = summarise([v for _, v in d], [s for s, _ in d], rng)
        top_d = [runs["dopamine"][s]["time_on_phone"] - runs["real"][s]["time_on_phone"] for s in common]
        paired["time_on_phone_signflip_test"] = perm_test_paired(top_d, rng)
        learn_d = [runs["dopamine"][s]["learning"] - runs["real"][s]["learning"] for s in common
                   if runs["dopamine"][s]["learning"] is not None and runs["real"][s]["learning"] is not None]
        paired["learning_signflip_test"] = perm_test_paired(learn_d, rng)
        comparisons["dopamine_minus_real_paired"] = paired
    if "real" in runs and "random" in runs:
        a = [r["time_on_phone"] for r in runs["real"].values()]
        b = [r["time_on_phone"] for r in runs["random"].values()]
        comparisons["real_vs_random_time_on_phone"] = perm_test_unpaired(a, b, rng)
    if "dopamine" in runs and "real" in runs:
        a = [r["time_on_phone"] for r in runs["dopamine"].values()]
        b = [r["time_on_phone"] for r in runs["real"].values()]
        comparisons["dopamine_vs_real_time_on_phone_unpaired"] = perm_test_unpaired(a, b, rng)
    if "shuffled" in runs and "real" in runs:
        a = [r["time_on_phone"] for r in runs["real"].values()]
        b = [r["time_on_phone"] for r in runs["shuffled"].values()]
        comparisons["real_vs_shuffled_time_on_phone"] = perm_test_unpaired(a, b, rng)

    summary = dict(
        prefix=prefix, out_dir=out_dir, conditions=conds, seeds_requested=seeds, missing=missing,
        any_fixture=any(r["fixture"] for rs in runs.values() for r in rs.values()),
        definitions=dict(debounce_s=DEBOUNCE_S, approach_window_s=APPROACH_WINDOW_S, approach_mm=APPROACH_MM,
                         random_match_factor=RANDOM_MATCH_FACTOR, n_bootstrap=N_BOOT, bootstrap_seed=BOOT_SEED,
                         doc="see analyze_arena.py module docstring"),
        per_run={c: {str(s): {k: v for k, v in r.items() if not k.startswith("_")} for s, r in sorted(rs.items())} for c, rs in runs.items()},
        per_condition=per_condition, comparisons=comparisons,
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{prefix}_summary.json"), "w") as f:
        json.dump(py(summary), f, indent=1)

    present = [c for c in conds if runs[c]]
    if present:
        fig_trajectories(runs, meta_any, conds, os.path.join(out_dir, f"{prefix}_trajectories.png"))
        fig_time_on_phone(per_condition, runs, conds, os.path.join(out_dir, f"{prefix}_time_on_phone.png"))
        fig_distance(runs, conds, os.path.join(out_dir, f"{prefix}_distance.png"))
        fig_w_ratio(runs, os.path.join(out_dir, f"{prefix}_w_ratio.png"))
    else:
        warn("no runs found; no figures written")

    # compact table
    tables = [
        ("behaviour", [("time_on_phone", "on phone", 2), ("within_reach", "in reach", 2), ("visits", "visits", 1),
                       ("mean_visit_s", "visit s", 1), ("swipes", "swipes", 1), ("blocked", "blocked", 1),
                       ("time_on_phone_h1", "1st half", 2), ("time_on_phone_h2", "2nd half", 2)]),
        ("approach + neural", [("approach_after_novel", "appr novel", 2), ("approach_random", "appr random", 2),
                               ("n_novel_events", "novel cuts", 0), ("mbon_hz", "MBON Hz", 2), ("kc_hz", "KC Hz", 2),
                               ("pam_hz", "PAM Hz", 2), ("final_w_ratio", "final w", 3), ("reward_events", "rewards", 1)]),
    ]
    for name, cols in tables:
        print(f"-- {name}: mean [95 % bootstrap CI across seeds] --")
        print(f"{'condition':<10}{'n':>3} " + " ".join(f"{lab:>20}" for _, lab, _ in cols))
        for cond in conds:
            pc = per_condition[cond]
            print(f"{cond:<10}{pc['n_runs']:>3} " + " ".join(f"{fmt_ci(pc[m], nd, pc['n_runs']):>20}" for m, _, nd in cols))
    if "dopamine_minus_real_paired" in comparisons:
        p = comparisons["dopamine_minus_real_paired"]
        print(f"dopamine - real (paired, n={p['n']}): time on phone {fmt_ci(p['time_on_phone'], 3)}, "
              f"learning (h2-h1) {fmt_ci(p['learning'], 3)}, visits {fmt_ci(p['visits'], 1)}, "
              f"final w {fmt_ci(p['final_w_ratio'], 3)}; sign-flip p (time on phone) = {fmt(p['time_on_phone_signflip_test']['p'], 3)}, "
              f"p (learning) = {fmt(p['learning_signflip_test']['p'], 3)}")
    for key in ("real_vs_random_time_on_phone", "real_vs_shuffled_time_on_phone", "dopamine_vs_real_time_on_phone_unpaired"):
        if key in comparisons:
            c = comparisons[key]
            print(f"{key}: diff of means {fmt(c['observed'], 3)}, two-sided permutation p = {fmt(c['p'], 3)} "
                  f"({c['method']}, {c.get('n_permutations', 0)} permutations, min attainable {fmt(c.get('min_attainable_p'), 3)})")
    if missing:
        print(f"missing runs ({len(missing)}): " + ", ".join(missing))
    if summary["any_fixture"]:
        print("NOTE: at least one input is a synthetic fixture (meta.fixture = true), not a simulation.")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarise arena runs: per-run metrics, per-condition bootstrap CIs, paired dopamine-vs-real comparison, figures.")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "out"))
    ap.add_argument("--conditions", default=",".join(CONDITIONS), help="comma-separated list")
    ap.add_argument("--seeds", default="0-4", help="e.g. 0-4 or 0,2,3")
    ap.add_argument("--prefix", default="arena", help="log prefix: <prefix>_<cond>_s<seed>.jsonl; also used for outputs")
    a = ap.parse_args()
    conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
    analyse(a.out_dir, conds, parse_seeds(a.seeds), a.prefix)


if __name__ == "__main__":
    main()

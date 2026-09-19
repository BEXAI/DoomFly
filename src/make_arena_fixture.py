#!/usr/bin/env python3
"""Write fake but schema-exact arena logs so `analyze_arena.py` can be developed before the
simulation exists (docs/ARENA_DESIGN.md, "Log format").

    python3 src/make_arena_fixture.py [--out-dir out] [--conditions real,dopamine,shuffled,random]
                                      [--seeds 0-4] [--duration 120] [--prefix arena_fixture]

Each run is a correlated random walk in the 400 x 400 mm room with the phone rectangle of the
contract.  "real" and "dopamine" bias the walk toward the phone and dwell on it; "dopamine"
lengthens dwells in the second half and decays w_ratio from 1.0 toward 0.6 on reward events;
"shuffled" and "random" are unbiased walkers.  Swipes happen only when the relevant reach is
non-null (bursts with a null reach are logged as swipe_blocked), novel flags flip randomly,
posts count up plausibly, pops are Poisson spike counts per control step.

THESE ARE FAKES.  The prefix must contain "fixture" (the default writes out/arena_fixture_<cond>_s<seed>.jsonl)
and an existing file is never overwritten, so the real ``out/arena_<cond>_s<seed>.jsonl`` logs cannot be clobbered.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROOM = 400.0                       # mm, centred on the origin
PANEL_L = (-79.0, -55.5, 0.0, 55.5)  # x0, y0, x1, y1
PANEL_R = (0.0, -55.5, 79.0, 55.5)
PHONE = (-79.0, -55.5, 79.0, 55.5)
CONTROL_DT = 0.016
REACH_MM = 22.0
REACH_DEG = 35.0
SETS = {"eye_L": 1770, "eye_R": 1785, "dn_L": 656, "dn_R": 648, "dna_L": 26, "dna_R": 26,
        "leg_L": 68, "leg_R": 67, "kc": 4064, "mbon": 97, "pam": 316, "med_L": 4429, "med_R": 4445}
POPS = [k for k in SETS if not k.startswith("med_")]   # sets logged per step (run_arena.py)
# baseline rates (Hz per cell) for the Poisson pops
BASE_HZ = {"eye_L": 2.0, "eye_R": 2.0, "dn_L": 0.3, "dn_R": 0.3, "dna_L": 0.4, "dna_R": 0.4,
           "leg_L": 0.8, "leg_R": 0.8, "kc": 0.4, "mbon": 4.0, "pam": 0.5}


def in_rect(x: float, y: float, r: tuple) -> bool:
    return r[0] <= x <= r[2] and r[1] <= y <= r[3]


def panel_under(x: float, y: float) -> Optional[str]:
    if in_rect(x, y, PANEL_L):
        return "L"
    if in_rect(x, y, PANEL_R):
        return "R"
    return None


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


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


def meta_for(cond: str, seed: int, duration: float) -> dict:
    biased = cond in ("real", "dopamine")
    return dict(
        condition=cond, seed=seed, duration=duration, control_dt=CONTROL_DT, substeps=8,
        n_steps=int(round(duration / CONTROL_DT)),
        fixture=True,  # marks a fake log
        room=dict(size_mm=ROOM, origin="centre", x="right", y="up"),
        phone=dict(centre=[0.0, 0.0], rect=list(PHONE), panels=dict(L=list(PANEL_L), R=list(PANEL_R)), panel_mm=[79.0, 111.0]),
        body=dict(v0=12.0, g_v=0.0 if cond == "random" else 6.0, g_omega=0.0 if cond == "random" else 3.0,
                  scale=10.0, body_len_mm=30.0, reach_mm=REACH_MM, reach_deg=REACH_DEG,
                  omega_sigma=1.2, omega_tau_s=0.5, v_stand_mm_s=3.0, v_max_mm_s=60.0),
        eye=dict(grid=[24, 18], cell_mm=2.0, d_min_mm=5.0, d_max_mm=300.0, az_deg_per_col=10.0),
        lif=dict(dt_ms=2.0, weight_scale=0.5, tau_m_ms=20.0, v_th_mv=-50.0, v_reset_mv=-70.0),
        plasticity=dict(enabled=cond == "dopamine", eta=0.05, tau_trace_s=1.0, w_floor=0.2),
        reward=dict(rule="rising edge of novel_visible on either panel while on_phone", pam_hz=100.0, window_s=0.3,
                    drives_pam=cond == "dopamine"),
        decoder=dict(mode="burst", burst_hz=1.5),
        shuffle=cond == "shuffled", brain_steers=cond != "random", biased_fixture_walk=biased,
        n_neurons=166700, n_edges=12_000_000,
        sets=dict(SETS),
        notes="FIXTURE: synthetic walk, Poisson pops. dist_mm = body centre to phone centre; reward = one-step event flag.",
    )


def write_run(cond: str, seed: int, duration: float, out_dir: str, prefix: str) -> str:
    cond_ids = {"real": 1, "dopamine": 2, "shuffled": 3, "random": 4}
    if cond not in cond_ids:
        raise SystemExit(f"unknown condition {cond!r}; choose from {sorted(cond_ids)}")
    rng = np.random.default_rng(seed * 7919 + cond_ids[cond])
    n_steps = int(round(duration / CONTROL_DT))
    dt = CONTROL_DT
    biased = cond in ("real", "dopamine")
    steers = cond != "random"

    # start pose: >= 120 mm from the phone centre
    while True:
        x, y = (float(u) for u in rng.uniform(-185, 185, size=2))
        if math.hypot(x, y) >= 120.0:
            break
    th = float(rng.uniform(-math.pi, math.pi))
    omega_noise = 0.0
    v = 12.0
    dwell_left = 0.0          # seconds of dwell remaining (on-phone stop)
    leave_until = -1.0        # after leaving, suppress the bias until this time
    push_until = -1.0         # ... and actively walk away from the phone until this time
    w_ratio = 1.0
    reward_left = 0
    n_reward = 0
    novel = {"L": False, "R": False}
    novel_left = {"L": 0.0, "R": 0.0}
    posts = {"L": 0, "R": 0}
    next_post = {"L": float(rng.uniform(2, 6)), "R": float(rng.uniform(2, 6))}
    was_on_phone = False

    path = os.path.join(out_dir, f"{prefix}_{cond}_s{seed}.jsonl")
    if "fixture" not in os.path.basename(path):
        raise ValueError(f"refusing to write a fixture to a real-looking log name: {path} (prefix must contain 'fixture')")
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite existing log {path}; delete it first")
    with open(path, "w") as f:
        f.write(json.dumps({"meta": meta_for(cond, seed, duration)}) + "\n")
        for k in range(n_steps):
            t = k * dt
            on_phone = in_rect(x, y, PHONE)
            second_half = t >= duration / 2

            # --- walk -------------------------------------------------------------------
            omega_noise += (-omega_noise / 0.5) * dt + 1.2 * math.sqrt(2 * dt / 0.5) * rng.normal()
            omega = omega_noise
            v0 = 12.0
            if biased and not on_phone and t > leave_until:
                want = math.atan2(-y, -x) + 0.25 * math.sin(0.7 * t)   # meandering approach
                omega += 0.6 * wrap(want - th)
            elif biased and t < push_until:
                omega += 0.8 * wrap(math.atan2(y, x) - th)             # walk away after a visit
            if on_phone:
                if not was_on_phone:
                    mean_dwell = 4.0
                    if cond == "dopamine" and second_half:
                        mean_dwell = 9.0
                    if not biased:
                        mean_dwell = 1.5
                    dwell_left = float(rng.exponential(mean_dwell))
                if dwell_left > 0:
                    v0 = 3.0 if biased else 8.0
                    omega *= 0.6
                    dwell_left -= dt
                    if dwell_left <= 0:
                        th = math.atan2(y, x) + float(rng.uniform(-0.6, 0.6))   # head away
                        leave_until = t + float(rng.uniform(15.0, 30.0))
                        push_until = t + 6.0
            v = float(np.clip(v0 + rng.normal(0, 2.0), 0, 60))
            th = wrap(th + omega * dt)
            x += v * math.cos(th) * dt
            y += v * math.sin(th) * dt
            lim = ROOM / 2 - 6.0
            if abs(x) > lim:
                x = math.copysign(lim, x)
                th = wrap(math.pi - th + math.radians(rng.uniform(-30, 30)))
            if abs(y) > lim:
                y = math.copysign(lim, y)
                th = wrap(-th + math.radians(rng.uniform(-30, 30)))
            on_phone = in_rect(x, y, PHONE)

            # --- reach / swipes --------------------------------------------------------------
            reach: Dict[str, Optional[str]] = {}
            for side, sgn in (("L", 1.0), ("R", -1.0)):
                a = th + sgn * math.radians(REACH_DEG)
                reach[side] = panel_under(x + REACH_MM * math.cos(a), y + REACH_MM * math.sin(a))
            burst = rng.random() < (0.012 if any(reach.values()) else 0.002)
            burst_hz = float(rng.uniform(1.6, 4.0)) if burst else float(abs(rng.normal(0.3, 0.2)))
            side_ev = float(rng.normal(0, 0.6))
            swipe: Optional[str] = None
            blocked: Optional[str] = None
            if burst:
                side = "L" if side_ev < 0 else "R"
                target = reach[side]
                if target is not None and steers:
                    swipe = target
                    posts[target] += 1
                else:
                    blocked = side

            # --- feeds: novel flips, autoplay posts ------------------------------------------------
            rising = False
            for p in "LR":
                if novel[p]:
                    novel_left[p] -= dt
                    if novel_left[p] <= 0:
                        novel[p] = False
                elif rng.random() < 0.12 * dt:
                    novel[p] = True            # rising edge of novel_visible on panel p
                    novel_left[p] = float(rng.uniform(0.6, 2.0))
                    rising = True
                next_post[p] -= dt
                if next_post[p] <= 0:
                    posts[p] += 1
                    next_post[p] = float(rng.uniform(3, 9))

            # --- reward ------------------------------------------------------------------------
            reward = bool(on_phone and rising)          # one-step event flag (run_arena.py)
            if reward:
                reward_left = int(round(0.3 / dt))       # PAM drive window (dopamine only)
                n_reward += 1
                if cond == "dopamine":
                    w_ratio = max(0.6, w_ratio * 0.98)
            in_pam_window = reward_left > 0
            if in_pam_window:
                reward_left -= 1

            # --- pops --------------------------------------------------------------------------
            pops: Dict[str, int] = {}
            for name in POPS:
                n = SETS[name]
                hz = BASE_HZ[name]
                if name.startswith("dn") or name.startswith("dna"):
                    hz = hz + (burst_hz if burst else 0.0) * 0.5
                if name == "pam" and in_pam_window and cond == "dopamine":
                    hz = 100.0
                if name == "mbon" and cond == "dopamine":
                    hz = hz * w_ratio
                if name == "kc" and on_phone:
                    hz = hz * 1.5
                pops[name] = int(rng.poisson(hz * n * dt))
            spikes = int(sum(pops.values()) + rng.poisson(0.35 * 166700 * dt))

            rec = {
                "step": k, "t": round(t, 4),
                "pose": {"x": round(x, 2), "y": round(y, 2), "th": round(th, 4), "v": round(v, 2)},
                "on_phone": on_phone, "dist_mm": round(math.hypot(x, y), 1),
                "reach": reach, "swipe": swipe, "swipe_blocked": blocked,
                "burst_hz": round(burst_hz, 3), "side_ev": round(side_ev, 3), "spikes": spikes,
                "pops": pops, "reward": reward, "w_ratio": round(w_ratio, 5),
                "novel": dict(novel), "posts": dict(posts),
            }
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            was_on_phone = on_phone
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "out"))
    ap.add_argument("--conditions", default="real,dopamine,shuffled,random")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--prefix", default="arena_fixture", help="file prefix; must contain 'fixture'")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for cond in [c.strip() for c in a.conditions.split(",") if c.strip()]:
        for seed in parse_seeds(a.seeds):
            p = write_run(cond, seed, a.duration, a.out_dir, a.prefix)
            print("wrote", p)


if __name__ == "__main__":
    main()

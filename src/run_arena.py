#!/usr/bin/env python3
"""Arena closed loop: free-roaming fly in a dark room with an open phone (docs/ARENA_DESIGN.md).

Per 16 ms control step (8 LIF substeps of 2 ms):
  1. feeds -> panel luminance grids (55 x 39) pasted into the room map; the panoramic eye
     samples (24, 18) = (distance, azimuth) grids for the left / right eye from the fly's pose
  2. EyeEncoder (L1 ON / L2 OFF only, l3_gain 0, flip=True so the nearest distance row lands
     on the ventral retina) turns each grid into Poisson rates
  3. the brain runs 8 substeps with those lamina cells forced; in condition `dopamine` the PAM
     set is additionally forced at 100 Hz during a 300 ms reward window
  4. population counts -> Decoder('burst'): DN population rate > 1.5 Hz/cell picks a side by
     medulla evidence -> the swipe is APPLIED only if that side's front-leg tip is over a panel
     (the panel under the tip is swiped), otherwise logged as swipe_blocked
  5. Body.step with the 80 ms traces (Hz per cell) of DN, DNa-left and DNa-right rates
  6. feeds.step; reward = rising edge of novel_visible on either panel while on the phone
  7. one JSON line per step (schema in docs/ARENA_DESIGN.md "Log format"); pose, on_phone,
     dist_mm and reach are all the post-step values (the swipe gate uses the pre-step reach)

Conditions: real | dopamine | shuffled | random  (see the contract). `random` keeps the brain
running and logged but disconnects it from the body (Body(connected=False): no speed or
steering modulation and no standing rule; g_v = g_omega = 0 as well) and from the feeds:
decoder bursts are logged as `swipe_blocked` and never applied.

Random streams: `SeedSequence(seed).spawn(2)` -> [0] brain (forced Poisson spikes), [1] body
(start pose, OU heading noise, wall kicks); feeds use RandomState(2*seed+1 / 2*seed+2).

    python3 src/run_arena.py --condition real --duration 120 --seed 0 [--spikes] [--out-dir out]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import Connectome, Brain, LIFConfig, SpikeRecorder  # noqa: E402
from feeds import make_pair  # noqa: E402
from encoder import make_encoders  # noqa: E402
from decoder import Decoder, DecoderConfig  # noqa: E402
from arena import Phone, Body, PanoramicEye, random_start_pose, PANEL_GRID, ROOM_MM, FLY_SCALE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOT_PATH = os.path.join(ROOT, "data", "raw", "body-annotations-male-cns-v1.0-minconf-0.5.feather")
SETS_ARENA_PATH = os.path.join(ROOT, "data", "graph", "sets_arena.json")
CONDITIONS = ("real", "dopamine", "shuffled", "random")
CONTROL_DT = 0.016
SUBSTEPS = 8
EYE_GRID = (24, 18)


# ---------------------------------------------------------------------- neuron sets
def load_dna_sets(annot_path: str = ANNOT_PATH, cache_path: str = SETS_ARENA_PATH) -> Dict[str, List[int]]:
    """{'dna_left': [...bodyIds], 'dna_right': [...]}: descending neurons whose `type` starts
    with 'DNa', per somaSide, from the annotations feather; cached to data/graph/sets_arena.json."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            sets = json.load(f)
        if "dna_left" in sets and "dna_right" in sets:
            return sets
    import pandas as pd
    t = pd.read_feather(annot_path, columns=["bodyId", "type", "somaSide", "superclass"])
    d = t[t["superclass"] == "descending_neuron"]
    d = d[d["type"].fillna("").astype(str).str.startswith("DNa")]
    sets = {
        "dna_left": sorted(int(b) for b in d[d["somaSide"] == "L"]["bodyId"]),
        "dna_right": sorted(int(b) for b in d[d["somaSide"] == "R"]["bodyId"]),
        "dna_types": sorted(str(x) for x in d["type"].unique()),
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(sets, f)
    return sets


def rate_trace_alpha(tau_s: float, dt_s: float = CONTROL_DT) -> float:
    return float(np.exp(-dt_s / tau_s))


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--condition", choices=CONDITIONS, default="real")
    ap.add_argument("--duration", type=float, default=120.0, help="seconds of simulated time")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spikes", action="store_true", help="also save whole-brain spike frames (for the render)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "out"))
    # brain (same as run_episode)
    ap.add_argument("--weight-scale", type=float, default=0.5)
    ap.add_argument("--adapt-mv", type=float, default=0.6)
    ap.add_argument("--std-u", type=float, default=0.15)
    ap.add_argument("--std-tau", type=float, default=480.0)
    ap.add_argument("--shuffle-seed", type=int, default=0, help="seed of Connectome.shuffled for `shuffled`")
    # encoder / decoder (same as run_episode)
    ap.add_argument("--rate-max", type=float, default=150.0)
    ap.add_argument("--enc-gain", type=float, default=6.0)
    ap.add_argument("--burst-hz", type=float, default=1.5)
    ap.add_argument("--tau-side", type=float, default=0.80)
    ap.add_argument("--refractory", type=float, default=0.4)
    ap.add_argument("--global-refractory", type=float, default=0.3)
    # body
    ap.add_argument("--v0", type=float, default=12.0)
    ap.add_argument("--g-v", type=float, default=6.0)
    ap.add_argument("--g-omega", type=float, default=3.0)
    ap.add_argument("--start-min-dist", type=float, default=120.0)
    # reward / plasticity
    ap.add_argument("--reward-hz", type=float, default=100.0)
    ap.add_argument("--reward-s", type=float, default=0.3)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--tau-trace", type=float, default=1.0)
    # feeds
    ap.add_argument("--autoplay", type=int, default=1)
    ap.add_argument("--autoplay-amp", type=float, default=0.8)
    ap.add_argument("--autoplay-hz", type=float, default=1.0)
    ap.add_argument("--autoplay-whole", type=int, default=1)
    args = ap.parse_args()

    cond = args.condition
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"arena_{cond}_s{args.seed}"
    log_path = os.path.join(args.out_dir, f"{tag}.jsonl")
    seed_brain, seed_body = np.random.SeedSequence(args.seed).spawn(2)
    rng = np.random.default_rng(seed_body)      # body: start pose, OU noise, wall kicks
    n_steps = int(round(args.duration / CONTROL_DT))

    # ---------------- brain
    print("loading connectome ...", flush=True)
    conn = Connectome.load()
    if cond == "shuffled":
        print(f"using wiring-shuffled control graph (seed {args.shuffle_seed})", flush=True)
        conn = conn.shuffled(args.shuffle_seed)
    cfg = LIFConfig(weight_scale=args.weight_scale, adapt_mv=args.adapt_mv, std_u=args.std_u, std_tau_rec_ms=args.std_tau)
    brain = Brain(conn, cfg, seed=seed_brain)

    dna_sets = load_dna_sets()
    dna_L, dna_R = conn.idx(dna_sets["dna_left"]), conn.idx(dna_sets["dna_right"])
    dna_source = "annotations DNa*"
    if dna_L.size == 0 or dna_R.size == 0:
        dna_L, dna_R = conn.idx("dn_left"), conn.idx("dn_right")
        dna_source = "fallback dn_left/dn_right"
    med = lambda side: np.unique(np.concatenate([conn.idx(f"eye_{side}_{t}") for t in ("Mi1", "Tm1", "Tm2", "Tm9", "L5")]))  # noqa: E731
    pops: Dict[str, np.ndarray] = {
        "eye_L": np.concatenate([conn.idx("eye_left_L1"), conn.idx("eye_left_L2")]),
        "eye_R": np.concatenate([conn.idx("eye_right_L1"), conn.idx("eye_right_L2")]),
        "dn_L": conn.idx("dn_left"), "dn_R": conn.idx("dn_right"),
        "dna_L": dna_L, "dna_R": dna_R,
        "leg_L": conn.idx("mn_frontleg_left"), "leg_R": conn.idx("mn_frontleg_right"),
        "kc": conn.idx("kc"), "mbon": conn.idx("mbon"), "pam": conn.idx("pam"),
        "med_L": med("left"), "med_R": med("right"),
    }
    LOG_POPS = ("eye_L", "eye_R", "dn_L", "dn_R", "dna_L", "dna_R", "leg_L", "leg_R", "kc", "mbon", "pam")
    # Hz per cell of the DN population uses the cells that are actually counted (dn_L + dn_R);
    # dn_all also holds a few unsided DNs that no population count includes.
    n_dn = int(np.union1d(pops["dn_L"], pops["dn_R"]).size)
    n_dn_all = int(conn.idx("dn_all").size)
    print(f"sets: DN {n_dn} sided of {n_dn_all} | DNa L {dna_L.size} R {dna_R.size} ({dna_source}) | KC {pops['kc'].size} MBON {pops['mbon'].size} PAM {pops['pam'].size}", flush=True)

    # ---------------- plasticity (dopamine condition only)
    plasticity_on = False
    plast_meta: Dict[str, object] = dict(enabled=False)
    if cond == "dopamine":
        if not hasattr(brain, "enable_plasticity"):
            raise SystemExit("condition `dopamine` needs Brain.enable_plasticity / dopamine_drive / plasticity_stats "
                             "in src/sim.py (plasticity API not available in this checkout)")
        brain.enable_plasticity(pops["kc"], pops["mbon"], pops["pam"], eta=args.eta, tau_trace_s=args.tau_trace)
        plasticity_on = True
        st = brain.plasticity_stats()
        plast_meta = dict(enabled=True, eta=args.eta, tau_trace_s=args.tau_trace, n_edges=int(st.get("n_edges", 0)),
                          rule="dopamine-gated depression, floor 0.2*w0, no recovery (v1)")
        print(f"plasticity on: {st}", flush=True)
    use_dopa_api = hasattr(brain, "dopamine_drive")

    # ---------------- world
    phone = Phone()
    x0, y0, th0 = random_start_pose(rng, phone, min_dist=args.start_min_dist)
    connected = cond != "random"
    g_v, g_omega = (args.g_v, args.g_omega) if connected else (0.0, 0.0)
    body = Body(phone, rng, x0, y0, th0, v0=args.v0, g_v=g_v, g_omega=g_omega, connected=connected)
    eye = PanoramicEye(phone)
    seed_l, seed_r = 2 * args.seed + 1, 2 * args.seed + 2
    feedL, feedR = make_pair(seed_l, seed_r, scale=0.25, autoplay=bool(args.autoplay), autoplay_amp=args.autoplay_amp,
                             autoplay_hz=args.autoplay_hz, autoplay_whole=bool(args.autoplay_whole))
    feeds = {"L": feedL, "R": feedR}
    # flip=True: eye-grid row 0 (nearest) -> screen bottom -> ventral retina (EyeEncoder's default
    # puts grid row 0 on the dorsal, hex2-up side); see docs/ARENA_DESIGN.md "Eye".
    encL, encR = make_encoders(grid=EYE_GRID, rate_max_hz=args.rate_max, gain=args.enc_gain, l3_gain=0.0, flip=True)
    n_drv = {"L": int(encL.idx_L1.size + encL.idx_L2.size), "R": int(encR.idx_L1.size + encR.idx_L2.size)}
    print(f"eye L drives {n_drv['L']} lamina cells, eye R {n_drv['R']} (L1+L2); start ({x0:.0f}, {y0:.0f}) th {math.degrees(th0):.0f} deg", flush=True)

    # ---------------- decoder (burst, as run_episode)
    dcfg = DecoderConfig(control_dt_s=CONTROL_DT, refractory_s=args.refractory, global_refractory_s=args.global_refractory,
                         burst_hz=args.burst_hz, tau_side_s=args.tau_side, n_dn=n_dn,
                         n_ol={"L": int(pops["med_L"].size), "R": int(pops["med_R"].size)})
    dec = Decoder("burst", dcfg, 1)

    rec = SpikeRecorder(conn.n, pops)
    a_tr = rate_trace_alpha(0.080)
    trace = {"dn": 0.0, "dna_L": 0.0, "dna_R": 0.0}   # 80 ms traces, Hz per cell
    n_cells = {"dn": n_dn, "dna_L": int(dna_L.size), "dna_R": int(dna_R.size)}

    meta = dict(
        condition=cond, seed=args.seed, seed_l=seed_l, seed_r=seed_r, duration=args.duration,
        rng="SeedSequence(seed).spawn(2): [0] brain, [1] body; feeds RandomState(seed_l / seed_r)",
        control_dt=CONTROL_DT, substeps=SUBSTEPS, n_steps=n_steps,
        room=dict(size_mm=ROOM_MM, origin="centre", x="right", y="up", heading="rad, CCW positive, 0 = +x"),
        phone=phone.to_json(), body=body.params(), start=dict(x=x0, y=y0, th=th0, min_dist_mm=args.start_min_dist),
        scale=FLY_SCALE, eye=eye.params(),
        encoder=dict(grid=list(EYE_GRID), rate_max_hz=args.rate_max, gain=args.enc_gain, l3_gain=0.0, driven_cells=n_drv, flip=True,
                     axes="rows = distance (row 0 nearest -> ventral retina, flip=True), cols = azimuth (col 0 frontal)"),
        decoder=dict(mode="burst", burst_hz=args.burst_hz, tau_side_s=args.tau_side, refractory_s=args.refractory,
                     global_refractory_s=args.global_refractory, warmup_s=dcfg.warmup_s, n_dn=n_dn, n_dn_all=n_dn_all,
                     n_ol={"L": int(pops["med_L"].size), "R": int(pops["med_R"].size)}),
        lif=cfg.to_json(), shuffle_seed=(args.shuffle_seed if cond == "shuffled" else None),
        plasticity=plast_meta,
        reward=dict(rule="rising edge of novel_visible on either panel while on_phone", pam_hz=args.reward_hz,
                    window_s=args.reward_s, drives_pam=(cond == "dopamine")),
        rate_trace_s=0.080, dna_source=dna_source, dna_types=dna_sets.get("dna_types"),
        n_neurons=conn.n, n_edges=int(conn.W.nnz), sets={k: int(v.size) for k, v in pops.items()},
        feeds=dict(autoplay=bool(args.autoplay), autoplay_amp=args.autoplay_amp, autoplay_hz=args.autoplay_hz,
                   autoplay_whole=bool(args.autoplay_whole), panel_grid=list(PANEL_GRID)),
        notes=("random: decoder bursts logged as swipe_blocked, never applied; Body(connected=False), g_v = g_omega = 0. "
               "dist_mm = body centre to phone centre. swipe = panel actually swiped (panel under the tip at the "
               "pre-step pose); pose, on_phone, dist_mm and reach are logged post-step. spikes = all spikes of the "
               "8 substeps including the forced lamina (and, in a reward window, PAM) spikes; n_dn = sided DNs "
               "(dn_L + dn_R) used for Hz/cell of burst_hz and the body traces."),
    )

    # ---------------- loop
    log = open(log_path, "w")
    log.write(json.dumps(dict(meta=meta)) + "\n")
    n_swipes = {"L": 0, "R": 0}
    n_blocked = 0
    n_rewards = 0
    reward_until = -1.0
    prev_novel = {"L": bool(feedL.state["novel_visible"]), "R": bool(feedR.state["novel_visible"])}

    def posts(feed) -> int:
        return int(feed.state["posts_consumed"])  # type: ignore[call-overload]

    on_steps = 0
    w_ratio = 1.0
    last_dopa_steps = -1      # plasticity_stats is recomputed only when the weights changed
    t_wall0 = time.time()
    t_eye = t_brain = 0.0
    next_print = 0.0
    pam_idx = pops["pam"]
    for k in range(n_steps):
        t = k * CONTROL_DT
        # 1. world -> eyes
        te0 = time.perf_counter()
        eye.update_panels(feedL.luminance_grid(*PANEL_GRID), feedR.luminance_grid(*PANEL_GRID))
        gL, gR = eye.grids(body)
        dL = encL.encode(gL, CONTROL_DT)
        dR = encR.encode(gR, CONTROL_DT)
        drive_idx = np.concatenate([dL["idx"], dR["idx"]])
        drive_rate = np.concatenate([dL["rate_hz"], dR["rate_hz"]]).astype(np.float32)
        t_eye += time.perf_counter() - te0
        # 2./3. brain
        tb0 = time.perf_counter()
        in_reward = plasticity_on and t < reward_until
        total = 0
        for _ in range(SUBSTEPS):
            if in_reward:
                if use_dopa_api:
                    brain.dopamine_drive(pam_idx, args.reward_hz)
                    fired = brain.step(drive_idx, drive_rate)
                else:
                    fired = brain.step(np.concatenate([drive_idx, pam_idx]),
                                       np.concatenate([drive_rate, np.full(pam_idx.size, args.reward_hz, np.float32)]))
            else:
                fired = brain.step(drive_idx, drive_rate)
            total += fired.size
            rec.add_step(fired)
        rec.end_frame()
        if not args.spikes:
            rec.frames.clear()   # keep only the population counts
        t_brain += time.perf_counter() - tb0
        cnt = {p: int(rec.pop_counts[p][-1]) for p in pops}

        # 4. decoder + contact gating
        want = dec.step(t, np.zeros(1, np.int32), cnt)
        burst_hz = dec.dn_fast / (n_dn * CONTROL_DT)
        reach_now = body.reach()      # contact gate at the current (pre-step) pose
        swipe: Optional[str] = None
        blocked: Optional[str] = None
        if want is not None:
            panel = reach_now[want]
            if cond != "random" and panel is not None:
                feeds[panel].swipe()
                swipe = panel
                n_swipes[panel] += 1
            else:
                blocked = want
                n_blocked += 1

        # 5. body
        for key, pop_key in (("dn", None), ("dna_L", "dna_L"), ("dna_R", "dna_R")):
            c = (cnt["dn_L"] + cnt["dn_R"]) if pop_key is None else cnt[pop_key]
            hz = c / (max(1, n_cells[key]) * CONTROL_DT)
            trace[key] = a_tr * trace[key] + (1.0 - a_tr) * hz
        body.step(CONTROL_DT, trace["dn"], trace["dna_L"], trace["dna_R"])
        on_phone = body.on_phone
        reach = body.reach()          # logged consistently with the post-step pose
        on_steps += int(on_phone)

        # 6. feeds + reward detection
        feedL.step(CONTROL_DT)
        feedR.step(CONTROL_DT)
        novel = {"L": bool(feedL.state["novel_visible"]), "R": bool(feedR.state["novel_visible"])}
        reward = on_phone and ((novel["L"] and not prev_novel["L"]) or (novel["R"] and not prev_novel["R"]))
        prev_novel = novel
        if reward:
            n_rewards += 1
            reward_until = t + CONTROL_DT + args.reward_s
        if plasticity_on:
            pst = brain.plasticity
            nd = pst.n_dopamine_steps if pst is not None else 0
            if nd != last_dopa_steps:     # weights only change in dopamine steps
                w_ratio = float(brain.plasticity_stats()["mean_w_over_w0"])
                last_dopa_steps = nd

        # 7. log
        log.write(json.dumps(dict(
            step=k, t=round(t, 4),
            pose=dict(x=round(body.x, 2), y=round(body.y, 2), th=round(body.th, 4), v=round(body.v, 2)),
            on_phone=bool(on_phone), dist_mm=round(body.dist_mm, 1),
            reach=reach, swipe=swipe, swipe_blocked=blocked,
            burst_hz=round(float(burst_hz), 3), side_ev=round(float(dec.last_side_evidence), 4), spikes=int(total),
            pops={p: cnt[p] for p in LOG_POPS},
            reward=bool(reward), w_ratio=round(w_ratio, 5),
            novel=novel, posts={"L": posts(feedL), "R": posts(feedR)},
        )) + "\n")

        if t >= next_print:
            next_print += 5.0
            el = time.time() - t_wall0
            print(f"t={t:5.1f}s  pose ({body.x:6.1f},{body.y:6.1f}) th {math.degrees(body.th):6.1f} v {body.v:4.1f}  "
                  f"on_phone {int(on_phone)} ({on_steps/(k+1)*100:4.1f}%)  dist {body.dist_mm:5.1f}  "
                  f"DN {trace['dn']:.2f} DNa L {trace['dna_L']:.2f} R {trace['dna_R']:.2f} Hz  "
                  f"swipes {n_swipes} blocked {n_blocked}  rewards {n_rewards}  w_ratio {w_ratio:.4f}  "
                  f"[{el:.0f}s, {el/(k+1)*1000:.1f} ms/step]", flush=True)
    log.close()

    wall = time.time() - t_wall0
    summary = dict(
        condition=cond, seed=args.seed, duration=args.duration, log=log_path,
        time_on_phone_frac=on_steps / max(1, n_steps), swipes=n_swipes, swipes_blocked=n_blocked, rewards=n_rewards,
        final_w_ratio=w_ratio, wall_hits=body.wall_hits, wall_s=wall, ms_per_control_step=wall / n_steps * 1000,
        ms_eye_encode_per_step=t_eye / n_steps * 1000, ms_brain_per_step=t_brain / n_steps * 1000,
        pop_rates_hz={p: float(np.sum(rec.pop_counts[p]) / args.duration / max(1, pops[p].size)) for p in pops},
        mean_pop_rate_hz=float(np.sum(rec.total) / n_steps / SUBSTEPS / conn.n / brain.dt_s),
    )
    if args.spikes:
        sp = os.path.join(args.out_dir, f"{tag}_spikes.npz")
        rec.save(sp, extra=dict(control_dt=np.float32(CONTROL_DT)))
        summary["spikes"] = sp
    with open(os.path.join(args.out_dir, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

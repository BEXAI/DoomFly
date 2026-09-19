#!/usr/bin/env python3
"""Offline closed loop: feeds -> eye encoder -> MaleCNS LIF brain -> motor readout -> swipes.

One control step = 16 ms = 8 LIF substeps of 2 ms (as hotocoo/malecns). Per control step:
  1. each panel's frame -> luminance grid -> Poisson rates for that eye's L1/L2(/L3) cells
  2. the brain runs 8 substeps with those cells forced to spike at the requested rates
  3. population spike counts (DNs, medulla L/R; DN + MN per cell in teacher/ridge/direct) update the decoder
  4. the decoder may emit swipe('L') / swipe('R'); the corresponding feed flicks
  5. the step is logged to events.jsonl; whole-brain spike bits go to spikes.npz

Modes:
  --mode burst     (default, used for the video) a swipe when the descending-neuron population
                   rate exceeds --burst-hz, to the side whose medulla is more active. No fit.
  --mode teacher   scripted swipes (calibration data collection); writes calib.npz with
                   features X (T, F) and teacher targets Y (T, 2). No brain->action link.
  --mode ridge     fitted linear readout (out/readout.npz) drives the swipes.
  --mode direct    front-leg motor-neuron pool rates drive the swipes.
  --scripted L:15,R:15   open-loop evaluation: swipes are scripted per panel block and the
                   decoder only logs what it would have done (`would_swipe`).
  --shuffle SEED   run on the wiring-shuffled control graph instead (Test C for the loop).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import Connectome, Brain, LIFConfig, SpikeRecorder  # noqa: E402
from feeds import make_pair  # noqa: E402
from encoder import make_encoders  # noqa: E402
from decoder import Decoder, DecoderConfig, load_readout  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=50.0, help="seconds of video/brain time")
    ap.add_argument("--mode", choices=["teacher", "ridge", "direct", "burst"], default="burst")
    ap.add_argument("--readout", default=os.path.join(ROOT, "out", "readout.npz"))
    ap.add_argument("--weight-scale", type=float, default=0.5)
    ap.add_argument("--adapt-mv", type=float, default=0.6)
    ap.add_argument("--std-u", type=float, default=0.15, help="short-term depression U per spike (flybrain: 0.08)")
    ap.add_argument("--std-tau", type=float, default=480.0, help="STD recovery time constant (ms)")
    ap.add_argument("--rate-max", type=float, default=150.0)
    ap.add_argument("--enc-gain", type=float, default=6.0)
    ap.add_argument("--l3-gain", type=float, default=0.0, help="sustained L3 drive gain (0 = L1/L2 only; L3 columns exist only on the right eye)")
    ap.add_argument("--tonic-hz", type=float, default=0.0, help="spontaneous baseline rate of every driven lamina cell (steady-light activity)")
    ap.add_argument("--grid", type=int, nargs=2, default=(24, 18))
    ap.add_argument("--threshold", type=float, default=None, help="z threshold for direct/ridge modes or burst mode with --burst-hz 0 (default: readout file value or 2.5)")
    ap.add_argument("--refractory", type=float, default=0.4)
    ap.add_argument("--tau-side", type=float, default=0.80, help="burst mode: side-evidence trace time constant (s)")
    ap.add_argument("--burst-hz", type=float, default=1.5, help="burst mode: swipe when DN population rate exceeds this (Hz/cell); 0 = z-score trigger")
    ap.add_argument("--autoplay", type=int, default=1, help="feeds autoplay cuts on (1) / static feeds (0)")
    ap.add_argument("--autoplay-amp", type=float, default=0.8)
    ap.add_argument("--autoplay-hz", type=float, default=1.0)
    ap.add_argument("--autoplay-whole", type=int, default=1)
    ap.add_argument("--feed-mode", choices=["cards", "reels"], default="cards", help="cards = synthetic card feed (all published results); reels = full-screen video posts from --clips-dir")
    ap.add_argument("--clips-dir", default=os.path.join(ROOT, "assets", "clips"))
    ap.add_argument("--teacher-gap", type=float, nargs=2, default=(0.9, 1.6), help="teacher: seconds between swipes on a panel (uniform range)")
    ap.add_argument("--scripted", default=None, help="evaluation: scripted swipes 'L:15,R:15' (panel:seconds blocks); the decoder only logs would_swipe")
    ap.add_argument("--teacher-panels", default="LR", help="teacher: which panels the teacher swipes (e.g. L for a left-only probe)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-l", type=int, default=1)
    ap.add_argument("--seed-r", type=int, default=2)
    ap.add_argument("--shuffle", type=int, default=None)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "out"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-spikes", action="store_true", help="skip whole-brain spike frames (faster, smaller)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or args.mode
    rng = np.random.default_rng(args.seed)

    control_dt = 0.016
    substeps = 8
    n_steps = int(round(args.duration / control_dt))

    print("loading connectome ...", flush=True)
    conn = Connectome.load()
    if args.shuffle is not None:
        print(f"using wiring-shuffled control graph (seed {args.shuffle})")
        conn = conn.shuffled(args.shuffle)
    cfg = LIFConfig(weight_scale=args.weight_scale, adapt_mv=args.adapt_mv, std_u=args.std_u, std_tau_rec_ms=args.std_tau)
    brain = Brain(conn, cfg, seed=args.seed)

    # populations
    pops = {
        "eye_L": np.concatenate([conn.idx("eye_left_L1"), conn.idx("eye_left_L2")]),
        "eye_R": np.concatenate([conn.idx("eye_right_L1"), conn.idx("eye_right_L2")]),
        "ol_L": conn.idx("ol_left"), "ol_R": conn.idx("ol_right"),
        "vpn_L": conn.idx("vpn_left"), "vpn_R": conn.idx("vpn_right"),
        "dn_L": conn.idx("dn_left"), "dn_R": conn.idx("dn_right"),
        "leg_L": conn.idx("mn_frontleg_left"), "leg_R": conn.idx("mn_frontleg_right"),
        # medulla cells one synapse downstream of the driven lamina cells (side evidence)
        "med_L": np.unique(np.concatenate([conn.idx(f"eye_left_{t}") for t in ("Mi1", "Tm1", "Tm2", "Tm9", "L5")])),
        "med_R": np.unique(np.concatenate([conn.idx(f"eye_right_{t}") for t in ("Mi1", "Tm1", "Tm2", "Tm9", "L5")])),
    }
    readout_cells = np.unique(np.concatenate([conn.idx("dn_all"), conn.idx("mn_all")]))
    ro_mask = np.zeros(conn.n, bool); ro_mask[readout_cells] = True
    ro_pos = -np.ones(conn.n, np.int64); ro_pos[readout_cells] = np.arange(readout_cells.size)
    pool_masks = {p: np.isin(readout_cells, pops[f"leg_{p}"]) for p in "LR"}
    print(f"readout cells: {readout_cells.size} (DN+MN); front-leg pools L={pool_masks['L'].sum()} R={pool_masks['R'].sum()}")

    # feeds + eyes
    feedL, feedR = make_pair(args.seed_l, args.seed_r, scale=0.25, mode=args.feed_mode, clips_dir=args.clips_dir, autoplay=bool(args.autoplay), autoplay_amp=args.autoplay_amp, autoplay_hz=args.autoplay_hz, autoplay_whole=bool(args.autoplay_whole))
    encL, encR = make_encoders(grid=tuple(args.grid), rate_max_hz=args.rate_max, gain=args.enc_gain, tonic_hz=args.tonic_hz, l3_gain=args.l3_gain)
    n_drv_L, n_drv_R = encL.encode(feedL.luminance_grid(*args.grid), 0.0)["idx"].size, encR.encode(feedR.luminance_grid(*args.grid), 0.0)["idx"].size
    print(f"eye L drives {n_drv_L} lamina cells, eye R {n_drv_R} (L1+L2{'+L3' if args.l3_gain > 0 else ''})")

    # decoder
    dcfg = DecoderConfig(control_dt_s=control_dt, refractory_s=args.refractory, burst_hz=args.burst_hz, tau_side_s=args.tau_side, n_dn=int(conn.idx("dn_all").size),
                         n_ol={"L": int(pops["med_L"].size), "R": int(pops["med_R"].size)})
    readout = None
    if args.mode == "ridge":
        readout = load_readout(args.readout)
        dcfg.threshold_z = args.threshold if args.threshold is not None else float(readout.get("thr", 2.5))
    elif args.threshold is not None:
        dcfg.threshold_z = args.threshold
    dec = Decoder(args.mode, dcfg, readout_cells.size, pool_masks, readout)

    rec = SpikeRecorder(conn.n, pops)
    events_path = os.path.join(args.out_dir, f"events_{tag}.jsonl")
    ev = open(events_path, "w")
    meta = dict(meta=dict(seed_l=args.seed_l, seed_r=args.seed_r, mode=args.mode, duration=args.duration,
                          control_dt=control_dt, substeps=substeps, lif=cfg.to_json(), n_neurons=conn.n,
                          n_edges=int(conn.W.nnz), shuffle=args.shuffle, threshold_z=dcfg.threshold_z,
                          readout_cells=int(readout_cells.size), grid=list(args.grid), rate_max=args.rate_max, tonic_hz=args.tonic_hz, burst_hz=args.burst_hz, autoplay=bool(args.autoplay), autoplay_amp=args.autoplay_amp, autoplay_hz=args.autoplay_hz, autoplay_whole=bool(args.autoplay_whole), enc_gain=args.enc_gain,
                          feed_mode=args.feed_mode, clips_dir=os.path.relpath(args.clips_dir, ROOT), clips=[c["name"] for c in feedL.clip_info] if args.feed_mode == "reels" else None,
                          l3_gain=args.l3_gain, seed=args.seed, refractory_s=args.refractory, global_refractory_s=dcfg.global_refractory_s, warmup_s=dcfg.warmup_s,
                          tau_side_s=args.tau_side, scripted=args.scripted, teacher_gap=list(args.teacher_gap), teacher_panels=args.teacher_panels,
                          driven_cells={"L": int(n_drv_L), "R": int(n_drv_R)}, n_dn=int(dcfg.n_dn),
                          leg_pool={"L": int(pool_masks["L"].sum()), "R": int(pool_masks["R"].sum())}))
    ev.write(json.dumps(meta) + "\n")

    X, Y = [], []
    next_teacher = {"L": rng.uniform(1.0, 1.6), "R": rng.uniform(1.3, 2.0)}
    script = None
    if args.scripted:
        script, t_acc = [], 0.0
        for blk in args.scripted.split(","):
            pnl, secs = blk.split(":")
            script.append((t_acc, t_acc + float(secs), pnl)); t_acc += float(secs)
        next_script = 1.0
    n_swipes = {"L": 0, "R": 0}
    t0 = time.time()
    counts_ro = np.zeros(readout_cells.size, np.int32)
    for k in range(n_steps):
        t = k * control_dt
        feedL.step(control_dt); feedR.step(control_dt)
        gL = feedL.luminance_grid(*args.grid); gR = feedR.luminance_grid(*args.grid)
        dL = encL.encode(gL, control_dt); dR = encR.encode(gR, control_dt)
        drive_idx = np.concatenate([dL["idx"], dR["idx"]])
        drive_rate = np.concatenate([dL["rate_hz"], dR["rate_hz"]]).astype(np.float32)

        counts_ro[:] = 0
        total = 0
        for _ in range(substeps):
            fired = brain.step(drive_idx, drive_rate)
            total += fired.size
            rec.add_step(fired)
            hit = fired[ro_mask[fired]]
            if hit.size:
                np.add.at(counts_ro, ro_pos[hit], 1)
        rec.end_frame()

        rec_pops = {p: rec.pop_counts[p][-1] for p in pops}
        swipe = dec.step(t, counts_ro, rec_pops)
        would = swipe
        if script is not None:
            swipe = None
            panel = next((p for a, b, p in script if a <= t < b), None)
            if panel is not None and t >= next_script:
                swipe = panel; next_script = t + rng.uniform(*args.teacher_gap)
        y = np.zeros(2, np.float32)
        swipes_now = [swipe] if swipe else []
        if args.mode == "teacher":
            for i, p in enumerate("LR"):
                if p in args.teacher_panels and t >= next_teacher[p]:
                    swipes_now.append(p)    # both panels may fall due in the same step
                    y[i] = 1.0
                    next_teacher[p] = t + rng.uniform(*args.teacher_gap)
            X.append(dec.last_features.copy()); Y.append(y)
            swipe = swipes_now[0] if swipes_now else None
        for p in swipes_now:
            (feedL if p == "L" else feedR).swipe(); n_swipes[p] += 1

        ev.write(json.dumps(dict(
            step=k, t=round(t, 4), swipe=swipe, would_swipe=would if script is not None else None,
            readout={"L": round(dec.last_value["L"], 3), "R": round(dec.last_value["R"], 3)},
            spikes=int(total),
            pops={p: int(rec_pops[p]) for p in ("eye_L", "eye_R", "dn_L", "dn_R", "leg_L", "leg_R", "vpn_L", "vpn_R")},
            posts={"L": int(feedL.state["posts_consumed"]), "R": int(feedR.state["posts_consumed"])},
            novel={"L": bool(feedL.state["novel_visible"]), "R": bool(feedR.state["novel_visible"])},
            eye={"L": [round(dL["on"], 1), round(dL["off"], 1)], "R": [round(dR["on"], 1), round(dR["off"], 1)]},
        )) + "\n")
        if k % 125 == 0:
            el = time.time() - t0
            print(f"t={t:5.1f}s  spikes/step={total:6d}  pop {total/substeps/conn.n/brain.dt_s:5.2f} Hz  "
                  f"eyeL {rec_pops['eye_L']:4d} eyeR {rec_pops['eye_R']:4d}  dnL {rec_pops['dn_L']:3d} dnR {rec_pops['dn_R']:3d}  "
                  f"legL {rec_pops['leg_L']:2d} legR {rec_pops['leg_R']:2d}  z L {dec.last_value['L']:+.2f} R {dec.last_value['R']:+.2f}  "
                  f"swipes {n_swipes}  posts L{feedL.state['posts_consumed']} R{feedR.state['posts_consumed']}  "
                  f"[{el:.0f}s, {el/(k+1)*1000:.0f} ms/step]", flush=True)
    ev.close()
    wall = time.time() - t0
    summary = dict(mode=args.mode, duration=args.duration, swipes=n_swipes,
                   posts={"L": int(feedL.state["posts_consumed"]), "R": int(feedR.state["posts_consumed"])},
                   total_spikes=int(np.sum(rec.total)), mean_pop_rate_hz=float(np.sum(rec.total) / n_steps / substeps / conn.n / brain.dt_s),
                   wall_s=wall, ms_per_control_step=wall / n_steps * 1000, events=events_path,
                   pop_rates_hz={p: float(np.sum(rec.pop_counts[p]) / args.duration / max(1, pops[p].size)) for p in pops})
    if not args.no_spikes:
        sp = os.path.join(args.out_dir, f"spikes_{tag}.npz")
        rec.save(sp, extra=dict(control_dt=np.float32(control_dt)))
        summary["spikes"] = sp
    if args.mode == "teacher":
        cp = os.path.join(args.out_dir, "calib.npz")
        np.savez_compressed(cp, X=np.stack(X).astype(np.float32), Y=np.stack(Y), cells=readout_cells)
        summary["calib"] = cp
    with open(os.path.join(args.out_dir, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

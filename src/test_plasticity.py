#!/usr/bin/env python3
"""Tests for the KC->MBON dopamine-gated depression in `src/sim.py` (docs/ARENA_DESIGN.md,
"Dopamine / mushroom-body plasticity").

Rule tests (PASS/FAIL).  The rest of the mushroom body is CLAMPED: every KC and PAM cell is in
the forced-Poisson drive (non-driven ones at 0 Hz, which the drive mechanism silences), so the
tests probe the plasticity rule and not the recurrent dynamics of the network:
    A  200 KCs @ 50 Hz + all PAM @ 100 Hz for 1 s -> mean w/w0 on the driven KCs' edges < 0.9,
                                                     edges of the other KCs == 1.0 exactly
    B  same KC drive, PAM silent                   -> every edge == 1.0
    C  PAM only, KCs silent                        -> every edge == 1.0 (no eligibility trace)
    D  dopamine_drive merges with the sensory drive, one-shot, no double counting
Open-network diagnostics (informational): the same three protocols with nothing clamped, which
show how far the KC / PAM drive spreads through the wiring (PAM->KC synapses, MBON->PAM feedback).
    E  visual drive -> KC / MBON / PAM / DN population rates: does the visual pathway reach the
       mushroom body at all?  (1) spec: eye_left_L1+L2 @ 60 Hz, 1 s;  (2) strong: both eyes
       L1+L2 @ 150 Hz (the encoder's rate_max), 1 s.
Also prints ms/step with plasticity enabled (active vs idle) against the same run with it off.

Usage: python3 src/test_plasticity.py   (exit code 0 iff all rule tests pass)
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import Brain, Connectome, LIFConfig  # noqa: E402

# The arena LIF configuration (docs/ARENA_DESIGN.md; run_episode.py defaults).
CFG = LIFConfig(weight_scale=0.5, adapt_mv=0.6, std_u=0.15)
STEPS = 500          # 1 s at dt = 2 ms
N_KC_DRIVEN = 200
KC_HZ, PAM_HZ, EYE_HZ, EYE_HZ_STRONG = 50.0, 100.0, 60.0, 150.0

Drive = Optional[Tuple[np.ndarray, np.ndarray]]


def run(brain: Brain, drive: Drive, dopa: Optional[Tuple[np.ndarray, float]], steps: int,
        pops: Dict[str, np.ndarray]) -> Tuple[Dict[str, float], float, np.ndarray]:
    """Run `steps` steps; returns (population rates Hz/cell, ms per step, spike count per neuron)."""
    masks = {k: np.zeros(brain.n, bool) for k in pops}
    for k, idx in pops.items():
        masks[k][idx] = True
    counts = {k: 0 for k in pops}
    per_cell = np.zeros(brain.n, np.int64)
    d_idx: Optional[np.ndarray] = drive[0] if drive is not None else None
    d_rate: Optional[np.ndarray] = drive[1] if drive is not None else None
    t0 = time.perf_counter()
    for _ in range(steps):
        if dopa is not None:
            brain.dopamine_drive(dopa[0], dopa[1])
        fired = brain.step(d_idx, d_rate)
        per_cell[fired] += 1
        for k, m in masks.items():
            counts[k] += int(m[fired].sum())
    ms = (time.perf_counter() - t0) / steps * 1000.0
    t_s = steps * brain.dt_s
    rates = {k: counts[k] / max(pops[k].size, 1) / t_s for k in pops}
    return rates, ms, per_cell


def edge_ratios(brain: Brain, kc_subset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(w/w0 of edges whose KC is in kc_subset, w/w0 of all other edges)."""
    e = brain.plasticity_edges()
    inmask = np.isin(e["kc"], kc_subset)
    return e["w_over_w0"][inmask], e["w_over_w0"][~inmask]


def fmt_rates(r: Dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.3f}" for k, v in r.items())


def main() -> int:
    conn = Connectome.load()
    kc, mbon, pam = conn.idx("kc"), conn.idx("mbon"), conn.idx("pam")
    pops = {"kc": kc, "mbon": mbon, "pam": pam}
    rng = np.random.default_rng(1)
    kc_driven = np.sort(rng.choice(kc, N_KC_DRIVEN, replace=False))
    results: Dict[str, bool] = {}
    print(f"N={conn.n:,}  KC={kc.size}  MBON={mbon.size}  PAM={pam.size}")
    print(f"cfg: weight_scale={CFG.weight_scale} adapt_mv={CFG.adapt_mv} std_u={CFG.std_u} dt={CFG.dt_ms} ms")

    def fresh(plastic: bool = True, quiet: bool = False) -> Brain:
        b = Brain(conn, CFG, seed=0)
        if plastic:
            n = b.enable_plasticity(kc, mbon, pam, eta=0.05, tau_trace_s=1.0)
            p = b.plasticity
            assert p is not None
            if not quiet:
                print(f"  enable_plasticity: n_edges={n:,}  MBONs with PAM input={p.n_mbon_with_dan}/{mbon.size}  "
                      f"mean_m sum_d G={p.G.sum(1).mean():.3f}  G.max={p.G.max():.3f}")
        return b

    def clamped(kc_hz: float, pam_hz: float) -> Drive:
        """Forced drive covering every KC and PAM: driven KCs at kc_hz, all PAM at pam_hz,
        the remaining KCs at 0 Hz (silenced by the drive mechanism)."""
        idx = np.concatenate([kc, pam])
        rate = np.zeros(idx.size, np.float32)
        rate[np.isin(idx, kc_driven)] = kc_hz
        rate[kc.size:] = pam_hz
        return idx, rate

    # ================================================================ rule tests (clamped)
    print("\n[A] rule: 200 KCs @ 50 Hz + all PAM @ 100 Hz, 1 s, other KCs clamped silent")
    b = fresh()
    rates, ms_active, per_cell = run(b, clamped(KC_HZ, PAM_HZ), None, STEPS, pops)
    st = b.plasticity_stats()
    r_in, r_out = edge_ratios(b, kc_driven)
    p = b.plasticity
    assert p is not None
    ok_a1 = bool(r_in.mean() < 0.9)
    ok_a2 = bool(np.all(r_out == 1.0))
    ok_a3 = bool(r_in.min() >= 0.2 - 1e-6 and np.all(np.sign(b.W.data[p.pos]) == p.w0_sign))
    print(f"  rates Hz/cell: {fmt_rates(rates)}   {ms_active:.2f} ms/step   dopamine steps={p.n_dopamine_steps}")
    print(f"  stats: {st}")
    print(f"  driven KCs' edges: n={r_in.size} mean w/w0={r_in.mean():.4f} min={r_in.min():.4f} "
          f"frac at floor={(r_in <= 0.2 + 1e-6).mean():.3f}  -> {'PASS' if ok_a1 else 'FAIL'} (mean < 0.9)")
    print(f"  other KCs' edges: n={r_out.size} all == 1.0 -> {'PASS' if ok_a2 else 'FAIL'}")
    print(f"  floor 0.2*w0 and sign preserved -> {'PASS' if ok_a3 else 'FAIL'}")
    results["A1 driven KC edges depressed (mean < 0.9)"] = ok_a1
    results["A2 non-driven KC edges exactly unchanged"] = ok_a2
    results["A3 floor 0.2*w0 and sign preserved"] = ok_a3
    ratio_a = st["mean_w_over_w0"]

    print("\n[B] rule: same KC drive, PAM clamped silent")
    b = fresh()
    rates, ms_b, _ = run(b, clamped(KC_HZ, 0.0), None, STEPS, pops)
    e = b.plasticity_edges()["w_over_w0"]
    ok_b = bool(np.all(e == 1.0))
    print(f"  rates Hz/cell: {fmt_rates(rates)}   {ms_b:.2f} ms/step")
    print(f"  stats: {b.plasticity_stats()}  -> {'PASS' if ok_b else 'FAIL'} (all edges == 1.0)")
    results["B KC drive without dopamine: unchanged"] = ok_b

    print("\n[C] rule: PAM @ 100 Hz only, KCs clamped silent")
    b = fresh()
    rates, ms_c, _ = run(b, clamped(0.0, PAM_HZ), None, STEPS, pops)
    e = b.plasticity_edges()["w_over_w0"]
    ok_c = bool(np.all(e == 1.0))
    print(f"  rates Hz/cell: {fmt_rates(rates)}   {ms_c:.2f} ms/step")
    print(f"  stats: {b.plasticity_stats()}  -> {'PASS' if ok_c else 'FAIL'} (all edges == 1.0)")
    results["C dopamine without KC trace: unchanged"] = ok_c

    print("\n[D] dopamine_drive merges with the sensory drive (one-shot, no double counting)")
    b = fresh()
    one = pam[:8]
    sens_idx = np.concatenate([kc_driven[:5], one])          # these DANs are ALSO in the sensory set, at 0 Hz
    sens_rate = np.concatenate([np.full(5, KC_HZ), np.zeros(one.size)]).astype(np.float32)
    b.dopamine_drive(one, 1e6)                              # p = 1: must fire despite the 0 Hz sensory entry
    fired = b.step(sens_idx, sens_rate)
    ok_d1 = bool(np.isin(one, fired).all())
    fired2 = b.step(sens_idx, sens_rate)                    # cleared after one step: the 0 Hz entry silences them
    ok_d2 = (not bool(np.isin(one, fired2).any())) and b._dopa_idx is None
    b.dopamine_drive(one, 100.0)
    m_idx, m_rate = b._merge_drive(sens_idx, sens_rate)
    ok_d3 = bool(np.unique(m_idx).size == m_idx.size and m_idx.size == sens_idx.size
                 and np.all(m_rate[np.isin(m_idx, one)] == 100.0))
    b._dopa_idx = b._dopa_rate = None
    ok_d = ok_d1 and ok_d2 and ok_d3
    print(f"  forced this step={ok_d1}  cleared next step={ok_d2}  merged unique with max rate={ok_d3}"
          f"  -> {'PASS' if ok_d else 'FAIL'}")
    results["D dopamine_drive merge / one-shot"] = ok_d

    # ================================================================ open-network diagnostics
    print("\n[open network] same protocols, nothing clamped, PAM via dopamine_drive (informational)")
    for label, drive, dopa in (("KC 50 Hz + PAM 100 Hz", (kc_driven, np.full(kc_driven.size, KC_HZ, np.float32)), (pam, PAM_HZ)),
                               ("KC 50 Hz, no PAM drive", (kc_driven, np.full(kc_driven.size, KC_HZ, np.float32)), None),
                               ("PAM 100 Hz, no KC drive", None, (pam, PAM_HZ))):
        b = fresh(quiet=True)
        rates, ms, per_cell = run(b, drive, dopa, STEPS, pops)
        st = b.plasticity_stats()
        r_in, r_out = edge_ratios(b, kc_driven)
        n_kc_fired = int((per_cell[kc] > 0).sum())
        n_pam_fired = int((per_cell[pam] > 0).sum())
        print(f"  {label:26s} rates {fmt_rates(rates)} | KCs that spiked {n_kc_fired}/{kc.size}, PAMs {n_pam_fired}/{pam.size}"
              f" | mean w/w0 all={st['mean_w_over_w0']:.4f} driven={r_in.mean():.4f} others={r_out.mean():.4f}"
              f" frac_changed={st['frac_edges_changed']:.3f}")
    print("  reading: PAM->KC synapses (2,595 edges, dopamine = excitatory in the Shiu sign convention) let a"
          " 100 Hz PAM drive recruit most KCs, and KC drive alone makes PAMs fire via MB feedback (KC->PAM,"
          " MBON->PAM), so in the open network dopamine depresses far beyond the driven KCs. The clamped"
          " tests above show the rule itself only touches KCs with a trace.")

    # ================================================================ E: visual drive
    print("\n[E] visual drive -> mushroom body? (plasticity enabled, no dopamine_drive; informational)")
    dn_all = conn.idx("dn_all")
    pops_e = dict(pops, dn_all=dn_all, ol_left=conn.idx("ol_left"), ol_right=conn.idx("ol_right"))
    eye_l = np.concatenate([conn.idx("eye_left_L1"), conn.idx("eye_left_L2")])
    eye_lr = np.concatenate([eye_l, conn.idx("eye_right_L1"), conn.idx("eye_right_L2")])
    for label, eye, hz in ((f"spec: eye_left_L1+L2 @ {EYE_HZ:.0f} Hz", eye_l, EYE_HZ),
                           (f"strong: both eyes L1+L2 @ {EYE_HZ_STRONG:.0f} Hz", eye_lr, EYE_HZ_STRONG)):
        b = fresh(quiet=True)
        rates, ms_e, per_cell = run(b, (eye, np.full(eye.size, hz, np.float32)), None, STEPS, pops_e)
        forced = eye.size * hz * b.dt_s
        print(f"  {label}: driven cells={eye.size}, {ms_e:.2f} ms/step, spikes/step={per_cell.sum() / STEPS:.0f}"
              f" (of which ~{forced:.0f} forced)")
        print(f"    population rates Hz/cell: {fmt_rates(rates)}")
        print(f"    cells that spiked at least once: KC {int((per_cell[kc] > 0).sum())}/{kc.size}, "
              f"MBON {int((per_cell[mbon] > 0).sum())}/{mbon.size}, PAM {int((per_cell[pam] > 0).sum())}/{pam.size}, "
              f"DN {int((per_cell[dn_all] > 0).sum())}/{dn_all.size}")
        print(f"    plasticity stats: {b.plasticity_stats()}")

    # ================================================================ timings
    print("\n[timing] ms/step over 500 steps (4 CPUs, scipy CSC gather per step)")
    b = fresh(quiet=True)
    _, ms_idle, _ = run(b, None, None, STEPS, pops)
    b = fresh(plastic=False)
    _, ms_off_active, _ = run(b, clamped(KC_HZ, PAM_HZ), None, STEPS, pops)
    b = fresh(plastic=False)
    _, ms_off_idle, _ = run(b, None, None, STEPS, pops)
    print(f"  plasticity ON  active (test A drive): {ms_active:.2f}   idle (no drive, no spikes): {ms_idle:.2f}")
    print(f"  plasticity OFF same drives:           {ms_off_active:.2f}   idle: {ms_off_idle:.2f}")
    print(f"  overhead: active +{ms_active - ms_off_active:.2f} ms/step, idle {ms_idle - ms_off_idle:+.2f} ms/step")

    print(f"\n[summary]  n_edges={p.pos.size:,}  test-A mean w/w0 (all edges)={ratio_a:.4f}")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

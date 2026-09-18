#!/usr/bin/env python3
"""Motor readout: connectome output-neuron activity -> swipe events.

Two modes, both operating only on spikes of real MaleCNS output neurons
(descending neurons + motor neurons, ~2,100 cells):

  direct  The mean rate of the front-left / front-right leg motor-neuron pools is
          low-passed, z-scored against a running baseline, and a swipe is emitted on
          the panel of the pool whose z crosses `threshold` (refractory >= 250 ms).
          Used when the leg MNs respond to visual drive on their own.

  burst   No fit. A swipe is triggered when the z-scored fast trace of the whole
          descending-neuron population crosses `threshold` (a DN burst); the panel is the
          side whose optic lobe (intrinsic cells downstream of the driven lamina cells,
          per cell) carried more of the recent activity, i.e. the eye that saw the
          change. Uses only real-neuron spike counts; nothing is fitted.

  ridge   A fitted *linear* readout (ridge regression) from two exponential traces
          (fast ~80 ms, slow ~800 ms) of every output cell's rate to a teacher signal
          ("swipe this panel now"), as hotocoo/malecns `calibrate.py` does. The only
          fitted object is the weight vector over real neurons; the brain dynamics are
          untouched. Fit with `python3 src/decoder.py fit out/calib.npz out/readout.npz`.

Features per control step (16 ms): for each readout cell, fast and slow EMA of its
spike count, minus the population mean of the same trace (removes global rate swings).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class DecoderConfig:
    control_dt_s: float = 0.016
    tau_fast_s: float = 0.080
    tau_slow_s: float = 0.800
    tau_baseline_s: float = 4.0     # running mean/std horizon for z-scoring
    threshold_z: float = 2.5
    refractory_s: float = 0.40
    warmup_s: float = 1.0           # no swipes before this
    burst_hz: float = 0.0           # burst mode: absolute DN population-rate threshold (Hz/cell); 0 = use z
    n_dn: int = 1314
    n_ol: dict = None               # per-side optic-lobe cell counts for side evidence
    global_refractory_s: float = 0.30  # minimum gap between any two swipes


class Features:
    """Fast/slow EMA traces over a fixed set of readout cells."""

    def __init__(self, n_cells: int, cfg: DecoderConfig):
        self.cfg = cfg
        self.n = n_cells
        self.a_fast = float(np.exp(-cfg.control_dt_s / cfg.tau_fast_s))
        self.a_slow = float(np.exp(-cfg.control_dt_s / cfg.tau_slow_s))
        self.fast = np.zeros(n_cells, np.float32)
        self.slow = np.zeros(n_cells, np.float32)

    def update(self, counts: np.ndarray) -> np.ndarray:
        """counts: spikes of each readout cell during this control step. Returns (2n,) features."""
        c = counts.astype(np.float32) / self.cfg.control_dt_s  # Hz
        self.fast = self.a_fast * self.fast + (1 - self.a_fast) * c
        self.slow = self.a_slow * self.slow + (1 - self.a_slow) * c
        f = np.concatenate([self.fast - self.fast.mean(), self.slow - self.slow.mean()])
        return f


class RunningZ:
    def __init__(self, tau_s: float, dt_s: float):
        self.a = float(np.exp(-dt_s / tau_s))
        self.m = 0.0
        self.v = 1.0
        self.n = 0

    def update(self, x: float) -> float:
        if self.n < 3:
            self.m = x if self.n == 0 else 0.5 * (self.m + x)
            self.n += 1
            return 0.0
        z = (x - self.m) / (np.sqrt(self.v) + 1e-6)
        self.m = self.a * self.m + (1 - self.a) * x
        self.v = self.a * self.v + (1 - self.a) * (x - self.m) ** 2
        self.n += 1
        return float(np.clip(z, -10, 10))


class Decoder:
    """Turns per-step readout-cell spike counts into swipe events."""

    def __init__(self, mode: str, cfg: DecoderConfig, n_cells: int,
                 pool_masks: Optional[Dict[str, np.ndarray]] = None,
                 readout: Optional[Dict[str, np.ndarray]] = None):
        assert mode in ("direct", "ridge", "teacher", "burst")
        self.mode = mode
        self.cfg = cfg
        self.feats = Features(n_cells, cfg)
        self.pool_masks = pool_masks or {}
        self.readout = readout
        self.z = {"L": RunningZ(cfg.tau_baseline_s, cfg.control_dt_s),
                  "R": RunningZ(cfg.tau_baseline_s, cfg.control_dt_s)}
        self.last_swipe = {"L": -1e9, "R": -1e9}
        self.last_value = {"L": 0.0, "R": 0.0}
        self.last_features: Optional[np.ndarray] = None
        # burst mode state: fast traces of DN population and of VPN L / R
        self.z_dn = RunningZ(cfg.tau_baseline_s, cfg.control_dt_s)
        self.a_fast = float(np.exp(-cfg.control_dt_s / cfg.tau_fast_s))
        self.dn_fast = 0.0
        self.vpn_fast = {"L": 0.0, "R": 0.0}
        # per-side normalisation of the side evidence (removes the structural L/R bias of
        # the reconstruction: the right optic lobe carries ~10% more synapses)
        self.z_side = {"L": RunningZ(cfg.tau_baseline_s, cfg.control_dt_s),
                       "R": RunningZ(cfg.tau_baseline_s, cfg.control_dt_s)}
        if self.cfg.n_ol is None:
            self.cfg.n_ol = {"L": 1, "R": 1}
        self.last_side_evidence = 0.0

    def step(self, t: float, counts: np.ndarray, aux: Optional[Dict[str, int]] = None) -> Optional[str]:
        """counts: spikes of each readout cell this control step; aux: population spike
        counts this step (needs 'dn_L','dn_R','vpn_L','vpn_R' for burst mode)."""
        f = self.feats.update(counts)
        self.last_features = f
        if self.mode == "teacher":
            return None
        if self.mode == "burst":
            return self._step_burst(t, aux or {})
        raw = {}
        if self.mode == "direct":
            for p in ("L", "R"):
                m = self.pool_masks[p]
                raw[p] = float(self.feats.fast[m].mean())
        else:
            for p in ("L", "R"):
                raw[p] = float(f @ self.readout["w_" + p] + self.readout["b_" + p])
        out = None
        best = None
        for p in ("L", "R"):
            z = self.z[p].update(raw[p]) if self.mode == "direct" else raw[p]
            self.last_value[p] = z
            if t >= self.cfg.warmup_s and z > self.cfg.threshold_z and t - self.last_swipe[p] >= self.cfg.refractory_s:
                if best is None or z > best:
                    best, out = z, p
        if out is not None:
            self.last_swipe[out] = t
        return out


    def _step_burst(self, t: float, aux: Dict[str, int]) -> Optional[str]:
        a = self.a_fast
        dn = float(aux.get("dn_L", 0) + aux.get("dn_R", 0))
        self.dn_fast = a * self.dn_fast + (1 - a) * dn
        for p in "LR":
            # side evidence: this side's optic-lobe activity *downstream* of the driven
            # lamina cells (ol minus L1/L2), per cell; falls back to VPNs if absent
            if "ol_" + p in aux:
                x = (float(aux["ol_" + p]) - float(aux.get("eye_" + p, 0))) / max(1.0, float(self.cfg.n_ol.get(p, 1)))
            else:
                x = float(aux.get("vpn_" + p, 0))
            self.vpn_fast[p] = a * self.vpn_fast[p] + (1 - a) * x
        if self.cfg.burst_hz > 0:
            # DN population rate in Hz per cell (fast trace of spikes per control step)
            z = self.dn_fast / (self.cfg.n_dn * self.cfg.control_dt_s) / self.cfg.burst_hz * self.cfg.threshold_z
        else:
            z = self.z_dn.update(self.dn_fast)
        zs = {p: self.z_side[p].update(self.vpn_fast[p]) for p in "LR"}
        ev = zs["L"] - zs["R"]
        self.last_side_evidence = ev
        # expose per-panel values for logging: burst signal assigned to the evidenced side
        self.last_value = {"L": z if ev >= 0 else 0.0, "R": z if ev < 0 else 0.0}
        if t < self.cfg.warmup_s or z <= self.cfg.threshold_z:
            return None
        if abs(ev) < 1e-6:
            return None
        p = "L" if ev > 0 else "R"
        if t - self.last_swipe[p] < self.cfg.refractory_s or t - max(self.last_swipe.values()) < self.cfg.global_refractory_s:
            return None
        self.last_swipe[p] = t
        return p


# ------------------------------------------------------------------ ridge fit
def fit_ridge(calib_path: str, out_path: str, lambdas=(1, 3, 10, 30, 100, 300, 1000, 3000)) -> Dict:
    """Fit w_L, w_R from calibration features X (T, F) to teacher targets Y (T, 2) in {0,1}.
    Held-out selection of lambda by alternating blocks; returns diagnostics."""
    d = np.load(calib_path)
    X = d["X"].astype(np.float64)
    Y = d["Y"].astype(np.float64)
    T, F = X.shape
    mu = X.mean(0)
    Xc = X - mu
    blocks = (np.arange(T) // 250) % 2 == 0   # alternate 4 s blocks train / test
    best = None
    G_tr = Xc[blocks].T @ Xc[blocks]
    for lam in lambdas:
        W = np.linalg.solve(G_tr + lam * np.eye(F), Xc[blocks].T @ Y[blocks])
        pred = Xc[~blocks] @ W
        r2 = 1 - ((pred - Y[~blocks]) ** 2).sum(0) / ((Y[~blocks] - Y[~blocks].mean(0)) ** 2).sum(0)
        corr = [np.corrcoef(pred[:, i], Y[~blocks][:, i])[0, 1] for i in range(2)]
        if best is None or np.mean(corr) > best[1]:
            best = (lam, float(np.mean(corr)), r2.tolist(), [float(c) for c in corr])
    lam = best[0]
    W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(F), Xc.T @ Y)
    pred = Xc @ W
    # scale each channel to unit std so `threshold_z` means the same as in direct mode
    sd = pred.std(0) + 1e-9
    W = W / sd
    b = -(mu @ W)
    pred_z = X @ W + b
    # threshold that reproduces the teacher's swipe count on the calibration set
    n_target = int(Y.sum(0).mean())
    thr = float(np.quantile(pred_z.max(1), 1 - n_target / T)) if n_target > 0 else 2.5
    np.savez(out_path, w_L=W[:, 0].astype(np.float32), w_R=W[:, 1].astype(np.float32),
             b_L=np.float32(b[0]), b_R=np.float32(b[1]), lam=np.float32(lam), thr=np.float32(thr),
             heldout_corr=np.asarray(best[3], np.float32), heldout_r2=np.asarray(best[2], np.float32))
    diag = dict(lambda_=lam, heldout_corr_L=best[3][0], heldout_corr_R=best[3][1],
                heldout_r2=best[2], suggested_threshold=thr, T=T, F=F, teacher_swipes=int(Y.sum()))
    # lateral specificity: does the L readout respond more to L-teacher events than R-teacher?
    for i, p in enumerate("LR"):
        on = pred_z[Y[:, i] > 0, i].mean() if (Y[:, i] > 0).any() else float("nan")
        other = pred_z[Y[:, 1 - i] > 0, i].mean() if (Y[:, 1 - i] > 0).any() else float("nan")
        diag[f"readout_{p}_mean_at_own_events"] = float(on)
        diag[f"readout_{p}_mean_at_other_events"] = float(other)
    print(json.dumps(diag, indent=1))
    return diag


def load_readout(path: str) -> Dict[str, np.ndarray]:
    d = np.load(path)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "fit":
        fit_ridge(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)

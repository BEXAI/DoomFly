#!/usr/bin/env python3
"""Leaky integrate-and-fire simulation of the MaleCNS v1.0 connectome (CPU, scipy).

Model (Shiu et al., Nature 2024; constants as in philshiu/Drosophila_brain_model and
hotocoo/malecns `brain.py`):

    tau_m dV/dt = -(V - V_rest) + I_syn
    tau_s dI/dt = -I,   presynaptic spike -> I_post += w_syn * |synapses| * sign(pre)

    V_rest = V_reset = -52 mV, V_thresh = -45 mV, t_ref = 2.2 ms,
    tau_m = 20 ms, tau_s = 5 ms, w_syn = 0.275 mV / synapse, delay = 1.8 ms.

Sensory input is delivered the way Shiu et al. do it: the driven neurons are made
to spike as Poisson processes at a requested rate (their own membrane is bypassed),
and those spikes propagate through the real wiring.  Nothing downstream is scripted.

Graph format (produced by `src/build_graph.py`):
    data/graph/W.npz      scipy CSC matrix, shape (N, N), W[post, pre] = signed synapse count
    data/graph/ids.npy    int64 bodyId per row/col
    data/graph/sets.json  name -> list of bodyIds

Optional spike-frequency adaptation (`adapt_mv`, hotocoo's addition, not in Shiu 2024)
and a global `weight_scale` are exposed for the runaway checks in `src/validate.py`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp

GRAPH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "graph")


@dataclass
class LIFConfig:
    v_rest_mv: float = -52.0
    v_reset_mv: float = -52.0
    v_thresh_mv: float = -45.0
    t_ref_ms: float = 2.2
    tau_m_ms: float = 20.0
    tau_s_ms: float = 5.0
    w_syn_mv: float = 0.275
    delay_ms: float = 1.8
    dt_ms: float = 2.0
    weight_scale: float = 1.0
    adapt_mv: float = 0.0        # 0 = pure Shiu model; hotocoo uses 0.6
    tau_adapt_ms: float = 120.0
    # Tsodyks-Markram short-term depression, one resource per PRESYNAPTIC neuron
    # (TheMrRaGe/flybrain `enable_std`: U measured 0.08, recovery 480 ms; must be global).
    # 0 = off. Keeps recurrent loops self-limiting so the network does not latch.
    std_u: float = 0.0
    std_tau_rec_ms: float = 480.0

    def to_json(self) -> Dict:
        return asdict(self)


class Connectome:
    """Signed sparse graph plus bodyId index and named neuron sets."""

    def __init__(self, W: sp.spmatrix, ids: np.ndarray, sets: Dict[str, List[int]]):
        self.W = sp.csc_matrix(W, dtype=np.float32)
        self.ids = np.asarray(ids, dtype=np.int64)
        self.n = len(self.ids)
        assert self.W.shape == (self.n, self.n), self.W.shape
        self.body_to_idx = {int(b): i for i, b in enumerate(self.ids)}
        self.sets = sets
        self._idx_cache: Dict[str, np.ndarray] = {}

    @classmethod
    def load(cls, graph_dir: str = GRAPH_DIR) -> "Connectome":
        W = sp.load_npz(os.path.join(graph_dir, "W.npz"))
        ids = np.load(os.path.join(graph_dir, "ids.npy"))
        with open(os.path.join(graph_dir, "sets.json")) as f:
            sets = json.load(f)
        return cls(W, ids, sets)

    def idx(self, name_or_bodies) -> np.ndarray:
        """Row indices for a named set or an explicit list of bodyIds (missing ids dropped)."""
        if isinstance(name_or_bodies, str):
            if name_or_bodies not in self._idx_cache:
                self._idx_cache[name_or_bodies] = self.idx(self.sets[name_or_bodies])
            return self._idx_cache[name_or_bodies]
        out = [self.body_to_idx[int(b)] for b in name_or_bodies if int(b) in self.body_to_idx]
        return np.asarray(sorted(set(out)), dtype=np.int64)

    def shuffled(self, seed: int = 0) -> "Connectome":
        """Degree-preserving-ish control: keep every neuron's out-degree list, but
        randomly permute which postsynaptic neuron each edge lands on. Weights and
        signs travel with the edge, so the weight distribution and each presynaptic
        neuron's total output are preserved; only the *wiring* is destroyed."""
        rng = np.random.default_rng(seed)
        coo = self.W.tocoo()
        rows = rng.permutation(coo.row)
        Ws = sp.csc_matrix((coo.data, (rows, coo.col)), shape=self.W.shape, dtype=np.float32)
        Ws.sum_duplicates()
        return Connectome(Ws, self.ids, self.sets)


class Brain:
    """Single-instance LIF network over a Connectome, Euler-integrated at dt."""

    def __init__(self, conn: Connectome, cfg: Optional[LIFConfig] = None, seed: int = 0):
        self.conn = conn
        self.cfg = cfg or LIFConfig()
        c = self.cfg
        self.n = conn.n
        self.rng = np.random.default_rng(seed)
        dt = c.dt_ms
        self.dt_s = dt / 1000.0
        self.decay_v = float(np.exp(-dt / c.tau_m_ms))
        self.decay_s = float(np.exp(-dt / c.tau_s_ms))
        self.decay_a = float(np.exp(-dt / c.tau_adapt_ms))
        # Exponential synapse through a membrane: pre-compensate the tau_s/tau_m loss
        # so that one synapse of weight 1 yields a peak PSP of ~w_syn (as hotocoo).
        v_scale = dt / c.tau_m_ms
        psp_gain = c.tau_m_ms / c.tau_s_ms
        self.syn_kick = v_scale * c.w_syn_mv * psp_gain * c.weight_scale
        self.adapt_kick = v_scale * c.adapt_mv * psp_gain
        self.u_thresh = c.v_thresh_mv - c.v_rest_mv
        self.u_reset = c.v_reset_mv - c.v_rest_mv
        self.ref_steps = max(1, int(round(c.t_ref_ms / dt)))
        self.delay_steps = max(1, int(round(c.delay_ms / dt)))
        # Pre-scale the matrix once so the per-step matvec has no extra pass.
        self.W = (conn.W * np.float32(self.syn_kick)).tocsc()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        n = self.n
        self.u = np.zeros(n, np.float32)        # membrane relative to rest (mV)
        self.j_syn = np.zeros(n, np.float32)    # synaptic drive (mV per step)
        self.adapt = np.zeros(n, np.float32)
        self.refrac = np.zeros(n, np.int16)
        self.std_x = np.ones(n, np.float32)     # available synaptic resource per presynaptic cell
        self.ring: List[np.ndarray] = [np.zeros(0, np.int64) for _ in range(self.delay_steps)]
        self.pos = 0
        self.t_steps = 0
        self.last_fired = np.zeros(0, np.int64)

    # ------------------------------------------------------------------ core
    def _drive_from(self, fired: np.ndarray) -> Optional[np.ndarray]:
        """Sum of the W columns of the neurons that fired (sparse column gather)."""
        if fired.size == 0:
            return None
        cols = self.W[:, fired]
        if self.cfg.std_u:
            return np.asarray(cols @ self.std_x[fired]).ravel().astype(np.float32, copy=False)
        return np.asarray(cols.sum(axis=1)).ravel().astype(np.float32, copy=False)

    def step(self, drive_idx: Optional[np.ndarray] = None, drive_rate_hz: Optional[np.ndarray] = None,
             extra_mv: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance one dt. Returns the int64 indices of neurons that spiked.

        drive_idx / drive_rate_hz: neurons forced to spike as Poisson processes at
        the given rates (Hz).  extra_mv: optional dense (n,) membrane injection.
        """
        c = self.cfg
        delayed = self.ring[self.pos % self.delay_steps]
        drive = self._drive_from(delayed)
        if drive is not None:
            self.j_syn += drive
        self.j_syn *= self.decay_s

        inflow = self.j_syn - self.adapt_kick * self.adapt if c.adapt_mv else self.j_syn.copy()
        if extra_mv is not None:
            inflow += extra_mv
        refractory = self.refrac > 0
        inflow[refractory] = 0.0
        self.u = self.decay_v * self.u + inflow

        fired_mask = self.u >= self.u_thresh
        if drive_idx is not None and drive_rate_hz is not None and drive_idx.size:
            p = np.clip(drive_rate_hz * self.dt_s, 0.0, 1.0)
            kicks = self.rng.random(drive_idx.size) < p
            forced = drive_idx[kicks]
            fired_mask[forced] = True
            fired_mask[drive_idx[~kicks]] = False   # driven cells only spike when told to
        fired = np.flatnonzero(fired_mask).astype(np.int64)

        self.u[fired] = self.u_reset
        if c.adapt_mv:
            self.adapt *= self.decay_a
            self.adapt[fired] += 1.0
        self.refrac[self.refrac > 0] -= 1
        self.refrac[fired] = self.ref_steps
        if c.std_u:
            # resource recovers toward 1, and each spike consumes a fraction U of what is left
            self.std_x += (1.0 - self.std_x) * (c.dt_ms / c.std_tau_rec_ms)
            self.std_x[fired] *= (1.0 - c.std_u)

        self.ring[self.pos % self.delay_steps] = fired
        self.pos = (self.pos + 1) % self.delay_steps
        self.t_steps += 1
        self.last_fired = fired
        return fired

    def run(self, steps: int, drive_idx=None, drive_rate_hz=None,
            record_sets: Optional[Dict[str, np.ndarray]] = None, progress: bool = False):
        """Run `steps` steps with constant drive; return dict name -> (steps,) spike counts
        plus 'all' -> total spikes per step."""
        counts = {k: np.zeros(steps, np.int32) for k in (record_sets or {})}
        counts["all"] = np.zeros(steps, np.int32)
        masks = {k: np.zeros(self.n, bool) for k in (record_sets or {})}
        for k, idx in (record_sets or {}).items():
            masks[k][idx] = True
        it = range(steps)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, desc="sim", mininterval=2.0)
        for s in it:
            fired = self.step(drive_idx, drive_rate_hz)
            counts["all"][s] = fired.size
            for k, m in masks.items():
                counts[k][s] = int(m[fired].sum())
        return counts


class SpikeRecorder:
    """Accumulates per-population spike counts per control step and packed
    whole-brain spike bits per control frame (for the PiP render)."""

    def __init__(self, n: int, pops: Dict[str, np.ndarray]):
        self.n = n
        self.pops = pops
        self.masks = {}
        for k, idx in pops.items():
            m = np.zeros(n, bool)
            m[idx] = True
            self.masks[k] = m
        self.frames: List[np.ndarray] = []      # packed bits, one per control step
        self.pop_counts: Dict[str, List[int]] = {k: [] for k in pops}
        self.total: List[int] = []
        self._acc = np.zeros(n, bool)
        self._pop_acc = {k: 0 for k in pops}
        self._tot = 0

    def add_step(self, fired: np.ndarray) -> None:
        self._acc[fired] = True
        self._tot += fired.size
        for k, m in self.masks.items():
            self._pop_acc[k] += int(m[fired].sum())

    def end_frame(self) -> None:
        self.frames.append(np.packbits(self._acc))
        self.total.append(self._tot)
        for k in self.pops:
            self.pop_counts[k].append(self._pop_acc[k])
            self._pop_acc[k] = 0
        self._acc[:] = False
        self._tot = 0

    def save(self, path: str, extra: Optional[Dict] = None) -> None:
        arrs = {"frames_packed": np.stack(self.frames) if self.frames else np.zeros((0, (self.n + 7) // 8), np.uint8),
                "n": np.int64(self.n), "total": np.asarray(self.total, np.int32)}
        for k, v in self.pop_counts.items():
            arrs["pop_" + k] = np.asarray(v, np.int32)
        for k, v in (extra or {}).items():
            arrs[k] = v
        np.savez_compressed(path, **arrs)  # type: ignore[arg-type]


def rates_hz(counts: np.ndarray, n_cells: int, dt_s: float) -> np.ndarray:
    """Convert per-step spike counts of a population to mean rate per cell (Hz)."""
    return counts / max(n_cells, 1) / dt_s


if __name__ == "__main__":  # quick smoke benchmark
    import time, sys
    conn = Connectome.load()
    print(f"N={conn.n:,}  nnz={conn.W.nnz:,}")
    brain = Brain(conn, LIFConfig(weight_scale=float(sys.argv[1]) if len(sys.argv) > 1 else 1.0))
    idx = conn.idx("eye_left_L1")
    t0 = time.time()
    c = brain.run(250, idx, np.full(idx.size, 50.0), progress=False)
    dt = (time.time() - t0) / 250
    print(f"{dt*1000:.1f} ms/step; mean spikes/step {c['all'].mean():.0f}; "
          f"pop rate {c['all'].mean()/conn.n/brain.dt_s:.2f} Hz")

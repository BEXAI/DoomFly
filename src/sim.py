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

Mushroom-body plasticity (docs/ARENA_DESIGN.md, "Dopamine / mushroom-body plasticity"):
`Brain.enable_plasticity` turns on dopamine-gated depression of the KC->MBON synapses
(flybrain conditioning4 / Hige et al. 2015): each KC keeps an eligibility trace, and when a
dopaminergic neuron spikes, every KC->MBON weight is multiplied by
(1 - eta * g_m * e_kc * dt) where g_m is the DAN input the MBON receives *in the wiring*.
Weights are bounded below at 0.2 * w0 and never recover (v1).  Off by default: with
plasticity disabled the model is bit-for-bit the one described above.  Note that the rule
fires on ANY dopaminergic spike (spontaneous or visually evoked as well as reward-driven), so
w drifts down slowly even without reward events; see docs/ARENA_DESIGN.md.

Implementation notes (all bit-identical to the straightforward numpy formulation):
  * the dense state passes in `step` write into preallocated float32 buffers (`out=`), and
    the set of refractory neurons is kept as the fired sets of the last `ref_steps` steps
    (exactly the neurons with refrac > 0, also when a forced cell re-fires while refractory)
    instead of a full-array compare + mask per step;
  * the column gather `W[:, fired] @ x` is done with the same scipy kernels
    (csr_row_index, csc_matvec, float32, same accumulation order) without building a
    csc_matrix object per step;
  * the forced-spike probabilities `clip(rate * dt)` are cached while the caller passes the
    same rate array object (callers must not mutate that array in place between steps).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union

import numpy as np
import scipy.sparse as sp
from collections import deque
from scipy.sparse import _sparsetools as _spt

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


@dataclass
class PlasticityState:
    """State of the KC->MBON dopamine-gated depression rule (see Brain.enable_plasticity).

    Edges are addressed by their flat position in `Brain.W.data` (CSC), so the update is a
    fancy-indexed read-modify-write on the live matrix and takes effect on the next matvec.
    """
    eta: float
    tau_trace_s: float
    decay_trace: float
    kc_idx: np.ndarray        # (n_kc,) neuron indices of the KC set
    mbon_idx: np.ndarray      # (n_mbon,)
    dan_idx: np.ndarray       # (n_dan,)
    kc_rank: np.ndarray       # (n,) rank of each neuron in kc_idx, -1 if not a KC
    dan_rank: np.ndarray      # (n,) rank in dan_idx, -1 if not a DAN
    pos: np.ndarray           # (n_edges,) flat positions into Brain.W.data
    w0: np.ndarray            # (n_edges,) initial signed weights (float32, already * syn_kick)
    w0_floor: np.ndarray      # (n_edges,) 0.2 * |w0|
    w0_sign: np.ndarray       # (n_edges,) sign(w0)
    kc_of_edge: np.ndarray    # (n_edges,) rank into kc_idx of the presynaptic KC
    m_of_edge: np.ndarray     # (n_edges,) rank into mbon_idx of the postsynaptic MBON
    G: np.ndarray             # (n_mbon, n_dan) |W0[m, d]| / mean_m(sum_d |W0[m, d]|)
    trace: np.ndarray         # (n_kc,) eligibility trace e_kc
    trace_active: bool = False
    trace_peak: float = 0.0   # upper bound of trace.max(), used to skip idle steps
    n_mbon_with_dan: int = 0
    n_dopamine_steps: int = 0  # steps in which a DAN spike met a non-zero KC trace

    def to_json(self) -> Dict:
        return dict(eta=self.eta, tau_trace_s=self.tau_trace_s, n_edges=int(self.pos.size),
                    n_kc=int(self.kc_idx.size), n_mbon=int(self.mbon_idx.size),
                    n_dan=int(self.dan_idx.size), n_mbon_with_dan=self.n_mbon_with_dan,
                    w_floor_frac=0.2, recovery="none (v1)")


class Brain:
    """Single-instance LIF network over a Connectome, Euler-integrated at dt."""

    def __init__(self, conn: Connectome, cfg: Optional[LIFConfig] = None,
                 seed: Union[int, np.random.SeedSequence] = 0):
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
        # KC->MBON plasticity (off unless enable_plasticity is called) and the one-shot
        # dopamine drive registered by dopamine_drive() for the next step.
        self._plast: Optional[PlasticityState] = None
        self._dopa_idx: Optional[np.ndarray] = None
        self._dopa_rate: Optional[np.ndarray] = None
        # preallocated dense work buffers (never exposed) and the forced-probability cache
        self._inflow = np.empty(self.n, np.float32)
        self._tmp = np.empty(self.n, np.float32)
        self._mask = np.empty(self.n, bool)
        self._p_src: Optional[np.ndarray] = None
        self._p: Optional[np.ndarray] = None
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
        # neurons with refrac > 0 at the start of a step == union of the fired sets of the
        # last ref_steps steps (refrac is set to ref_steps at spike time, -1 per step).
        self._ref_hist: deque = deque([np.zeros(0, np.int64)] * self.ref_steps, maxlen=self.ref_steps)
        self._dopa_idx = self._dopa_rate = None
        self._p_src = self._p = None
        if self._plast is not None:       # traces are dynamic state; learned weights persist
            self._plast.trace[:] = 0.0
            self._plast.trace_active = False
            self._plast.trace_peak = 0.0

    # ------------------------------------------------------------------ core
    def _drive_from(self, fired: np.ndarray) -> Optional[np.ndarray]:
        """Sum of the W columns of the neurons that fired (sparse column gather).

        Does exactly what ``self.W[:, fired] @ x`` (x = std_x[fired], or ones without STD)
        does inside scipy -- csr_row_index gather of the CSC columns, then csc_matvec in
        float32 in the same order -- without building a csc_matrix object and its validation
        (~0.1 ms of Python per call)."""
        m = fired.size
        if m == 0:
            return None
        W = self.W
        indptr, indices, data = W.indptr, W.indices, W.data
        idx_dtype = indptr.dtype
        cols = fired.astype(idx_dtype, copy=False)
        sub_indptr = np.zeros(m + 1, dtype=idx_dtype)
        np.cumsum(indptr[cols + 1] - indptr[cols], out=sub_indptr[1:])
        nnz = int(sub_indptr[-1])
        sub_indices = np.empty(nnz, dtype=idx_dtype)
        sub_data = np.empty(nnz, dtype=data.dtype)
        _spt.csr_row_index(m, cols, indptr, indices, data, sub_indices, sub_data)
        out = np.zeros(self.n, np.float32)
        if self.cfg.std_u:
            x = self.std_x[fired]
        else:
            x = np.ones(m, np.float32)
        _spt.csc_matvec(self.n, m, sub_indptr, sub_indices, sub_data, x, out)
        return out

    def step(self, drive_idx: Optional[np.ndarray] = None, drive_rate_hz: Optional[np.ndarray] = None,
             extra_mv: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance one dt. Returns the int64 indices of neurons that spiked.

        drive_idx / drive_rate_hz: neurons forced to spike as Poisson processes at
        the given rates (Hz).  extra_mv: optional dense (n,) membrane injection.
        """
        c = self.cfg
        if self._dopa_idx is not None:     # one-shot: registered for this step only
            drive_idx, drive_rate_hz = self._merge_drive(drive_idx, drive_rate_hz)
            self._dopa_idx = self._dopa_rate = None
        delayed = self.ring[self.pos % self.delay_steps]
        drive = self._drive_from(delayed)
        if drive is not None:
            self.j_syn += drive
        self.j_syn *= self.decay_s

        inflow = self._inflow
        if c.adapt_mv:
            tmp = self._tmp
            np.multiply(self.adapt, self.adapt_kick, out=tmp)     # == adapt_kick * adapt (float32)
            np.subtract(self.j_syn, tmp, out=inflow)
        else:
            inflow[:] = self.j_syn
        if extra_mv is not None:
            inflow += extra_mv
        ref_hist = self._ref_hist
        ref_idx = ref_hist[0] if self.ref_steps == 1 else np.concatenate(ref_hist)
        if ref_idx.size:
            inflow[ref_idx] = 0.0
        u = self.u
        np.multiply(u, self.decay_v, out=u)                          # == decay_v * u
        np.add(u, inflow, out=u)

        fired_mask = self._mask
        np.greater_equal(u, self.u_thresh, out=fired_mask)
        if drive_idx is not None and drive_rate_hz is not None and drive_idx.size:
            p = self._p
            if p is None or drive_rate_hz is not self._p_src:         # same array object -> same p
                p = np.clip(drive_rate_hz * self.dt_s, 0.0, 1.0)
                self._p, self._p_src = p, drive_rate_hz
            kicks = self.rng.random(drive_idx.size) < p
            forced = drive_idx[kicks]
            fired_mask[forced] = True
            fired_mask[drive_idx[~kicks]] = False   # driven cells only spike when told to
        fired = np.flatnonzero(fired_mask).astype(np.int64)

        u[fired] = self.u_reset
        if c.adapt_mv:
            self.adapt *= self.decay_a
            self.adapt[fired] += 1.0
        if ref_idx.size:
            self.refrac[ref_idx] -= 1
        self.refrac[fired] = self.ref_steps
        ref_hist.append(fired)
        if c.std_u:
            tmp = self._tmp
            np.subtract(1.0, self.std_x, out=tmp)
            np.multiply(tmp, (c.dt_ms / c.std_tau_rec_ms), out=tmp)
            np.add(self.std_x, tmp, out=self.std_x)
            self.std_x[fired] *= (1.0 - c.std_u)

        if self._plast is not None:
            self._plasticity_step(fired)

        self.ring[self.pos % self.delay_steps] = fired
        self.pos = (self.pos + 1) % self.delay_steps
        self.t_steps += 1
        self.last_fired = fired
        return fired

    # ------------------------------------------------------------------ plasticity
    def enable_plasticity(self, kc_idx, mbon_idx, dan_idx, eta: float = 0.05,
                          tau_trace_s: float = 1.0) -> int:
        """Turn on dopamine-gated depression of the KC->MBON synapses.

        Finds every existing edge W[m, k] with k in kc_idx and m in mbon_idx (postsynaptic
        MBON row, presynaptic KC column) and caches its flat position in `self.W.data`
        together with w0.  Each KC gets an eligibility trace e_kc (decays with tau_trace_s,
        += 1 per KC spike).  The per-MBON dopamine gain is read off the wiring:
        G[m, d] = |W0[m, d]| for d in dan_idx, normalised so that the mean over MBONs of the
        total DAN input sum_d G[m, d] is 1 (scale-free w.r.t. syn_kick / weight_scale).
        An MBON with no DAN input simply never depresses.

        Rule, applied every step in which some DAN fires (see _plasticity_step):
            g_m = sum_{fired d} G[m, d]
            w  <- w * (1 - eta * g_m * e_kc * dt_s),  bounded so |w| >= 0.2 |w0|, sign kept.
        No recovery in v1.  Returns the number of edges found.  Calling it again re-reads the
        current weights as the new w0 (i.e. resets the reference, not the weights).
        """
        kc = np.unique(np.asarray(kc_idx, dtype=np.int64))
        mb = np.unique(np.asarray(mbon_idx, dtype=np.int64))
        dan = np.unique(np.asarray(dan_idx, dtype=np.int64))
        if not isinstance(self.W, sp.csc_matrix):
            self.W = sp.csc_matrix(self.W)
        W = self.W
        n = self.n
        kc_rank = np.full(n, -1, np.int64)
        kc_rank[kc] = np.arange(kc.size)
        mbon_rank = np.full(n, -1, np.int64)
        mbon_rank[mb] = np.arange(mb.size)
        dan_rank = np.full(n, -1, np.int64)
        dan_rank[dan] = np.arange(dan.size)

        # Flat positions of every stored entry in the KC columns of the CSC matrix.
        starts = W.indptr[kc].astype(np.int64)
        lens = (W.indptr[kc + 1] - W.indptr[kc]).astype(np.int64)
        total = int(lens.sum())
        col_rank = np.repeat(np.arange(kc.size), lens)
        within = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
        pos_all = np.repeat(starts, lens) + within
        rows = W.indices[pos_all].astype(np.int64)
        sel = (mbon_rank[rows] >= 0) & (W.data[pos_all] != 0)
        pos = pos_all[sel]
        w0 = W.data[pos].astype(np.float32, copy=True)

        G = np.abs(np.asarray(W[mb][:, dan].todense(), dtype=np.float64))
        tot = G.sum(axis=1)
        mean_tot = float(tot.mean()) if tot.size else 0.0
        if mean_tot > 0:
            G /= mean_tot
        dt_s = self.dt_s
        self._plast = PlasticityState(
            eta=float(eta), tau_trace_s=float(tau_trace_s),
            decay_trace=float(np.exp(-dt_s / tau_trace_s)),
            kc_idx=kc, mbon_idx=mb, dan_idx=dan, kc_rank=kc_rank, dan_rank=dan_rank,
            pos=pos, w0=w0, w0_floor=(0.2 * np.abs(w0)).astype(np.float32),
            w0_sign=np.sign(w0).astype(np.float32),
            kc_of_edge=col_rank[sel], m_of_edge=mbon_rank[rows[sel]],
            G=G.astype(np.float32), trace=np.zeros(kc.size, np.float32),
            n_mbon_with_dan=int((tot > 0).sum()))
        return int(pos.size)

    def disable_plasticity(self, restore_weights: bool = False) -> None:
        """Stop updating; optionally put w0 back into the matrix."""
        p = self._plast
        if p is not None and restore_weights:
            self.W.data[p.pos] = p.w0
        self._plast = None

    @property
    def plasticity(self) -> Optional[PlasticityState]:
        return self._plast

    def _plasticity_step(self, fired: np.ndarray) -> None:
        """Trace update + dopamine-gated depression for one step (called after spikes)."""
        p = self._plast
        assert p is not None
        if fired.size:
            kr = p.kc_rank[fired]
            fk = kr[kr >= 0]
            dr = p.dan_rank[fired]
            fd = dr[dr >= 0]
        else:
            fk = fd = fired
        if p.trace_active:
            p.trace *= p.decay_trace
            p.trace_peak *= p.decay_trace
        if fk.size:
            p.trace[fk] += 1.0
            p.trace_active = True
            p.trace_peak = float(p.trace.max())
        elif p.trace_active and p.trace_peak < 1e-3:
            p.trace[:] = 0.0            # every trace is below 1e-3: go idle
            p.trace_active = False
            p.trace_peak = 0.0
        if fd.size and p.trace_active:
            g = p.G[:, fd].sum(axis=1) if fd.size > 1 else p.G[:, fd[0]]
            factor = 1.0 - np.float32(p.eta * self.dt_s) * g[p.m_of_edge] * p.trace[p.kc_of_edge]
            np.maximum(factor, 0.0, out=factor)
            data = self.W.data
            w = np.abs(data[p.pos] * factor)
            np.maximum(w, p.w0_floor, out=w)
            data[p.pos] = w * p.w0_sign
            p.n_dopamine_steps += 1

    def dopamine_drive(self, dan_idx, rate_hz) -> None:
        """Register an extra forced-Poisson drive (same mechanism as the sensory drive) for the
        NEXT call of step(), after which it is cleared.  Call it every step of a reward window.
        `rate_hz` may be a scalar or an array matching dan_idx.  If a cell is also in the
        sensory drive of that step the two entries are merged (max rate), never doubled."""
        idx = np.asarray(dan_idx, dtype=np.int64).ravel()
        rate = np.broadcast_to(np.asarray(rate_hz, dtype=np.float64), idx.shape).copy()
        if self._dopa_idx is None:
            self._dopa_idx, self._dopa_rate = idx, rate
        else:   # several calls in one step accumulate into one (deduplicated) set
            self._dopa_idx = np.concatenate([self._dopa_idx, idx])
            self._dopa_rate = np.concatenate([np.asarray(self._dopa_rate, np.float64), rate])

    def _merge_drive(self, drive_idx: Optional[np.ndarray], drive_rate_hz: Optional[np.ndarray]):
        """Union of the sensory drive and the registered dopamine drive, one entry per cell.
        Duplicates take the max rate so a forced cell is never counted twice (the forced
        mechanism in step() is order-dependent for repeated indices)."""
        d_idx, d_rate = self._dopa_idx, self._dopa_rate
        assert d_idx is not None and d_rate is not None
        if drive_idx is None or drive_rate_hz is None or drive_idx.size == 0:
            all_idx, all_rate = d_idx, d_rate
        else:
            all_idx = np.concatenate([np.asarray(drive_idx, np.int64), d_idx])
            all_rate = np.concatenate([np.asarray(drive_rate_hz, np.float64), d_rate])
        uniq, inv = np.unique(all_idx, return_inverse=True)
        if uniq.size == all_idx.size:
            return all_idx, all_rate
        rate_u = np.zeros(uniq.size, np.float64)
        np.maximum.at(rate_u, inv, all_rate)
        return uniq, rate_u

    def plasticity_stats(self) -> Dict:
        """Diagnostics of the KC->MBON weights relative to w0 (mean 1.0 / zeros when off)."""
        p = self._plast
        if p is None:
            return dict(mean_w_over_w0=1.0, min_w_over_w0=1.0, frac_edges_changed=0.0,
                        n_edges=0, n_kc_active_trace=0)
        ratio = self.W.data[p.pos] / p.w0
        return dict(mean_w_over_w0=float(ratio.mean()) if ratio.size else 1.0,
                    min_w_over_w0=float(ratio.min()) if ratio.size else 1.0,
                    frac_edges_changed=float((np.abs(ratio - 1.0) > 1e-3).mean()) if ratio.size else 0.0,
                    n_edges=int(p.pos.size),
                    n_kc_active_trace=int((p.trace > 0.1).sum()))

    def plasticity_edges(self) -> Dict[str, np.ndarray]:
        """Per-edge view: neuron indices of the KC and MBON of each cached edge and w/w0."""
        p = self._plast
        if p is None:
            e = np.zeros(0, np.int64)
            return dict(kc=e, mbon=e, w_over_w0=np.zeros(0, np.float32))
        return dict(kc=p.kc_idx[p.kc_of_edge], mbon=p.mbon_idx[p.m_of_edge],
                    w_over_w0=self.W.data[p.pos] / p.w0)

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
        self._fr: List[np.ndarray] = []      # fired arrays of the substeps of the open frame

    def add_step(self, fired: np.ndarray) -> None:
        """Record the spikes of one LIF substep (kept as index arrays until end_frame)."""
        self._fr.append(fired)

    def end_frame(self) -> None:
        """Close the control frame: population spike counts (every spike counts, a cell may
        fire several times per frame) and the packed 'fired at least once' bits."""
        fr = self._fr
        cat = fr[0] if len(fr) == 1 else (np.concatenate(fr) if fr else np.zeros(0, np.int64))
        self._fr = []
        self._acc[cat] = True
        self.frames.append(np.packbits(self._acc))
        self.total.append(int(cat.size))
        for k, m in self.masks.items():
            self.pop_counts[k].append(int(m[cat].sum()))
        self._acc[:] = False

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

#!/usr/bin/env python3
"""Retina -> lamina encoder: turns a feed's luminance grid into Poisson rates
for the lamina monopolar cells L1 (ON), L2 (OFF) and optionally L3 (sustained
luminance) of one compound eye, using the real optic-lobe column coordinates.

Geometry
--------
``data/graph/eye_map.npz`` (from ``src/build_graph.py``) stores, per side S in
{L, R} and type T in {L1, L2, L3}, ``f"{S}_{T}_bodies"`` (bodyIds) and
``f"{S}_{T}_hex"`` = (assignedOlHex1, assignedOlHex2), the axial hex coordinates
of the ~880 ommatidial columns (hex1 in 1..36, hex2 in 1..39).

Axial -> cartesian.  The two hex axes are 120 deg apart in this dataset::

    x = hex1 - hex2 / 2,      y = hex2 * sqrt(3) / 2

(With ``x = hex1 + hex2/2`` the eye comes out as a 3.3:1 sheared bar; with the
minus convention it is a 1.15:1 upright-ish ellipse, i.e. an eye.)  The point
cloud (union of all L1/L2/L3 columns of that eye) is then PCA-rotated so its
major axis is vertical, and min/max-normalised to the unit square.

Orientation assumption (the one thing that cannot be read off the data):
**the eye's long (dorso-ventral) axis is the feed's vertical axis, and
increasing hex2 (after the upright rotation) points to the TOP of the screen.**
``flip=True`` reverses the vertical direction; ``mirror=True`` reverses the
horizontal one.  Both eyes use the identical convention, so a feature at the
same hex coordinate on both eyes lands on the same screen cell of its panel.

Each column is assigned the luminance-grid cell containing its normalised
position (a uniform ``rows x cols`` partition of the unit square).  The eye is
an ellipse so the corners of the rectangular grid stay unpopulated.

Coding
------
A slowly adapting per-cell background ``bg`` (exponential, ``tau_adapt_s``)
tracks the luminance.  Temporal contrast ``c = (lum - bg) / (bg + 0.05)``;

    L1 rate = clip( gain * c, 0, 1) * rate_max      (ON  pathway)
    L2 rate = clip(-gain * c, 0, 1) * rate_max      (OFF pathway)
    L3 rate = l3_gain * lum * rate_max              (sustained luminance)

plus ``tonic_hz`` on every cell.  ``encode()`` returns simulator row indices and
rates ready for ``Brain.step(drive_idx=..., drive_rate_hz=...)``.

Run ``python -m src.encoder`` for a self-test that does not load the brain.
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_DIR = os.path.join(ROOT, "data", "graph")
EYE_MAP_PATH = os.path.join(GRAPH_DIR, "eye_map.npz")
IDS_PATH = os.path.join(GRAPH_DIR, "ids.npy")

TYPES = ("L1", "L2", "L3")
BG_EPS = 0.05  # Weber-contrast floor so dark backgrounds do not explode


def hex_to_xy(hex_coords: np.ndarray) -> np.ndarray:
    """Axial (hex1, hex2) -> cartesian (x, y) with axes 120 deg apart (see module doc)."""
    h = np.asarray(hex_coords, dtype=np.float64).reshape(-1, 2)
    x = h[:, 0] - 0.5 * h[:, 1]
    y = h[:, 1] * (math.sqrt(3.0) / 2.0)
    return np.stack([x, y], axis=1)


class EyeFrame:
    """Deterministic map from hex coordinates to the unit square for one eye.

    Fitted on all columns of the eye: PCA rotation that makes the major axis
    vertical (choosing the rotation of smallest magnitude, so hex2 stays 'up'),
    then per-axis min/max normalisation to [0, 1].
    """

    def __init__(self, all_hex: np.ndarray):
        xy = hex_to_xy(all_hex)
        self.center = xy.mean(axis=0)
        p0 = xy - self.center
        _, _, vt = np.linalg.svd(p0, full_matrices=False)
        major = vt[0]
        if major[1] < 0:            # fix the sign so the major axis points +y
            major = -major
        theta = math.atan2(major[1], major[0])
        rot = math.pi / 2.0 - theta  # rotate major axis onto +y
        c, s = math.cos(rot), math.sin(rot)
        self.R = np.array([[c, -s], [s, c]])
        q = p0 @ self.R.T
        self.lo = q.min(axis=0)
        self.hi = q.max(axis=0)
        self.rotation_deg = math.degrees(rot)
        sv = np.linalg.svd(q, compute_uv=False)
        self.aspect = float(sv[0] / max(sv[1], 1e-9))

    def unit(self, hex_coords: np.ndarray) -> np.ndarray:
        """(n, 2) positions in [0, 1]^2: column 0 = horizontal u, column 1 = vertical v (up = 1)."""
        q = (hex_to_xy(hex_coords) - self.center) @ self.R.T
        return np.clip((q - self.lo) / np.maximum(self.hi - self.lo, 1e-9), 0.0, 1.0)


class EyeEncoder:
    """Luminance grid -> L1 / L2 / L3 Poisson rates for one eye.

    Args:
        side: ``'L'`` or ``'R'``.
        eye_map_path, ids_path: graph files (defaults under ``data/graph``).
        grid: ``(rows, cols)`` of the luminance grid the feed will supply.
        rate_max_hz: rate at saturating contrast.
        tau_adapt_s: time constant of the background (adaptation) filter.
        gain: contrast gain; ``c = 1/gain`` saturates.
        tonic_hz: baseline added to every driven cell.
        l3_gain: L3 rate = ``l3_gain * lum * rate_max`` (0 disables L3 drive).
        flip: reverse the vertical mapping (hex2-up becomes screen-bottom).
        mirror: reverse the horizontal mapping.
        bg_init: ``'frame'`` (background starts equal to the first frame, no
            start-up burst) or ``'zero'`` (starts dark: strong initial ON burst
            that decays with ``tau_adapt_s``).
    """

    def __init__(
        self,
        side: str,
        eye_map_path: str = EYE_MAP_PATH,
        ids_path: str = IDS_PATH,
        grid: Tuple[int, int] = (24, 18),
        rate_max_hz: float = 150.0,
        tau_adapt_s: float = 0.25,
        gain: float = 6.0,
        tonic_hz: float = 0.0,
        l3_gain: float = 0.3 * 0.3,
        flip: bool = False,
        mirror: bool = False,
        bg_init: str = "frame",
    ) -> None:
        if side not in ("L", "R"):
            raise ValueError("side must be 'L' or 'R'")
        if bg_init not in ("frame", "zero"):
            raise ValueError("bg_init must be 'frame' or 'zero'")
        self.side = side
        self.rows, self.cols = int(grid[0]), int(grid[1])
        self.n_cells = self.rows * self.cols
        self.rate_max_hz = float(rate_max_hz)
        self.tau_adapt_s = float(tau_adapt_s)
        self.gain = float(gain)
        self.tonic_hz = float(tonic_hz)
        self.l3_gain = float(l3_gain)
        self.flip = bool(flip)
        self.mirror = bool(mirror)
        self.bg_init = bg_init

        z = np.load(eye_map_path)
        ids = np.load(ids_path).astype(np.int64)
        body_to_row: Dict[int, int] = {int(b): i for i, b in enumerate(ids)}

        raw: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for t in TYPES:
            kb, kh = f"{side}_{t}_bodies", f"{side}_{t}_hex"
            bodies = z[kb].astype(np.int64) if kb in z.files else np.zeros(0, np.int64)
            hexes = z[kh].astype(np.float32).reshape(-1, 2) if kh in z.files else np.zeros((0, 2), np.float32)
            keep = np.array([int(b) in body_to_row for b in bodies], dtype=bool)
            raw[t] = (bodies[keep], hexes[keep])
        all_hex = np.concatenate([raw[t][1] for t in TYPES], axis=0)
        if all_hex.shape[0] < 3:
            raise RuntimeError(f"eye {side}: no lamina columns found in {eye_map_path}")
        self.frame = EyeFrame(np.unique(all_hex, axis=0))

        self.bodies: Dict[str, np.ndarray] = {}
        self.idx: Dict[str, np.ndarray] = {}
        self.uv: Dict[str, np.ndarray] = {}
        self.cell_of: Dict[str, np.ndarray] = {}
        for t in TYPES:
            bodies, hexes = raw[t]
            self.bodies[t] = bodies
            self.idx[t] = np.asarray([body_to_row[int(b)] for b in bodies], dtype=np.int64)
            uv = self.frame.unit(hexes) if len(hexes) else np.zeros((0, 2))
            self.uv[t] = uv
            self.cell_of[t] = self._cells_from_uv(uv)

        # Public aliases required by the sim.
        self.idx_L1, self.idx_L2, self.idx_L3 = self.idx["L1"], self.idx["L2"], self.idx["L3"]
        self.cell_of_L1, self.cell_of_L2, self.cell_of_L3 = self.cell_of["L1"], self.cell_of["L2"], self.cell_of["L3"]
        self.has_L3 = self.idx_L3.size > 0 and self.l3_gain > 0.0
        self._out_idx = np.concatenate([self.idx_L1, self.idx_L2] + ([self.idx_L3] if self.has_L3 else []))

        # Adaptation state.
        self.bg: Optional[np.ndarray] = None if bg_init == "frame" else np.zeros(self.n_cells, np.float32)
        self._prev_me: Optional[np.ndarray] = None
        self.last_contrast = np.zeros(self.n_cells, np.float32)

    # ------------------------------------------------------------------ geometry
    def _cells_from_uv(self, uv: np.ndarray) -> np.ndarray:
        """Flattened grid cell (row * cols + col) containing each unit-square position."""
        if uv.shape[0] == 0:
            return np.zeros(0, np.int64)
        u = uv[:, 0]
        v = uv[:, 1]
        if self.mirror:
            u = 1.0 - u
        v_top = v if self.flip else 1.0 - v          # default: v = 1 (hex2 up) -> row 0 (top)
        col = np.clip(np.floor(u * self.cols).astype(np.int64), 0, self.cols - 1)
        row = np.clip(np.floor(v_top * self.rows).astype(np.int64), 0, self.rows - 1)
        return row * self.cols + col

    def occupancy(self, t: str = "L1") -> np.ndarray:
        """(rows, cols) count of type-``t`` columns per grid cell."""
        return np.bincount(self.cell_of[t], minlength=self.n_cells).reshape(self.rows, self.cols)

    def n_populated(self, t: str = "L1") -> int:
        return int(np.count_nonzero(self.occupancy(t)))

    # ------------------------------------------------------------------ coding
    def reset(self) -> None:
        self.bg = None if self.bg_init == "frame" else np.zeros(self.n_cells, np.float32)
        self._prev_me = None
        self.last_contrast[:] = 0.0

    def _check(self, lum_grid: np.ndarray) -> np.ndarray:
        lum = np.asarray(lum_grid, dtype=np.float32)
        if lum.shape != (self.rows, self.cols):
            raise ValueError(f"luminance grid shape {lum.shape} != {(self.rows, self.cols)}")
        return np.clip(lum.reshape(-1), 0.0, 1.0)

    def encode(self, lum_grid: np.ndarray, dt_s: float) -> Dict[str, Any]:
        """One frame -> dict(idx, rate_hz, on, off, motion).

        ``idx`` (int64, k) and ``rate_hz`` (float32, k) concatenate L1, L2 (and
        L3 when enabled), aligned with ``idx_L1``, ``idx_L2``, ``idx_L3``.
        ``on`` / ``off`` are the mean L1 / L2 rates (Hz), ``motion`` = mean |c|.
        """
        lum = self._check(lum_grid)
        if self.bg is None:
            self.bg = lum.copy()
        c = (lum - self.bg) / (self.bg + BG_EPS)
        # Update the background AFTER computing contrast (contrast is relative to the past).
        alpha = 1.0 - math.exp(-max(float(dt_s), 0.0) / self.tau_adapt_s)
        self.bg += alpha * (lum - self.bg)
        self.last_contrast = c

        on_cell = np.clip(self.gain * c, 0.0, 1.0) * self.rate_max_hz
        off_cell = np.clip(-self.gain * c, 0.0, 1.0) * self.rate_max_hz
        r1 = on_cell[self.cell_of_L1] + self.tonic_hz
        r2 = off_cell[self.cell_of_L2] + self.tonic_hz
        parts = [r1, r2]
        if self.has_L3:
            l3_cell = self.l3_gain * lum * self.rate_max_hz
            parts.append(l3_cell[self.cell_of_L3] + self.tonic_hz)
        rate = np.concatenate(parts).astype(np.float32, copy=False)
        return dict(
            idx=self._out_idx,
            rate_hz=rate,
            on=float(r1.mean()) if r1.size else 0.0,
            off=float(r2.mean()) if r2.size else 0.0,
            motion=float(np.abs(c).mean()),
        )

    def motion_energy(self, lum_grid: np.ndarray) -> float:
        """Mean |frame - previous frame| over grid cells (own state, independent of encode)."""
        lum = self._check(lum_grid)
        if self._prev_me is None:
            self._prev_me = lum.copy()
            return 0.0
        e = float(np.abs(lum - self._prev_me).mean())
        self._prev_me = lum.copy()
        return e

    # ------------------------------------------------------------------ info
    def summary(self) -> str:
        n = {t: int(self.idx[t].size) for t in TYPES}
        pop = {t: self.n_populated(t) for t in TYPES if n[t]}
        return (
            f"eye {self.side}: L1={n['L1']} L2={n['L2']} L3={n['L3']} cells | grid {self.rows}x{self.cols} "
            f"({self.n_cells} cells), populated {pop} | frame rot {self.frame.rotation_deg:+.1f} deg, "
            f"aspect {self.frame.aspect:.2f} | flip={self.flip} mirror={self.mirror} L3drive={self.has_L3}"
        )


def make_encoders(grid: Tuple[int, int] = (24, 18), **kw) -> Tuple[EyeEncoder, EyeEncoder]:
    """(left, right) encoders sharing all keyword settings."""
    return EyeEncoder("L", grid=grid, **kw), EyeEncoder("R", grid=grid, **kw)


# --------------------------------------------------------------------------- #
# Self-test (no brain)
# --------------------------------------------------------------------------- #
def _self_test(args: argparse.Namespace) -> None:
    import sys

    sys.path.insert(0, ROOT)
    from src.feeds import make_pair

    grid = (args.rows, args.cols)
    encL, encR = make_encoders(grid=grid, flip=args.flip, mirror=args.mirror, bg_init="zero")
    print(encL.summary())
    print(encR.summary())

    fps = args.fps
    n = int(round(args.seconds * fps))
    dt = 1.0 / fps
    feedL, feedR = make_pair()
    swipes_l = {int(round(t * fps)) for t in (0.5, 1.8)}

    ts = np.arange(n) * dt
    rec: Dict[str, np.ndarray] = {k: np.zeros(n, np.float32) for k in ("L_on", "L_off", "L_mot", "L_me", "R_on", "R_off", "R_mot", "R_me")}
    for f in range(n):
        if f in swipes_l:
            feedL.swipe()
        feedL.step(dt)
        feedR.step(dt)
        gL = feedL.luminance_grid(*grid)
        gR = feedR.luminance_grid(*grid)
        oL = encL.encode(gL, dt)
        oR = encR.encode(gR, dt)
        assert oL["idx"].shape == oL["rate_hz"].shape and oL["idx"].dtype == np.int64 and oL["rate_hz"].dtype == np.float32
        assert oR["rate_hz"].min() >= 0.0 and oR["rate_hz"].max() <= encR.rate_max_hz + encR.tonic_hz + 1e-3
        rec["L_on"][f], rec["L_off"][f], rec["L_mot"][f] = oL["on"], oL["off"], oL["motion"]
        rec["R_on"][f], rec["R_off"][f], rec["R_mot"][f] = oR["on"], oR["off"], oR["motion"]
        rec["L_me"][f] = encL.motion_energy(gL)
        rec["R_me"][f] = encR.motion_energy(gR)
        if f % 10 == 0:
            print(
                f"t={ts[f]:4.2f}s  L: on {oL['on']:6.1f} off {oL['off']:6.1f} |c| {oL['motion']:.3f} ME {rec['L_me'][f]:.3f}"
                f"   R: on {oR['on']:6.1f} off {oR['off']:6.1f} |c| {oR['motion']:.3f} ME {rec['R_me'][f]:.3f}"
            )

    # ---- assertions ------------------------------------------------------
    rmax = encR.rate_max_hz
    R_tot = rec["R_on"] + rec["R_off"]
    L_tot = rec["L_on"] + rec["L_off"]
    r_start = float(R_tot[: int(0.1 * fps)].max())
    r_after1 = float(R_tot[ts >= 1.0].mean())
    r_after2 = float(R_tot[ts >= 2.0].mean())
    print(f"R (static): start-up peak {r_start:.1f} Hz, mean after 1 s {r_after1:.2f} Hz, after 2 s {r_after2:.3f} Hz")
    assert r_start > 0.2 * rmax, "expected a start-up ON burst on the right eye (bg_init='zero')"
    assert r_after1 < 0.05 * rmax, f"R did not adapt: mean rate after 1 s = {r_after1:.1f} Hz"
    assert r_after2 < 0.01 * rmax, f"R did not decay to ~0: mean rate after 2 s = {r_after2:.2f} Hz"
    assert float(rec["R_me"][ts >= 0.1].max()) == 0.0, "static right feed should have zero motion energy"

    def win(lo: float, hi: float) -> float:
        return float(L_tot[(ts >= lo) & (ts < hi)].mean())

    quiet = win(1.3, 1.7)
    b1, b2 = win(0.5, 0.9), win(1.8, 2.2)
    print(f"L (swipes): burst1 {b1:.1f} Hz, quiet {quiet:.1f} Hz, burst2 {b2:.1f} Hz")
    assert b1 > 3.0 * max(quiet, 0.5) and b2 > 3.0 * max(quiet, 0.5), "left eye should burst on swipes"
    assert float(rec["L_me"][(ts >= 0.5) & (ts < 0.9)].max()) > 0.0, "left swipe should produce motion energy"

    # ---- plot -------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_ON, C_OFF, INK, MUTED, GRID_C = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1.35, 1.0]))
    fig.patch.set_facecolor("#fcfcfb")
    for ax, side in zip(axes[:2], ("L", "R")):
        ax.set_facecolor("#fcfcfb")
        ax.plot(ts, rec[f"{side}_on"], color=C_ON, lw=2, label="L1 (ON)")
        ax.plot(ts, rec[f"{side}_off"], color=C_OFF, lw=2, label="L2 (OFF)")
        if side == "L":
            for t_sw in (0.5, 1.8):
                ax.axvline(t_sw, color=MUTED, lw=1, ls=":")
            ax.text(0.5, encL.rate_max_hz * 0.97, " swipe", color=MUTED, fontsize=8, va="top")
            ax.text(1.8, encL.rate_max_hz * 0.97, " swipe", color=MUTED, fontsize=8, va="top")
        ax.set_title(f"{'Left' if side == 'L' else 'Right'} eye - mean lamina rate (Hz)", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("time (s)", color=MUTED)
        ax.set_ylim(0, encL.rate_max_hz * 1.02)
        ax.grid(axis="y", color=GRID_C, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED)
        ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax = axes[2]
    ax.set_facecolor("#fcfcfb")
    uv = encL.uv["L1"]
    cells = encL.cell_of_L1
    row, col = cells // encL.cols, cells % encL.cols
    # Colour by grid cell: hue from column, lightness from row (a 2D sequential-ish map).
    hsv = np.stack([col / max(encL.cols - 1, 1), np.full(len(col), 0.75), 0.35 + 0.6 * (1 - row / max(encL.rows - 1, 1))], 1)
    ax.scatter(uv[:, 0], uv[:, 1], c=matplotlib.colors.hsv_to_rgb(hsv), s=16, edgecolors="none")
    for k in range(encL.cols + 1):
        ax.axvline(k / encL.cols, color=GRID_C, lw=0.6)
    for k in range(encL.rows + 1):
        ax.axhline(k / encL.rows, color=GRID_C, lw=0.6)
    ax.set_aspect("equal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Left eye L1 columns ({len(uv)}) on {encL.rows}x{encL.cols} grid", color=INK, fontsize=11, loc="left")
    ax.set_xlabel("screen x (0 = left)", color=MUTED)
    ax.set_ylabel("hex2-up (1 = screen top unless --flip)", color=MUTED)
    ax.tick_params(colors=MUTED)
    for sp in ax.spines.values():
        sp.set_color("#c3c2b7")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")
    print("self-test passed")


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--flip", action="store_true", help="reverse vertical mapping (hex2-up -> screen bottom)")
    p.add_argument("--mirror", action="store_true", help="reverse horizontal mapping")
    p.add_argument("--rows", type=int, default=24)
    p.add_argument("--cols", type=int, default=18)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--out", default=os.path.join(ROOT, "out", "encoder_test.png"))
    return p.parse_args()


if __name__ == "__main__":
    _self_test(_cli())

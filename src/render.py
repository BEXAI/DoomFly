#!/usr/bin/env python3
"""DoomFly offline renderer: composites a recorded episode into a 1080x1920 @ 60 fps mp4.

Scene: a stylised Drosophila perched over the hinge of an open two-panel foldable
("Duo" mock-up, no logo) doomscrolling two synthetic feeds at once; a
picture-in-picture shows the simulated connectome firing as a point cloud; a HUD
shows post counters, whole-brain spike rate and swipe flashes; storyboard cards
(title / stats / attribution) fade in and out.

Inputs (defaults under ``out/`` and ``data/graph/``):
    events.jsonl   one JSON object per 16 ms control step (optional first line
                   ``{"meta": {...}}`` with ``seed_l`` / ``seed_r``)
    spikes.npz     ``frames_packed`` (T, ceil(N/8)) uint8 packbits, ``n``, ``total``
    positions.npy  float32 (N, 3) soma positions (NaN = unknown), optional groups.npy int8

Feeds are replayed deterministically from ``src.feeds.make_pair`` using the seeds in
the meta line (defaults 1 / 2) and the recorded swipe events.

Usage::

    python3 src/render.py --events out/events.jsonl --spikes out/spikes.npz \
        --positions data/graph/positions.npy --out out/final.mp4 [--duration S]
        [--fixture] [--preview] [--stats stats.json] [--png-dir DIR --png-every N]
        [--stills 1,6,9]

``--fixture`` generates a synthetic episode (random point cloud, Poisson spikes with
bursts, alternating swipes) when any input is missing, so the renderer can be
developed end to end without a simulation run.  ``--preview`` renders 540x960 and
every other frame.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
try:
    from src.feeds import make_pair  # noqa: E402
except ImportError:  # running from inside src/
    from feeds import make_pair  # type: ignore  # noqa: E402

# --------------------------------------------------------------------------- #
# Design constants (all geometry is specified at 1080 x 1920 and scaled by k)
# --------------------------------------------------------------------------- #
DESIGN_W, DESIGN_H = 1080, 1920
FPS = 60
CTRL_DT = 0.016
FEED_SCALE = 0.35          # feed raster scale at k = 1 (1335 x 1878 -> 467 x 657)
NOMINAL_LEN_S = 50.0       # storyboard timings are specified for a 50 s episode

# Device quad in the frame (design px): TL, TR, BR, BL (mild keystone, seen from above).
DEVICE_QUAD = np.array([[68, 530], [1012, 530], [1068, 1265], [12, 1265]], np.float32)
DEVICE_BEVEL = 26
DEVICE_HINGE = 14

FLY_CENTER = (540, 945)    # thorax centre (design px)
SWIPE_ANIM_S = 0.42        # reach 0.10 s, drag 0.22 s, return 0.10 s
SWIPE_DRAG_PX = 130        # tarsus travel (forward = toward the head = content scrolls up)

PIP_XY, PIP_SIZE = (600, 44), 460
PIP_DECAY = 0.82

# BGR colours
COL_WHITE = (245, 245, 245)
COL_GREY = (150, 150, 158)
COL_DIM = (95, 95, 105)
COL_L = (230, 200, 40)      # cyan
COL_R = (40, 150, 255)      # orange
GROUP_COLORS = {
    0: (125, 122, 118),     # other: grey
    1: COL_L,               # left eye input
    2: COL_R,               # right eye input
    3: (110, 230, 70),      # descending neurons: green
    4: (220, 80, 240),      # front-leg motor neurons: magenta
}
GROUP_NAMES = {1: "eye L", 2: "eye R", 3: "descending", 4: "leg motor"}

TITLE = "A fly brain doomscrolls two feeds at once"
SUBTITLE = "MaleCNS v1.0 connectome · spiking simulation · nothing scripted"
STATS_HEADLINE = "{n:,} neurons · {e:.2f} M connections (≥5 synapses) · 0 lines of hand-written behaviour"
ATTRIBUTION = (
    "Connectome: MaleCNS v1.0 — HHMI Janelia FlyEM, Cambridge Drosophila Connectomics Group, "
    "Google Research. CC-BY 4.0. male-cns.janelia.org — Berg et al., Cell (2026). "
    "NT predictions: Eckstein, Bates et al. 2024. Neuron model after Shiu et al., Nature 2024. "
    "iPhone Duo is a trademark of Apple Inc.; the device shown is a mock-up."
)

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def ease_in_out(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def ease_out(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 1.0 - (1.0 - x) ** 3


def clamp(x: float, a: float, b: float) -> float:
    return a if x < a else b if x > b else x


def rounded_mask(h: int, w: int, r: int) -> np.ndarray:
    m = np.zeros((max(h, 1), max(w, 1)), np.uint8)
    r = int(max(0, min(r, h // 2, w // 2)))
    cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 255, -1)
    if r > 0:
        for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
            cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return m


def alpha_blit(frame: np.ndarray, bgr: np.ndarray, alpha: np.ndarray, x: int, y: int, gain: float = 1.0) -> None:
    """Alpha-composite ``bgr`` (h,w,3 uint8) with ``alpha`` (h,w uint8) onto frame at (x, y)."""
    H, W = frame.shape[:2]
    h, w = alpha.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sx, sy = x0 - x, y0 - y
    a = alpha[sy : sy + (y1 - y0), sx : sx + (x1 - x0)].astype(np.float32) * (gain / 255.0)
    a = a[..., None]
    roi = frame[y0:y1, x0:x1]
    src = bgr[sy : sy + (y1 - y0), sx : sx + (x1 - x0)]
    roi[:] = (roi.astype(np.float32) * (1.0 - a) + src.astype(np.float32) * a).astype(np.uint8)


def fill_panel(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, r: int, color, alpha: float,
               border: Optional[Tuple[int, int, int]] = None) -> None:
    """Translucent rounded rectangle (with optional 2 px border) blended onto frame."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    roi = frame[y0:y1, x0:x1]
    over = np.empty_like(roi)
    over[:] = color
    mask = rounded_mask(y1 - y0, x1 - x0, r)
    blended = cv2.addWeighted(roi, 1.0 - alpha, over, alpha, 0.0)
    cv2.copyTo(blended, mask, roi)
    if border is not None:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi, cnts, -1, border, 2, cv2.LINE_AA)


class TextRenderer:
    """PIL-rasterised text sprites (cached per string) blitted with alpha onto BGR frames."""

    def __init__(self, k: float):
        self.k = k
        self._fonts: Dict[Tuple[int, bool], "ImageFont.FreeTypeFont"] = {}
        self._cache: Dict[Tuple[str, int, bool], Tuple[np.ndarray, int, int]] = {}
        from PIL import ImageFont  # local import keeps module import light
        self._ImageFont = ImageFont

    def font(self, size: float, bold: bool = False):
        px = max(8, int(round(size * self.k)))
        key = (px, bold)
        f = self._fonts.get(key)
        if f is None:
            path = FONT_BOLD if bold else FONT_REGULAR
            try:
                f = self._ImageFont.truetype(path, px)
            except OSError:
                f = self._ImageFont.load_default(size=px)  # type: ignore[assignment]
            self._fonts[key] = f  # type: ignore[assignment]
        return f

    def measure(self, text: str, size: float, bold: bool = False) -> int:
        return int(math.ceil(self.font(size, bold).getlength(text)))

    def sprite(self, text: str, size: float, bold: bool = False) -> Tuple[np.ndarray, int, int]:
        """Return (alpha uint8 (h,w), ascent_offset, height) for a string."""
        px = max(8, int(round(size * self.k)))
        key = (text, px, bold)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        from PIL import Image, ImageDraw
        f = self.font(size, bold)
        ascent, descent = f.getmetrics()
        w = int(math.ceil(f.getlength(text))) + 4
        h = ascent + descent + 2
        img = Image.new("L", (max(w, 1), max(h, 1)), 0)
        ImageDraw.Draw(img).text((2, 1), text, fill=255, font=f)
        a = np.asarray(img, dtype=np.uint8)
        if len(self._cache) > 512:
            self._cache.clear()
        self._cache[key] = (a, ascent, h)
        return self._cache[key]

    def draw(self, frame: np.ndarray, text: str, x: float, y: float, size: float, color, bold: bool = False,
             align: str = "left", alpha: float = 1.0) -> int:
        """Draw ``text`` with its top-left at design coords (x, y); returns width in frame px."""
        if not text or alpha <= 0:
            return 0
        a, _, _ = self.sprite(text, size, bold)
        h, w = a.shape
        px, py = int(round(x * self.k)), int(round(y * self.k))
        if align == "center":
            px -= w // 2
        elif align == "right":
            px -= w
        bgr = np.empty((h, w, 3), np.uint8)
        bgr[:] = color
        alpha_blit(frame, bgr, a, px, py, gain=alpha)
        return w

    def wrap(self, text: str, size: float, max_width_design: float, bold: bool = False) -> List[str]:
        f = self.font(size, bold)
        max_px = max_width_design * self.k
        lines: List[str] = []
        for para in text.split("\n"):
            words = para.split()
            cur = ""
            for wd in words:
                trial = (cur + " " + wd).strip()
                if f.getlength(trial) <= max_px or not cur:
                    cur = trial
                else:
                    lines.append(cur)
                    cur = wd
            lines.append(cur)
        return lines

    def line_height(self, size: float) -> float:
        return size * 1.32


# --------------------------------------------------------------------------- #
# Fixture generator
# --------------------------------------------------------------------------- #
def make_fixture(out_dir: str, duration: float = 50.0, n: int = 20000, seed: int = 0) -> Dict[str, str]:
    """Write a synthetic episode (events.jsonl, spikes.npz, positions.npy, groups.npy).

    Returns a dict with the four paths.  Two optic lobes, a central brain and a VNC
    blob; Poisson background spiking with sensory -> descending -> motor bursts around
    each swipe; swipes every ~1.2 s alternating L / R with z-scored readout bumps.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_l = n_r = int(n * 0.18)
    n_v = int(n * 0.22)
    n_c = n - n_l - n_r - n_v
    pos = np.concatenate([
        rng.normal([-330.0, -10.0, 0.0], [48.0, 70.0, 55.0], (n_l, 3)),
        rng.normal([330.0, -10.0, 0.0], [48.0, 70.0, 55.0], (n_r, 3)),
        rng.normal([0.0, 20.0, 0.0], [130.0, 80.0, 60.0], (n_c, 3)),
        rng.normal([0.0, 560.0, 20.0], [60.0, 190.0, 45.0], (n_v, 3)),
    ]).astype(np.float32)
    groups = np.zeros(n, np.int8)
    groups[:n_l] = 1
    groups[n_l : n_l + n_r] = 2
    central = np.arange(n_l + n_r, n_l + n_r + n_c)
    dn = rng.choice(central[pos[central, 1] > 40], 400, replace=False)
    groups[dn] = 3
    vnc = np.arange(n_l + n_r + n_c, n)
    mn = rng.choice(vnc[pos[vnc, 1] < 470], 300, replace=False)
    groups[mn] = 4
    pos[rng.choice(n, 60, replace=False)] = np.nan
    pops = {"eye_L": groups == 1, "eye_R": groups == 2, "dn_L": groups == 3, "dn_R": groups == 3,
            "leg_L": groups == 4, "leg_R": groups == 4}
    # Split DN / leg MN into left/right halves by x for the hero counters.
    half = pos[:, 0] < 0
    pops["dn_L"] = (groups == 3) & half
    pops["dn_R"] = (groups == 3) & ~half
    pops["leg_L"] = (groups == 4) & half
    pops["leg_R"] = (groups == 4) & ~half

    T = int(round(duration / CTRL_DT))
    swipe_steps: List[Tuple[int, str]] = []
    t = 1.6
    side = "L"
    while t < duration - 0.5:
        swipe_steps.append((int(round(t / CTRL_DT)), side))
        side = "R" if side == "L" else "L"
        t += float(rng.uniform(0.9, 1.6))
    swipe_at = {s: sd for s, sd in swipe_steps}

    base_rate = 0.012
    rate = np.full((T, 5), base_rate, np.float32)  # per group per step
    readout = np.zeros((T, 2), np.float32)
    for s, sd in swipe_steps:
        g_eye = 1 if sd == "L" else 2
        j = 0 if sd == "L" else 1
        for d in range(-7, 0):
            if 0 <= s + d < T:
                rate[s + d, g_eye] = max(rate[s + d, g_eye], 0.30 * (1 + d / 8))
        for d in range(-3, 2):
            if 0 <= s + d < T:
                rate[s + d, 3] = max(rate[s + d, 3], 0.35)
        for d in range(0, 5):
            if 0 <= s + d < T:
                rate[s + d, 4] = max(rate[s + d, 4], 0.45 * (1 - d / 6))
        for d in range(-5, 40):
            if 0 <= s + d < T:
                readout[s + d, j] += 3.0 * math.exp(-max(d, 0) / 10.0) * (1.0 if d >= 0 else (1 + d / 5.0))
    noise = rng.normal(0, 0.35, (T + 8, 2)).astype(np.float32)
    kernel = np.ones(8, np.float32) / 8
    for j in range(2):
        readout[:, j] += np.convolve(noise[:, j], kernel, "valid")[:T]

    packed = np.empty((T, (n + 7) // 8), np.uint8)
    total = np.zeros(T, np.int32)
    pop_counts = {k: np.zeros(T, np.int32) for k in pops}
    events = []
    posts = {"L": 0, "R": 0}
    for s in range(T):
        r = rate[s][groups]
        fired = rng.random(n) < r
        packed[s] = np.packbits(fired)
        total[s] = int(fired.sum())
        for k, m in pops.items():
            pop_counts[k][s] = int(np.count_nonzero(fired & m))
        sw = swipe_at.get(s)
        if sw:
            posts[sw] += int(rng.integers(1, 3))
        events.append({
            "step": s, "t": round(s * CTRL_DT, 4), "swipe": sw,
            "readout": {"L": round(float(readout[s, 0]), 3), "R": round(float(readout[s, 1]), 3)},
            "spikes": int(total[s]),
            "pops": {k: int(pop_counts[k][s]) for k in pops},
            "posts": dict(posts),
        })

    paths = {
        "events": os.path.join(out_dir, "fixture_events.jsonl"),
        "spikes": os.path.join(out_dir, "fixture_spikes.npz"),
        "positions": os.path.join(out_dir, "fixture_positions.npy"),
        "groups": os.path.join(out_dir, "fixture_groups.npy"),
    }
    with open(paths["events"], "w") as f:
        f.write(json.dumps({"meta": {"seed_l": 1, "seed_r": 2, "dt": CTRL_DT, "n": n, "fixture": True}}) + "\n")
        for e in events:
            f.write(json.dumps(e) + "\n")
    arrs = {"frames_packed": packed, "n": np.int64(n), "total": total}
    for k, v in pop_counts.items():
        arrs["pop_" + k] = v
    np.savez_compressed(paths["spikes"], **arrs)  # type: ignore[arg-type]
    np.save(paths["positions"], pos)
    np.save(paths["groups"], groups)
    return paths


# --------------------------------------------------------------------------- #
# Input loading
# --------------------------------------------------------------------------- #
def load_events(path: str) -> Tuple[dict, List[dict]]:
    meta: dict = {}
    events: List[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if "meta" in o and "step" not in o:
                meta = o["meta"] or {}
                continue
            events.append(o)
    events.sort(key=lambda e: e["step"])
    return meta, events


def load_spikes(path: Optional[str]) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    d = np.load(path)
    out: Dict[str, Any] = {"frames_packed": d["frames_packed"], "n": int(d["n"]), "total": d["total"]}
    return out


# --------------------------------------------------------------------------- #
# Background
# --------------------------------------------------------------------------- #
FLOOR_VP_Y = -5600.0       # design px: vanishing point implied by the device keystone
FLOOR_TOP = (30, 15, 9)    # BGR navy (#090f1e)
FLOOR_BOTTOM = (40, 22, 14)  # BGR navy (#0e1628)
GRID_COLOR = (178, 168, 132)  # BGR cyan-grey


def build_background(W: int, H: int, k: float, quad: np.ndarray) -> np.ndarray:
    """Dark navy floor with a glowing perspective grid, vignette, grain and the device drop shadow."""
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    xs = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    top = np.array(FLOOR_TOP, np.float32)
    bot = np.array(FLOOR_BOTTOM, np.float32)
    col = top[None, None, :] * (1 - ys[..., None]) + bot[None, None, :] * ys[..., None]  # (H,1,3)
    yy = (ys - 0.5) * 2
    r2 = xs ** 2 * 0.9 + yy ** 2
    vig = 1.0 - 0.5 * np.clip(r2, 0, 1)                                                  # (H,W)
    bg = col * vig[..., None]                                                              # (H,W,3)
    # Perspective grid on the floor: lines converge to the vanishing point of the device
    # keystone (far above the frame), rows are spaced uniformly in 1/(y - vp) so they
    # compress toward the distance; brightness fades with distance.
    grid = np.zeros((H, W), np.float32)
    vp_y = FLOOR_VP_Y * k
    cx = W / 2.0
    step = 118.0 * k
    n_v = int(W / step) + 6
    for j in range(-n_v, n_v + 1):
        xb = cx + (j + 0.5) * step
        xt = cx + (xb - cx) * (0 - vp_y) / (H - vp_y)
        cv2.line(grid, _pt((xt, 0)), _pt((xb, H)), 1.0, 1, cv2.LINE_AA)
    y = H + 8.0 * k
    u = 1.0 / (y - vp_y)
    du = step * 0.95 / ((H - vp_y) ** 2)
    while y > -2:
        cv2.line(grid, (0, int(round(y))), (W, int(round(y))), 1.0, 1, cv2.LINE_AA)
        u += du
        y = vp_y + 1.0 / u
    fade = ((np.arange(H, dtype=np.float32) - vp_y) / (H - vp_y)) ** 3.0
    grid *= fade[:, None]
    gcol = np.array(GRID_COLOR, np.float32)
    bloom = cv2.GaussianBlur(grid, (0, 0), 3.0 * max(k, 0.5))
    bg += grid[..., None] * gcol[None, None, :] * 0.10 + bloom[..., None] * gcol[None, None, :] * 0.09
    rng = np.random.default_rng(3)
    grain: np.ndarray = rng.normal(0, 2.8, (H, W)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), 0.8 * max(k, 0.5))
    bg += grain[..., None]
    # Soft cool spot light behind the device.
    scx, scy = W * 0.5, H * 0.47
    spot = np.exp(-(((np.arange(W) - scx) / (W * 0.55)) ** 2)[None, :] - (((np.arange(H) - scy) / (H * 0.28)) ** 2)[:, None])
    bg += (spot[..., None] * np.array([30, 24, 18], np.float32))
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    # Drop shadow of the device (contact-dark core + wide soft penumbra).
    q = quad.copy()
    q[:, 1] += 22 * k
    c = q.mean(axis=0)
    q = (q - c) * np.array([1.03, 1.04], np.float32) + c
    sh = np.zeros((H, W), np.float32)
    cv2.fillPoly(sh, [np.asarray(q, np.int32).reshape(-1, 1, 2)], 1.0)
    sh_wide = cv2.GaussianBlur(sh, (0, 0), 30 * k)
    sh_core = cv2.GaussianBlur(sh, (0, 0), 9 * k)
    shadow = np.clip(0.55 * sh_wide + 0.35 * sh_core, 0, 1)
    bg = (bg.astype(np.float32) * (1.0 - 0.8 * shadow[..., None])).astype(np.uint8)
    return bg


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
class Device:
    """Open two-panel foldable seen from slightly above (warped flat composite)."""

    def __init__(self, k: float, panel_w: int, panel_h: int, quad_design: np.ndarray, hinge_color):
        self.k = k
        B = max(4, int(round(DEVICE_BEVEL * k)))
        G = max(3, int(round(DEVICE_HINGE * k)))
        self.B, self.G = B, G
        self.pw, self.ph = panel_w, panel_h
        Wc, Hc = 2 * panel_w + G + 2 * B, panel_h + 2 * B
        self.Wc, self.Hc = Wc, Hc
        r = int(round(48 * k))
        self.mask = rounded_mask(Hc, Wc, r)
        base = np.zeros((Hc, Wc, 3), np.uint8)
        # Titanium bevel: light gradient top-left -> darker bottom-right.
        gx = np.linspace(0, 1, Wc, dtype=np.float32)[None, :]
        gy = np.linspace(0, 1, Hc, dtype=np.float32)[:, None]
        g = 0.5 * gx + 0.5 * gy
        c0 = np.array([176, 172, 164], np.float32)
        c1 = np.array([92, 90, 86], np.float32)
        bevel = (c0 * (1 - g[..., None]) + c1 * g[..., None]).astype(np.uint8)
        base[:] = bevel
        # Inner dark rim.
        inner = rounded_mask(Hc - 2 * (B - 3), Wc - 2 * (B - 3), max(1, r - B + 3))
        rim = np.empty((inner.shape[0], inner.shape[1], 3), np.uint8)
        rim[:] = (24, 24, 26)
        cv2.copyTo(rim, inner, base[B - 3 : Hc - B + 3, B - 3 : Wc - B + 3])
        # Panel areas (black until the feeds land) and hinge.
        base[B : B + panel_h, B : B + panel_w] = (0, 0, 0)
        base[B : B + panel_h, B + panel_w + G : B + 2 * panel_w + G] = (0, 0, 0)
        hx0 = B + panel_w
        base[B : B + panel_h, hx0 : hx0 + G] = hinge_color
        cv2.line(base, (hx0 + G // 2, B), (hx0 + G // 2, B + panel_h), (40, 40, 44), max(1, G // 4))
        self.base = base
        self.slot_l = (slice(B, B + panel_h), slice(B, B + panel_w))
        self.slot_r = (slice(B, B + panel_h), slice(B + panel_w + G, B + 2 * panel_w + G))

        quad = quad_design * k
        x0, y0 = np.floor(quad.min(axis=0)).astype(int)
        x1, y1 = np.ceil(quad.max(axis=0)).astype(int)
        self.roi = (max(0, x0), max(0, y0), x1 + 1, y1 + 1)
        src = np.array([[0, 0], [Wc, 0], [Wc, Hc], [0, Hc]], np.float32)
        dst = quad - np.array([self.roi[0], self.roi[1]], np.float32)
        self.Hm = cv2.getPerspectiveTransform(src, dst)
        self.roi_size = (self.roi[2] - self.roi[0], self.roi[3] - self.roi[1])
        wm = cv2.warpPerspective(self.mask, self.Hm, self.roi_size, flags=cv2.INTER_LINEAR)
        self.wmask = (wm >= 128).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(self.wmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.outline = cnts
        self.quad = quad

    def to_frame(self, pts_composite: Sequence[Tuple[float, float]]) -> np.ndarray:
        p = np.asarray(pts_composite, np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(p, self.Hm).reshape(-1, 2)
        return out + np.array([self.roi[0], self.roi[1]], np.float32)

    def panel_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        B, G, pw, ph = self.B, self.G, self.pw, self.ph
        pts = self.to_frame([(B + pw / 2, B + ph / 2), (B + pw + G + pw / 2, B + ph / 2)])
        return pts[0], pts[1]

    def panel_bottoms(self) -> Tuple[np.ndarray, np.ndarray]:
        B, G, pw, ph = self.B, self.G, self.pw, self.ph
        pts = self.to_frame([(B + pw / 2, B + ph), (B + pw + G + pw / 2, B + ph)])
        return pts[0], pts[1]

    def draw(self, frame: np.ndarray, feed_l: np.ndarray, feed_r: np.ndarray) -> None:
        comp = self.base.copy()
        comp[self.slot_l] = feed_l
        comp[self.slot_r] = feed_r
        warped = cv2.warpPerspective(comp, self.Hm, self.roi_size, flags=cv2.INTER_LINEAR)
        x0, y0, x1, y1 = self.roi
        roi = frame[y0:y1, x0:x1]
        cv2.copyTo(warped, self.wmask, roi)
        cv2.drawContours(roi, self.outline, -1, (18, 18, 20), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Fly
# --------------------------------------------------------------------------- #
def _hex_bgr(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


FLY_BASE = _hex_bgr("#d9b98a")
FLY_SHADOW = _hex_bgr("#8a6a3c")
FLY_DEEP = _hex_bgr("#4d3319")
FLY_BAND = _hex_bgr("#3a2412")
EYE_BASE = _hex_bgr("#ff5a2a")
EYE_SHADOW = _hex_bgr("#b01a10")
LEG_COL = _hex_bgr("#4a3320")
LEG_EDGE = _hex_bgr("#1c1208")
LEG_LIGHT = _hex_bgr("#8a6a3c")
LEG_JOINT = _hex_bgr("#5c4128")
# Key light: from the top-left, toward the viewer (image coords, y down, z toward the viewer).
_L = np.array([-0.45, -0.60, 0.66], np.float32)
KEY_LIGHT = _L / np.linalg.norm(_L)
_Hv = KEY_LIGHT + np.array([0.0, 0.0, 1.0], np.float32)
HALF_VEC = _Hv / np.linalg.norm(_Hv)


class _Layer:
    """Premultiplied float compositing canvas with ellipsoid shading helpers (sprite build only)."""

    def __init__(self, h: int, w: int):
        self.h, self.w = h, w
        self.pm = np.zeros((h, w, 3), np.float32)   # premultiplied colour
        self.alp = np.zeros((h, w), np.float32)

    # ---- compositing ------------------------------------------------------------
    def over(self, bgr, a: np.ndarray, sl=None) -> None:
        sl = sl or (slice(0, self.h), slice(0, self.w))
        bgr = np.asarray(bgr, np.float32)
        if bgr.ndim == 1:
            bgr = bgr[None, None, :]
        a3 = a[..., None]
        self.pm[sl] = self.pm[sl] * (1 - a3) + bgr * a3
        self.alp[sl] = self.alp[sl] * (1 - a) + a

    def darken(self, a: np.ndarray, strength: float, sl=None) -> None:
        sl = sl or (slice(0, self.h), slice(0, self.w))
        self.pm[sl] *= (1.0 - np.clip(a, 0, 1) * strength)[..., None]

    def light(self, a: np.ndarray, strength: float, color=(255, 250, 240), sl=None) -> None:
        sl = sl or (slice(0, self.h), slice(0, self.w))
        c = np.asarray(color, np.float32)[None, None, :]
        self.pm[sl] += (np.clip(a, 0, 1) * strength * self.alp[sl])[..., None] * c

    def result(self) -> Tuple[np.ndarray, np.ndarray]:
        a = np.clip(self.alp, 0, 1)
        col = self.pm / np.maximum(a, 1e-4)[..., None]
        return np.clip(col, 0, 255), a

    # ---- shapes -------------------------------------------------------------------
    def bbox(self, cx: float, cy: float, rx: float, ry: float):
        x0, x1 = int(max(0, math.floor(cx - rx - 2))), int(min(self.w, math.ceil(cx + rx + 3)))
        y0, y1 = int(max(0, math.floor(cy - ry - 2))), int(min(self.h, math.ceil(cy + ry + 3)))
        return (slice(y0, y1), slice(x0, x1))

    def local(self, sl, cx: float, cy: float, a: float, b: float, ang_deg: float):
        ys, xs = sl
        X, Y = np.meshgrid(np.arange(xs.start, xs.stop, dtype=np.float32), np.arange(ys.start, ys.stop, dtype=np.float32))
        dx, dy = X - cx, Y - cy
        ca, sa = math.cos(math.radians(ang_deg)), math.sin(math.radians(ang_deg))
        u = (dx * ca + dy * sa) / a
        v = (-dx * sa + dy * ca) / b
        return u, v, ca, sa

    def shade(self, u, v, ca, sa, base, shadow, ambient=0.28, spec=0.35, spec_pow=26.0, rim=0.22,
              rim_col=(215, 205, 185), flatten=1.0):
        """Lambert + Blinn specular + rim for a unit ellipsoid given local coords (u, v)."""
        r2 = u * u + v * v
        nz = np.sqrt(np.clip(1.0 - r2, 0, 1)) ** flatten
        nx = u * ca - v * sa
        ny = u * sa + v * ca
        ndl = nx * KEY_LIGHT[0] + ny * KEY_LIGHT[1] + nz * KEY_LIGHT[2]
        lam = np.clip(ndl, 0, 1)
        sh = ambient + (1 - ambient) * lam
        base = np.asarray(base, np.float32)
        shadow = np.asarray(shadow, np.float32)
        col = shadow[None, None, :] + (base - shadow)[None, None, :] * sh[..., None]
        if spec > 0:
            sp = np.clip(nx * HALF_VEC[0] + ny * HALF_VEC[1] + nz * HALF_VEC[2], 0, 1) ** spec_pow * spec
            col += sp[..., None] * np.array([235, 245, 255], np.float32)[None, None, :]
        if rim > 0:
            rf = (1 - nz) ** 3 * rim * np.clip(0.35 - 0.65 * (nx * KEY_LIGHT[0] + ny * KEY_LIGHT[1]), 0, 1)
            col += rf[..., None] * np.asarray(rim_col, np.float32)[None, None, :]
        return col

    def ellipsoid(self, cx: float, cy: float, a: float, b: float, ang_deg: float, base, shadow, opacity: float = 1.0,
                  soft: float = 1.0, **kw):
        """Shade and composite an ellipsoid; returns (slice, alpha, u, v) for follow-up detailing."""
        sl = self.bbox(cx, cy, max(a, b), max(a, b))
        u, v, ca, sa = self.local(sl, cx, cy, a, b, ang_deg)
        r = np.sqrt(u * u + v * v)
        alpha = np.clip((1.0 - r) * min(a, b) / soft, 0, 1) * opacity
        col = self.shade(u, v, ca, sa, base, shadow, **kw)
        self.over(col, alpha, sl)
        return sl, alpha, u, v

    def rounded_shape(self, alpha: np.ndarray, base, shadow, ambient=0.27, spec=0.3, spec_pow=24.0, rim=0.2,
                      rim_col=(215, 205, 185), flatten=1.0):
        """Shade an arbitrary (vertically elongated) mask as a rounded body: the height field is
        a half-circle profile across each row (radius = the row's half-width from the distance
        transform), normals come from its gradient."""
        m8 = (alpha >= 0.5).astype(np.uint8)
        d = cv2.distanceTransform(m8, cv2.DIST_L2, 5).astype(np.float32)
        wrow = np.maximum(d.max(axis=1, keepdims=True), 1e-3)
        wrow = cv2.GaussianBlur(wrow, (1, 0), 4.0) if wrow.shape[0] > 8 else wrow
        h = np.sqrt(np.clip(d * (2.0 * wrow - d), 0, None)) ** flatten
        h = cv2.GaussianBlur(h, (0, 0), 1.2)
        gy, gx = np.gradient(h)
        nz = np.ones_like(h)
        inv = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
        nx, ny, nz = -gx * inv, -gy * inv, nz * inv
        lam = np.clip(nx * KEY_LIGHT[0] + ny * KEY_LIGHT[1] + nz * KEY_LIGHT[2], 0, 1)
        sh = ambient + (1 - ambient) * lam
        base = np.asarray(base, np.float32)
        shadow = np.asarray(shadow, np.float32)
        col = shadow[None, None, :] + (base - shadow)[None, None, :] * sh[..., None]
        if spec > 0:
            sp = np.clip(nx * HALF_VEC[0] + ny * HALF_VEC[1] + nz * HALF_VEC[2], 0, 1) ** spec_pow * spec
            col += sp[..., None] * np.array([235, 245, 255], np.float32)[None, None, :]
        if rim > 0:
            rf = (1 - nz) ** 3 * rim * np.clip(0.35 - 0.65 * (nx * KEY_LIGHT[0] + ny * KEY_LIGHT[1]), 0, 1)
            col += rf[..., None] * np.asarray(rim_col, np.float32)[None, None, :]
        self.over(col, np.clip(alpha, 0, 1))

    def halo(self, sl, alpha: np.ndarray, spread: float, strength: float, dx: float = 0.0, dy: float = 0.0) -> None:
        """Ambient-occlusion style dark halo just outside ``alpha`` (drawn onto whatever is beneath)."""
        ys, xs = sl
        pad = int(math.ceil(spread * 3 + abs(dx) + abs(dy))) + 1
        y0, y1 = max(0, ys.start - pad), min(self.h, ys.stop + pad)
        x0, x1 = max(0, xs.start - pad), min(self.w, xs.stop + pad)
        big = np.zeros((y1 - y0, x1 - x0), np.float32)
        big[ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0] = alpha
        M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
        shifted = cv2.warpAffine(big, M, (x1 - x0, y1 - y0))
        blurred = cv2.GaussianBlur(shifted, (0, 0), spread)
        ao = np.clip(blurred - big, 0, 1)
        self.darken(ao, strength, (slice(y0, y1), slice(x0, x1)))


class Fly:
    """Top-down 2.5D Drosophila: shaded body sprite (built once, 2x supersampled) + procedural legs."""

    # Design-space geometry relative to the thorax centre; +x = viewer's right = fly's right
    # (head toward the top of the frame, so the fly's left is the viewer's left).
    LEG_ATTACH = {"front": (34, -30), "mid": (52, 2), "hind": (40, 32)}
    LEG_REST = {"front": (150, -108), "mid": (196, 24), "hind": (166, 160)}
    LEG_THICK = {"front": (8, 6, 4), "mid": (7, 5, 3), "hind": (8, 5, 3)}
    EYE_CENTER = (28, -80)

    def __init__(self, k: float, center_px: Tuple[float, float]):
        self.k = k
        self.cx, self.cy = center_px
        self.canvas_origin = (120, 150)  # design px offset of thorax centre inside the sprite
        self.sprite_bgr, self.sprite_a = self._build_sprite()
        self.shadow_a, self.shadow_origin = self._build_contact_shadow()
        self.leg_color = LEG_COL
        self.leg_edge = LEG_EDGE
        self.rng = np.random.default_rng(11)
        self.twitch_phase = self.rng.uniform(0, 6.28, 2)

    # ---- sprite -----------------------------------------------------------------
    def _build_sprite(self) -> Tuple[np.ndarray, np.ndarray]:
        k = self.k
        ss = 2
        s = k * ss
        cw, ch = int(240 * s), int(420 * s)
        ox, oy = self.canvas_origin
        L = _Layer(ch, cw)
        rng = np.random.default_rng(5)

        def X(x: float) -> float:
            return (x + ox) * s

        def Y(y: float) -> float:
            return (y + oy) * s

        def P(x: float, y: float) -> Tuple[int, int]:
            return int(round(X(x))), int(round(Y(y)))

        def R(v: float) -> int:
            return max(1, int(round(v * s)))

        def mask_alpha(m: np.ndarray) -> np.ndarray:
            return m.astype(np.float32) / 255.0

        def ell(x, y, a, b, ang, base, shadow, **kw):
            return L.ellipsoid(X(x), Y(y), a * s, b * s, ang, base, shadow, **kw)

        # --- abdomen: one tapered ovoid shaded as a rounded body, then soft dark tergite
        #     bands that curve with the segment edges and an AO groove above each band
        ys_ = np.linspace(22, 214, 48)
        left, right = [], []
        for yv in ys_:
            uu = (yv - 112) / 100.0
            w = 50 * math.sqrt(max(0.0, 1 - uu * uu)) * (1 - 0.28 * (yv - 22) / 192)
            left.append(P(-w, yv))
            right.append(P(w, yv))
        abd_poly = np.array(left + right[::-1], np.int32).reshape(-1, 1, 2)
        am = np.zeros((ch, cw), np.uint8)
        cv2.fillPoly(am, [abd_poly], 255, cv2.LINE_AA)
        abd_a = mask_alpha(am)
        L.rounded_shape(abd_a, FLY_BASE, FLY_SHADOW, ambient=0.26, spec=0.28, rim=0.22, flatten=0.9)
        Yg, Xg = np.mgrid[0:ch, 0:cw].astype(np.float32)
        xd = Xg / s - ox                     # design units
        yd = Yg / s - oy
        for i, yb in enumerate((58, 92, 126, 158, 186)):
            curve = yb + 0.045 * xd * xd / 10.0          # tergite edges bow toward the rear at the flanks
            hw = 7.5 if i < 3 else 6.0
            band = np.clip(1.0 - (np.abs(yd - curve - 4.0) - hw) / 3.0, 0, 1) * abd_a
            band *= np.clip(1.0 - (xd / 52.0) ** 2 * 0.25, 0, 1)
            L.darken(band, 0.62)
            L.over(FLY_BAND, band * 0.40)
            groove = np.clip(1.0 - np.abs(yd - curve + hw - 1.5) / 2.2, 0, 1) * abd_a
            L.darken(groove, 0.35)
            ridge = np.clip(1.0 - np.abs(yd - curve - hw - 6.5) / 1.6, 0, 1) * abd_a
            L.light(ridge, 0.10)
        # dark tip (male genital segment)
        tipm = np.clip((yd - 196) / 10.0, 0, 1) * abd_a
        L.darken(tipm, 0.55)
        L.over(FLY_BAND, tipm * 0.5)
        # dorsal sheen along the abdomen midline
        sh = np.zeros((ch, cw), np.uint8)
        cv2.ellipse(sh, P(-10, 110), (R(14), R(78)), 0, 0, 360, 255, -1, cv2.LINE_AA)
        sh_a = cv2.GaussianBlur(mask_alpha(sh), (0, 0), 6 * s)
        L.light(sh_a, 0.10)
        # --- wings (translucent, over the abdomen; bases hidden under the thorax) ------
        for side in (-1, 1):
            wcx, wcy, wa, wb, wang = X(side * 42), Y(126), 31 * s, 128 * s, side * 9
            sl = L.bbox(wcx, wcy, max(wa, wb), max(wa, wb))
            u, v, ca, sa = L.local(sl, wcx, wcy, wa, wb, wang)
            prof = np.sqrt(np.clip(1 - v * v, 0, 1)) * (0.28 + 0.72 * np.clip((v + 1) / 2, 0, 1) ** 0.65)
            dist = (prof - np.abs(u)) * wa            # px to the wing edge
            wmask = np.clip(dist / 1.0, 0, 1)
            wmask *= np.clip((v + 1.0) * 6, 0, 1)      # square off the hinge end
            # membrane
            L.over((208, 212, 220), wmask * 0.25, sl)
            # iridescent sheen (very faint hue sweep) + broad specular streak
            t1 = (np.sin(v * 3.2 + u * 1.6 + side) + 1) / 2
            sheen = np.stack([255 * (1 - t1) + 205 * t1, 190 * (1 - t1) + 150 * t1, 150 * (1 - t1) + 255 * t1], axis=-1)
            L.over(sheen, wmask * 0.06, sl)
            streak = np.exp(-((u + side * 0.25) / 0.28) ** 2) * np.exp(-((v - 0.05) / 0.55) ** 2)
            L.light(streak * wmask, 0.16, (245, 245, 250), sl)
            # edge
            edge = np.clip(1.0 - dist / (1.8 * s), 0, 1) * wmask
            L.over((205, 210, 220), edge * 0.35, sl)
            # veins in wing-local coordinates -> pixels
            def wp(uu: float, vv: float) -> Tuple[int, int]:
                dx, dy = uu * wa, vv * wb
                return int(round(wcx + dx * ca - dy * sa)), int(round(wcy + dx * sa + dy * ca))
            vm = np.zeros((ch, cw), np.uint8)
            veins = [((0.0, -0.98), (-0.25, -0.4), (-0.58, 0.55)), ((0.0, -0.98), (-0.1, 0.0), (-0.28, 0.92)),
                     ((0.02, -0.98), (0.1, 0.1), (0.1, 0.99)), ((0.05, -0.98), (0.3, 0.0), (0.48, 0.86)),
                     ((0.08, -0.98), (0.45, -0.3), (0.78, 0.45)), ((0.1, -0.98), (0.55, -0.6), (0.9, -0.05))]
            for p0, p1, p2 in veins:
                pts = []
                for tt in np.linspace(0, 1, 12):
                    uu = (1 - tt) ** 2 * p0[0] + 2 * (1 - tt) * tt * p1[0] + tt ** 2 * p2[0]
                    vv = (1 - tt) ** 2 * p0[1] + 2 * (1 - tt) * tt * p1[1] + tt ** 2 * p2[1]
                    pts.append(wp(side * uu, vv))
                cv2.polylines(vm, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, 255, R(1.4), cv2.LINE_AA)
            for (a0, b0), (a1, b1) in (((-0.2, 0.15), (0.1, 0.2)), ((0.12, 0.45), (0.45, 0.4))):  # crossveins
                cv2.line(vm, wp(side * a0, b0), wp(side * a1, b1), 255, R(1.2), cv2.LINE_AA)
            va = mask_alpha(vm)[sl] * np.clip(wmask * 3, 0, 1)
            L.over((112, 120, 140), va * 0.6, sl)
        # --- neck (under head and thorax) ----------------------------------------
        ell(0, -47, 15, 11, 0, FLY_SHADOW, FLY_DEEP, ambient=0.3, spec=0.1, rim=0.0)
        # --- thorax ----------------------------------------------------------------
        sl, al, u, v = ell(0, 2, 54, 50, 0, FLY_BASE, FLY_SHADOW, ambient=0.27, spec=0.32, spec_pow=22, rim=0.22, flatten=0.9)
        L.halo(sl, al, spread=3.0 * s, strength=0.42, dy=2.5 * s)
        # bristle texture: many short dark hairs + fine pores, denser toward the rear
        bm = np.zeros((ch, cw), np.uint8)
        for _ in range(110):
            r = math.sqrt(rng.uniform(0, 1))
            th = rng.uniform(0, 2 * math.pi)
            bx, by = 50 * r * math.cos(th), 46 * r * math.sin(th) + 2
            ln = rng.uniform(2.5, 5.0)
            ang = -math.pi / 2 + rng.normal(0, 0.35) + (0.5 if bx > 0 else -0.5) * (r * 0.8)
            cv2.line(bm, P(bx, by), P(bx + ln * math.cos(ang), by + ln * math.sin(ang)), 255, R(0.9), cv2.LINE_AA)
        for _ in range(140):
            r = math.sqrt(rng.uniform(0, 1))
            th = rng.uniform(0, 2 * math.pi)
            cv2.circle(bm, P(50 * r * math.cos(th), 46 * r * math.sin(th) + 2), R(0.6), 255, -1, cv2.LINE_AA)
        L.darken(mask_alpha(bm), 0.14)
        # faint longitudinal stripes + midline groove (Drosophila notum pattern)
        gm = np.zeros((ch, cw), np.uint8)
        for gx in (-22, 22):
            cv2.line(gm, P(gx, -34), P(gx * 1.1, 30), 255, R(5), cv2.LINE_AA)
        L.darken(cv2.GaussianBlur(mask_alpha(gm), (0, 0), 3 * s), 0.10)
        gm[:] = 0
        cv2.line(gm, P(0, -42), P(0, 34), 255, R(1.4), cv2.LINE_AA)
        L.darken(cv2.GaussianBlur(mask_alpha(gm), (0, 0), 0.8 * s), 0.16)
        # scutellum (rounded shield at the rear of the thorax)
        sl, al, u, v = ell(0, 44, 24, 14, 0, tuple(int(0.75 * a_ + 0.25 * b_) for a_, b_ in zip(FLY_BASE, FLY_SHADOW)), FLY_SHADOW,
                           ambient=0.3, spec=0.25, rim=0.15)
        L.halo(sl, al, spread=2.2 * s, strength=0.5, dy=1.5 * s)
        # --- head ------------------------------------------------------------------
        sl, al, u, v = ell(0, -78, 35, 33, 0, FLY_BASE, FLY_SHADOW, ambient=0.27, spec=0.3, rim=0.2)
        L.halo(sl, al, spread=2.6 * s, strength=0.5, dy=2.0 * s)
        # frons: slightly darker band between the eyes
        fm = np.zeros((ch, cw), np.uint8)
        cv2.ellipse(fm, P(0, -84), (R(9), R(22)), 0, 0, 360, 255, -1, cv2.LINE_AA)
        L.darken(cv2.GaussianBlur(mask_alpha(fm), (0, 0), 2.5 * s), 0.12)
        # proboscis
        ell(0, -108, 8, 11, 0, FLY_SHADOW, FLY_DEEP, ambient=0.3, spec=0.2, rim=0.1)
        ell(0, -113, 5, 4, 0, FLY_DEEP, (20, 26, 40), ambient=0.3, spec=0.3, rim=0.0)
        # antennal bases
        for side in (-1, 1):
            ell(side * 9, -100, 4.5, 4.5, 0, FLY_SHADOW, FLY_DEEP, ambient=0.3, spec=0.3, rim=0.0)
        # ocelli: three tiny glossy amber lenses
        for ox_, oy_ in ((0, -104), (-7, -96), (7, -96)):
            ell(ox_, oy_, 2.8, 2.8, 0, (60, 150, 230), (20, 50, 120), ambient=0.35, spec=0.9, spec_pow=12, rim=0.0)
        # compound eyes
        ex, ey = self.EYE_CENTER
        for side in (-1, 1):
            cxe, cye, ea, eb, eang = X(side * ex), Y(ey), 21 * s, 29 * s, side * 14
            # dark socket rim
            sl = L.bbox(cxe, cye, eb + 2 * s, eb + 2 * s)
            u, v, ca, sa = L.local(sl, cxe, cye, ea + 1.6 * s, eb + 1.6 * s, eang)
            ring = np.clip((1 - np.sqrt(u * u + v * v)) * (ea + 1.6 * s) / 1.0, 0, 1)
            L.over((16, 14, 60), ring * 0.85, sl)
            sl, al, u, v = L.ellipsoid(cxe, cye, ea, eb, eang, EYE_BASE, EYE_SHADOW, ambient=0.34, spec=0.0, rim=0.35,
                                       rim_col=(90, 150, 255))
            ca, sa = math.cos(math.radians(eang)), math.sin(math.radians(eang))
            # hex-facet micro texture (dark lattice + faint bright dot per facet), clipped to the eye
            fm = np.zeros((ch, cw), np.uint8)
            fm2 = np.zeros((ch, cw), np.uint8)
            step = 3.3
            row = 0
            fy: float = ey - 32
            while fy < ey + 32:
                fx: float = side * ex - 24 + (step / 2 if row % 2 else 0)
                while fx < side * ex + 24:
                    cv2.circle(fm, P(fx, fy), R(1.05), 255, -1, cv2.LINE_AA)
                    cv2.circle(fm2, P(fx - 0.5, fy - 0.5), R(0.45), 255, -1, cv2.LINE_AA)
                    fx += step
                fy += step * 0.87
                row += 1
            L.darken(mask_alpha(fm)[sl] * al, 0.30, sl)
            L.light(mask_alpha(fm2)[sl] * al, 0.10, (255, 240, 230), sl)
            # two specular highlights: a soft broad one and a sharp small one (top-left key)
            hm = np.zeros((ch, cw), np.uint8)
            cv2.ellipse(hm, P(side * (ex - 6), ey - 12), (R(5.5), R(9.5)), eang, 0, 360, 255, -1, cv2.LINE_AA)
            hl = cv2.GaussianBlur(mask_alpha(hm), (0, 0), 2.2 * s)[sl] * al
            L.light(hl, 0.62, (255, 250, 245), sl)
            hm[:] = 0
            cv2.circle(hm, P(side * (ex - 8), ey - 17), R(2.4), 255, -1, cv2.LINE_AA)
            L.light(mask_alpha(hm)[sl] * al, 0.95, (255, 255, 255), sl)
            hm[:] = 0
            cv2.ellipse(hm, P(side * (ex + 6), ey + 13), (R(2.2), R(4.5)), eang, 0, 360, 255, -1, cv2.LINE_AA)
            L.light(cv2.GaussianBlur(mask_alpha(hm), (0, 0), 1.2 * s)[sl] * al, 0.30, (255, 220, 200), sl)

        col, a = L.result()
        bgr = col.astype(np.uint8)
        a8: np.ndarray = (a * 255).astype(np.uint8)
        out_w, out_h = int(240 * k), int(420 * k)
        bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)  # type: ignore[assignment]
        a8 = cv2.resize(a8, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return bgr, a8

    def _build_contact_shadow(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Soft elliptical shadow under the body (alpha only), origin = offset of the thorax centre."""
        k = self.k
        w, h = int(300 * k), int(440 * k)
        m = np.zeros((h, w), np.float32)
        cv2.ellipse(m, (int(w / 2), int(h / 2 + 30 * k)), (int(62 * k), int(150 * k)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        cv2.circle(m, (int(w / 2), int(h / 2 - 78 * k)), int(36 * k), 1.0, -1, cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), 14 * max(k, 0.5))  # type: ignore[assignment]
        a = (np.clip(m, 0, 1) * 0.55 * 255).astype(np.uint8)
        return a, (int(w / 2 - 10 * k), int(h / 2 - 16 * k))

    # ---- per-frame ------------------------------------------------------------------
    def eye_positions(self, t: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        bob = self._bob(t)
        ex, ey = self.EYE_CENTER
        k = self.k
        return ((self.cx - ex * k, self.cy + (ey + bob) * k), (self.cx + ex * k, self.cy + (ey + bob) * k))

    def _bob(self, t: float) -> float:
        return 1.6 * math.sin(2 * math.pi * 1.1 * t) + 0.6 * math.sin(2 * math.pi * 2.7 * t + 1.0)

    def _zero_like(self, roi: np.ndarray) -> np.ndarray:
        z = getattr(self, "_zeros", None)
        if z is None or z.shape != roi.shape:
            z = np.zeros_like(roi)
            self._zeros = z
        return z

    def _front_tip(self, side: int, age: Optional[float]) -> Tuple[float, float]:
        rx, ry = self.LEG_REST["front"]
        rest = (side * rx, ry)
        if age is None or age < 0 or age >= SWIPE_ANIM_S:
            return rest
        t_reach, t_drag = 0.10, 0.32
        plant = (rest[0] + side * 18, rest[1] + 56)
        end = (plant[0] + side * 6, plant[1] - SWIPE_DRAG_PX)
        if age < t_reach:
            p = ease_in_out(age / t_reach)
            return (rest[0] + (plant[0] - rest[0]) * p, rest[1] + (plant[1] - rest[1]) * p)
        if age < t_drag:
            p = ease_out((age - t_reach) / (t_drag - t_reach))
            return (plant[0] + (end[0] - plant[0]) * p, plant[1] + (end[1] - plant[1]) * p)
        p = ease_in_out((age - t_drag) / (SWIPE_ANIM_S - t_drag))
        return (end[0] + (rest[0] - end[0]) * p, end[1] + (rest[1] - end[1]) * p)

    def _leg_points(self, name: str, side: int, A: np.ndarray, T: np.ndarray) -> List[np.ndarray]:
        d = T - A
        Ln = float(np.hypot(*d)) + 1e-6
        u = d / Ln
        n = np.array([-u[1], u[0]], np.float32)
        if n[0] * side + n[1] * (-0.45) < 0:
            n = -n
        bend = 0.36 if name != "front" else 0.30
        knee = A + u * (0.46 * Ln) + n * (bend * Ln)
        ankle = A + u * (0.86 * Ln) + n * (0.10 * Ln)
        return [A, knee, ankle, T]

    def _draw_leg(self, frame: np.ndarray, pts: List[np.ndarray], th: Tuple[int, int, int]) -> None:
        k = self.k
        # 1) dark outline, 2) body, 3) light stripe offset toward the key light (cylinder shading)
        for i in range(3):
            p0, p1 = _pt(pts[i]), _pt(pts[i + 1])
            cv2.line(frame, p0, p1, self.leg_edge, max(1, int(round((th[i] + 2.6) * k))), cv2.LINE_AA)
        for i in range(3):
            p0, p1 = _pt(pts[i]), _pt(pts[i + 1])
            cv2.line(frame, p0, p1, self.leg_color, max(1, int(round(th[i] * k))), cv2.LINE_AA)
        for i in range(3):
            d = pts[i + 1] - pts[i]
            Ln = float(np.hypot(*d)) + 1e-6
            n = np.array([-d[1], d[0]], np.float32) / Ln
            if n[0] * KEY_LIGHT[0] + n[1] * KEY_LIGHT[1] < 0:
                n = -n
            off = n * (th[i] * 0.22 * k)
            w = th[i] * 0.34 * k
            if w >= 0.8:
                p0, p1 = _pt(pts[i] + off), _pt(pts[i + 1] + off)
                cv2.line(frame, p0, p1, LEG_LIGHT, max(1, int(round(w))), cv2.LINE_AA)
        # rounded joints with a small highlight
        for j, r_ in ((1, th[0] * 0.62), (2, th[1] * 0.60)):
            c = _pt(pts[j])
            rr = max(1, int(round(r_ * k)))
            cv2.circle(frame, c, rr + max(1, int(round(1.2 * k))), self.leg_edge, -1, cv2.LINE_AA)
            cv2.circle(frame, c, rr, LEG_JOINT, -1, cv2.LINE_AA)
            hc = _pt(pts[j] + np.array([KEY_LIGHT[0], KEY_LIGHT[1]]) * (r_ * 0.35 * k))
            cv2.circle(frame, hc, max(1, int(round(r_ * 0.33 * k))), LEG_LIGHT, -1, cv2.LINE_AA)
        # claw at the tarsus tip
        d = pts[3] - pts[2]
        Ln = float(np.hypot(*d)) + 1e-6
        u = d / Ln
        for ang in (-0.55, 0.55):
            ca, sa = math.cos(ang), math.sin(ang)
            v = np.array([u[0] * ca - u[1] * sa, u[0] * sa + u[1] * ca], np.float32)
            cv2.line(frame, _pt(pts[3]), _pt(pts[3] + v * (5.5 * k)), self.leg_edge, max(1, int(round(1.6 * k))), cv2.LINE_AA)

    def draw(self, frame: np.ndarray, t: float, swipe_age: Dict[str, Optional[float]]) -> None:
        k = self.k
        bob = self._bob(t)
        ox, oy = self.cx, self.cy + bob * k
        H, W = frame.shape[:2]

        def F(x: float, y: float, with_bob: bool = True) -> Tuple[int, int]:
            return int(round(ox + x * k)), int(round((oy if with_bob else self.cy) + y * k))

        legs: List[Tuple[str, int, List[np.ndarray]]] = []
        for name in ("hind", "mid", "front"):
            ax, ay = self.LEG_ATTACH[name]
            for side in (-1, 1):
                A = np.asarray(F(side * ax, ay), np.float32)
                if name == "front":
                    age = swipe_age["L"] if side < 0 else swipe_age["R"]
                    tx, ty = self._front_tip(side, age)
                    T = np.asarray(F(tx, ty, with_bob=False), np.float32)
                else:
                    rx, ry = self.LEG_REST[name]
                    T = np.asarray(F(side * rx, ry, with_bob=False), np.float32)
                legs.append((name, side, self._leg_points(name, side, A, T)))

        # ---- shadows on the device: soft contact shadow under the body + leg shadows ----
        x0, y0 = max(0, int(ox - 250 * k)), max(0, int(oy - 175 * k))
        x1, y1 = min(W, int(ox + 250 * k)), min(H, int(oy + 240 * k))
        if x1 > x0 and y1 > y0:
            sh = np.zeros((y1 - y0, x1 - x0), np.uint8)
            offs = np.array([x0 - 9 * k, y0 - 15 * k], np.float32)
            for name, side, pts in legs:
                th = self.LEG_THICK[name]
                for i in range(3):
                    cv2.line(sh, _pt(pts[i] - offs), _pt(pts[i + 1] - offs), 255, max(1, int(round((th[i] + 1.5) * k))), cv2.LINE_AA)
            sh = cv2.GaussianBlur(sh, (0, 0), 4.0 * max(k, 0.5))
            wgt = sh.astype(np.float32) * (0.42 / 255.0)
            sax, say = self.shadow_origin
            _paste_max(wgt, self.shadow_a.astype(np.float32) / 255.0, int(round(ox - sax)) - x0, int(round(oy - say)) - y0)
            roi = frame[y0:y1, x0:x1]
            roi[:] = cv2.blendLinear(roi, self._zero_like(roi), 1.0 - wgt, wgt)

        # ---- legs (under the body) ----
        for name, side, pts in legs:
            self._draw_leg(frame, pts, self.LEG_THICK[name])
            # tarsus contact ring while dragging
            if name == "front":
                age = swipe_age["L"] if side < 0 else swipe_age["R"]
                if age is not None and 0.08 <= age < 0.36:
                    a = 1.0 - abs((age - 0.22) / 0.14)
                    a = clamp(a, 0, 1)
                    colr = COL_L if side < 0 else COL_R
                    cc = tuple(int(c * a + 40 * (1 - a)) for c in colr)
                    cv2.circle(frame, _pt(pts[3]), int(round((16 + 14 * (1 - a)) * k)), cc, max(1, int(round(3.5 * k))), cv2.LINE_AA)

        # ---- antennae (idle twitch) ----
        for i, side in enumerate((-1, 1)):
            ph = self.twitch_phase[i]
            tw = 0.18 * math.sin(2 * math.pi * 0.6 * t + ph) + 0.10 * math.sin(2 * math.pi * 4.1 * t + ph * 2) * max(0.0, math.sin(2 * math.pi * 0.23 * t + ph))
            ang = -math.pi / 2 + side * (0.62 + tw)
            base = F(side * 9, -100)
            mid = F(side * 9 + 16 * math.cos(ang), -100 + 16 * math.sin(ang))
            tip = F(side * 9 + 30 * math.cos(ang) + side * 4, -100 + 30 * math.sin(ang) - 3)
            cv2.line(frame, base, mid, LEG_EDGE, max(1, int(round(4.4 * k))), cv2.LINE_AA)
            cv2.line(frame, base, mid, FLY_SHADOW, max(1, int(round(2.8 * k))), cv2.LINE_AA)
            cv2.line(frame, mid, tip, LEG_EDGE, max(1, int(round(2.6 * k))), cv2.LINE_AA)
            cv2.line(frame, mid, tip, LEG_COL, max(1, int(round(1.4 * k))), cv2.LINE_AA)
            cv2.circle(frame, mid, max(1, int(round(2.6 * k))), LEG_JOINT, -1, cv2.LINE_AA)
            # arista (feathery tip)
            for j in range(4):
                a2 = ang + side * (0.45 + 0.32 * j)
                p1 = (int(mid[0] + (9 + j) * k * math.cos(a2)), int(mid[1] + (9 + j) * k * math.sin(a2)))
                cv2.line(frame, mid, p1, LEG_EDGE, max(1, int(round(1.1 * k))), cv2.LINE_AA)

        # ---- body sprite ----
        sx = int(round(ox - self.canvas_origin[0] * k))
        sy = int(round(oy - self.canvas_origin[1] * k))
        alpha_blit(frame, self.sprite_bgr, self.sprite_a, sx, sy)


def _paste_max(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """dst[y:, x:] = max(dst, src) with clipping at the borders (single-channel float masks)."""
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sx, sy = x0 - x, y0 - y
    np.maximum(dst[y0:y1, x0:x1], src[sy : sy + (y1 - y0), sx : sx + (x1 - x0)], out=dst[y0:y1, x0:x1])


# --------------------------------------------------------------------------- #
# Brain picture-in-picture
# --------------------------------------------------------------------------- #
# Region colours (BGR, 0..1): optic lobes warm amber/orange, central brain blue, VNC cyan,
# descending neurons green-cyan, leg motor neurons magenta.
PIP_REGION_COLORS = np.array([
    [1.00, 0.56, 0.24],   # 0 central brain  (#3d8fff)
    [0.92, 0.78, 0.22],   # 1 VNC            (#38c7eb)
    [0.20, 0.66, 1.00],   # 2 optic lobe L   (#ffa833)
    [0.14, 0.52, 1.00],   # 3 optic lobe R   (#ff8524)
    [0.72, 0.92, 0.36],   # 4 descending     (#5cebb8)
    [0.95, 0.35, 0.95],   # 5 leg motor      (#f259f2)
], np.float32)
PIP_REGION_GAIN = np.array([0.55, 0.60, 0.62, 0.62, 2.4, 2.8], np.float32)
PIP_YAW_AMP_DEG = 12.0
PIP_YAW_PERIOD_S = 20.0
PIP_YAW_STEP_DEG = 0.25     # cached projection granularity (frames blend between neighbours)
PIP_TILT_DEG = 30.0         # dorsal-anterior oblique: brain in front, VNC hanging below/behind


class BrainPiP:
    """3D point cloud of soma positions with additive depth-weighted rendering and bloom.

    The static cloud is accumulated per yaw step (0.25 deg) into a cache and blended
    between neighbouring steps per frame; spikes lift points toward white into a glow
    buffer with exponential decay (``PIP_DECAY``).
    """

    def __init__(self, positions: Optional[np.ndarray], groups: Optional[np.ndarray], n: int, size: int, k: float):
        self.size = size
        self.k = k
        self.n = n
        S = size
        self.glow = np.zeros((S, S, 3), np.float32)
        self._cache: Dict[int, np.ndarray] = {}
        self._proj: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.expo = 1.0
        self.counts: Dict[int, int] = {}
        if positions is None:
            self.valid_idx = np.zeros(0, np.int64)
            self.flat = np.zeros(0, np.int64)
            self.w = np.zeros(0, np.float32)
            self.spike_cols = np.zeros((0, 3), np.float32)
            return
        pos = positions[:n] if positions.shape[0] >= n else np.vstack([positions, np.full((n - positions.shape[0], 3), np.nan, np.float32)])
        if groups is None:
            groups = np.zeros(n, np.int8)
        groups = groups[:n] if groups.shape[0] >= n else np.concatenate([groups, np.zeros(n - groups.shape[0], np.int8)])
        valid = np.isfinite(pos).all(axis=1)
        valid_idx = np.flatnonzero(valid)
        P = pos[valid].astype(np.float32)
        P -= P.mean(axis=0)
        g = groups[valid].astype(np.int64)
        g[(g < 0) | (g > 4)] = 0
        # Region classes: group 0 is split into central brain / VNC with a 2-means split
        # along its widest axis (the VNC is a separate blob posterior to the brain).
        region = np.zeros(g.size, np.int64)
        region[g == 1] = 2
        region[g == 2] = 3
        region[g == 3] = 4
        region[g == 4] = 5
        g0 = np.flatnonzero(g == 0)
        if g0.size > 50:
            Q = P[g0][:, 1:]                       # (y, z)
            ax = int(np.argmax(Q.std(axis=0)))
            q = Q[:, ax]
            c0, c1 = np.percentile(q, 20), np.percentile(q, 80)
            for _ in range(12):
                lab = np.abs(q - c1) < np.abs(q - c0)
                if lab.all() or (~lab).all():
                    break
                c0, c1 = q[~lab].mean(), q[lab].mean()
            lab = np.abs(q - c1) < np.abs(q - c0)
            if lab.any() and (~lab).any():
                sep = abs(c1 - c0)
                spread = q[lab].std() + q[~lab].std()
                if sep > 1.1 * spread and 0.05 < lab.mean() < 0.6:
                    region[g0[lab]] = 1
        # Sort by region so per-region accumulation uses contiguous slices.
        order = np.argsort(region, kind="stable")
        self.valid_idx = valid_idx[order]
        self.region = region[order]
        self.bounds = np.searchsorted(self.region, np.arange(7))
        P = P[order]
        g = g[order]
        for gi in range(5):
            self.counts[gi] = int((g == gi).sum())
        # Mirror x: the volume's x axis runs from the fly's right to its left (anterior view),
        # while the fly in the scene is drawn head-up with its left at the viewer's left.
        self.x = -P[:, 0]
        self.y = P[:, 1]
        self.z = P[:, 2]
        tilt = math.radians(PIP_TILT_DEG)
        self.ct, self.st = math.cos(tilt), math.sin(tilt)
        # Fit: screen extents over the yaw range, mild perspective.
        span = float(max(np.ptp(self.x), np.ptp(self.y), np.ptp(self.z), 1e-3))
        self.focal = 2.6 * span
        xs, ys_, ds = [], [], []
        for yaw in (-PIP_YAW_AMP_DEG, 0.0, PIP_YAW_AMP_DEG):
            sx, sy, d = self._transform(math.radians(yaw))
            xs.append(sx)
            ys_.append(sy)
            ds.append(d)
        allx, ally, alld = np.concatenate(xs), np.concatenate(ys_), np.concatenate(ds)
        self.d_min, self.d_max = float(alld.min()), float(alld.max())
        lo_x, hi_x = np.percentile(allx, [0.2, 99.8])
        lo_y, hi_y = np.percentile(ally, [0.2, 99.8])
        pad = S * 0.08
        self.scale = (S - 2 * pad) / max(hi_x - lo_x, hi_y - lo_y, 1e-6)
        self.off_x = S / 2 - (lo_x + hi_x) / 2 * self.scale
        self.off_y = S / 2 - (lo_y + hi_y) / 2 * self.scale
        # Spike colours: region colour lifted most of the way to white (L/R eye tint kept
        # from GROUP_COLORS so the swiping side still reads).
        gcols = np.array([GROUP_COLORS[i] for i in range(5)], np.float32) / 255.0
        sc = PIP_REGION_COLORS[self.region] * 0.35 + 0.65
        sc[g == 1] = gcols[1] * 0.4 + 0.6
        sc[g == 2] = gcols[2] * 0.4 + 0.6
        self.spike_cols = sc.astype(np.float32)
        # Initial projection + exposure so the dense core lands near white for any point count.
        self.flat, self.w, self.wn = self._project(0.0)
        base0 = self._base(0)
        lum = base0.max(axis=2)
        p = float(np.percentile(lum[lum > 0.02], 96)) if (lum > 0.02).any() else 1.0
        self.expo = 1.75 / max(p, 1e-6)

    # ---- projection -----------------------------------------------------------------
    def _transform(self, yaw: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        x1 = self.x * cy_ + self.z * sy_
        z1 = -self.x * sy_ + self.z * cy_
        sy = self.y * self.ct + z1 * self.st          # posterior (z) tips downward on screen
        d = -self.y * self.st + z1 * self.ct          # depth: smaller = nearer (anterior)
        return x1, sy, d

    def _project(self, yaw: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        S = self.size
        sx, sy, d = self._transform(yaw)
        persp = self.focal / (self.focal + (d - self.d_min))
        px = sx * persp * self.scale + self.off_x
        py = sy * persp * self.scale + self.off_y
        pxi = np.clip(np.rint(px).astype(np.int32), 0, S - 1)
        pyi = np.clip(np.rint(py).astype(np.int32), 0, S - 1)
        flat = pyi * S + pxi
        near = np.clip((self.d_max - d) / max(self.d_max - self.d_min, 1e-6), 0, 1)
        w = (0.40 + 0.95 * near ** 1.6).astype(np.float32)
        wn = (np.clip((near - 0.55) / 0.45, 0, 1) ** 2 * w).astype(np.float32)
        return flat, w, wn

    def _accumulate(self, flat: np.ndarray, w: np.ndarray) -> np.ndarray:
        S = self.size
        acc = np.zeros((S * S, 3), np.float32)
        for r in range(6):
            a, b = self.bounds[r], self.bounds[r + 1]
            if b <= a:
                continue
            cnt = np.bincount(flat[a:b], weights=w[a:b], minlength=S * S).astype(np.float32)
            acc += np.multiply.outer(cnt, PIP_REGION_COLORS[r] * PIP_REGION_GAIN[r])
        return acc.reshape(S, S, 3)

    def _base(self, idx: int) -> np.ndarray:
        """Static cloud for yaw step ``idx`` (cached, float16): fine layer + softer 'near' layer."""
        hit = self._cache.get(idx)
        if hit is not None:
            return hit
        S = self.size
        flat, w, wn = self._project(math.radians(idx * PIP_YAW_STEP_DEG))
        k = max(self.k, 0.5)
        img = cv2.GaussianBlur(self._accumulate(flat, w), (0, 0), 0.75 * k)
        near = self._accumulate(flat, wn)
        h = cv2.resize(near, (S // 2, S // 2), interpolation=cv2.INTER_AREA)
        h = cv2.GaussianBlur(h, (0, 0), 1.1 * k)
        near = cv2.resize(h, (S, S), interpolation=cv2.INTER_LINEAR) * 4.0
        base = img + 0.55 * near
        if len(self._cache) > 120:
            self._cache.clear()
            self._proj.clear()
        self._cache[idx] = base
        self._proj[idx] = (flat, w)
        return base

    # ---- per-frame ---------------------------------------------------------------------
    def yaw_deg(self, t: float) -> float:
        return PIP_YAW_AMP_DEG * math.sin(2 * math.pi * t / PIP_YAW_PERIOD_S)

    def deposit(self, spiked: np.ndarray) -> None:
        """Add one control step of spikes (bool (N,)) to the glow buffer at the current projection."""
        if self.flat.size == 0:
            return
        s = spiked[self.valid_idx]
        f = self.flat[s]
        if f.size == 0:
            return
        S = self.size
        dep: np.ndarray = np.zeros((S * S, 3), np.float32)
        np.add.at(dep, f, self.spike_cols[s] * self.w[s][:, None])
        dep = dep.reshape(S, S, 3)
        dep = cv2.GaussianBlur(dep, (0, 0), 1.15 * max(self.k, 0.6)) * 3.6
        self.glow += dep

    def decay(self) -> None:
        self.glow *= PIP_DECAY

    def render(self, t: float = 0.0) -> np.ndarray:
        S = self.size
        if self.flat.size == 0:
            img = self.glow
        else:
            f = self.yaw_deg(t) / PIP_YAW_STEP_DEG
            i0 = int(math.floor(f))
            fr = f - i0
            base = cv2.addWeighted(self._base(i0), 1.0 - fr, self._base(i0 + 1), fr, 0.0) if fr > 1e-3 else self._base(i0)
            # spikes deposit at the nearest cached projection (<= 0.125 deg off, sub-pixel)
            self.flat, self.w = self._proj[i0 if fr < 0.5 else i0 + 1]
            img = cv2.add(base, self.glow)  # type: ignore[assignment]
        # two-scale bloom (computed at 1/4 and 1/8 resolution)
        q = cv2.resize(img, (max(S // 4, 8), max(S // 4, 8)), interpolation=cv2.INTER_AREA)
        b1 = cv2.resize(cv2.GaussianBlur(q, (0, 0), 1.4), (S, S), interpolation=cv2.INTER_LINEAR)
        e = cv2.resize(img, (max(S // 8, 4), max(S // 8, 4)), interpolation=cv2.INTER_AREA)
        b2 = cv2.resize(cv2.GaussianBlur(e, (0, 0), 2.0), (S, S), interpolation=cv2.INTER_LINEAR)
        hdr = cv2.addWeighted(img, self.expo, cv2.addWeighted(b1, 0.40, b2, 0.45, 0.0), self.expo, 0.0)
        out = 1.0 - cv2.exp(-np.maximum(hdr, 0.0))   # filmic-ish soft shoulder
        out = np.sqrt(out * np.sqrt(out))          # gamma 0.75: lift mid-tones, keep the black floor
        return cv2.convertScaleAbs(out, alpha=255.0)


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #
class Renderer:
    def __init__(self, args, meta: dict, events: List[dict], spikes: Optional[dict],
                 positions: Optional[np.ndarray], groups: Optional[np.ndarray], stats: Optional[dict]):
        self.args = args
        self.k = 0.5 if args.preview else 1.0
        k = self.k
        self.W, self.H = int(DESIGN_W * k), int(DESIGN_H * k)
        self.frame_step = 2 if args.preview else 1
        self.events = events
        self.spikes = spikes
        self.meta = meta
        self.stats = stats or {}
        ev_end = (events[-1]["t"] + CTRL_DT) if events else 0.0
        dur = ev_end if args.duration is None else min(args.duration, ev_end)
        self.duration = max(dur, 0.5)
        self.n_frames = int(round(self.duration * FPS))
        self.n_neurons = spikes["n"] if spikes else (positions.shape[0] if positions is not None else 166700)

        seed_l = int(meta.get("seed_l", 1)) if meta else 1
        seed_r = int(meta.get("seed_r", 2)) if meta else 2
        ap_kw = {key: meta[key] for key in ("autoplay", "autoplay_amp", "autoplay_hz", "autoplay_whole") if meta and key in meta}
        self.feed_l, self.feed_r = make_pair(seed_l, seed_r, scale=FEED_SCALE * k, **ap_kw)
        pw, ph = self.feed_l.out_w, self.feed_l.out_h

        self.text = TextRenderer(k)
        quad = DEVICE_QUAD * k
        self.bg = build_background(self.W, self.H, k, quad)
        self.device = Device(k, pw, ph, DEVICE_QUAD, self.feed_l.theme["hinge"])
        self.fly = Fly(k, (FLY_CENTER[0] * k, FLY_CENTER[1] * k))
        self.pip = BrainPiP(positions, groups, self.n_neurons, int(PIP_SIZE * k), k)
        self.pip_mask = rounded_mask(int(PIP_SIZE * k), int(PIP_SIZE * k), int(26 * k))
        pc_l, pc_r = self.device.panel_centers()
        self.panel_centers = {"L": pc_l, "R": pc_r}

        # Storyboard timings scale with the episode length.
        f = min(1.0, self.duration / NOMINAL_LEN_S)
        D = self.duration
        self.t_title = (0.0, 4.0 * f)
        self.t_intro_cue = 6.0 * f
        self.t_stats = (D - 12.0 * f, D - 4.0 * f)
        self.t_attrib = (D - 4.0 * f, D)

        # Dynamic state
        self.ev_ptr = 0
        self.swipe_time = {"L": -1e9, "R": -1e9}
        self.posts = {"L": 0, "R": 0}
        self.readout = {"L": 0.0, "R": 0.0}
        self.readout_disp = {"L": 0.0, "R": 0.0}
        self.spike_hist: deque = deque(maxlen=int(3.0 / CTRL_DT))
        self.total_spikes = 0
        self.total_swipes = {"L": 0, "R": 0}
        self.last_step = -1
        # whole-episode totals for the stats card (independent of the frame being drawn)
        self.ep_swipes = {"L": sum(1 for e in events if e.get("swipe") == "L"), "R": sum(1 for e in events if e.get("swipe") == "R")}
        self.ep_spikes = int(sum(int(e.get("spikes", 0)) for e in events))
        self.ep_posts = dict(events[-1].get("posts", {"L": 0, "R": 0})) if events else {"L": 0, "R": 0}
        self.ep_steps = len(events)

    # ---- event application -------------------------------------------------------------
    def apply_events(self, t: float) -> None:
        while self.ev_ptr < len(self.events) and self.events[self.ev_ptr]["t"] <= t + 1e-9:
            e = self.events[self.ev_ptr]
            self.ev_ptr += 1
            sw = e.get("swipe")
            if sw in ("L", "R"):
                (self.feed_l if sw == "L" else self.feed_r).swipe()
                self.swipe_time[sw] = e["t"]
                self.total_swipes[sw] += 1
            ro = e.get("readout") or {}
            self.readout["L"] = float(ro.get("L", 0.0))
            self.readout["R"] = float(ro.get("R", 0.0))
            p = e.get("posts") or {}
            self.posts["L"] = int(p.get("L", self.feed_l.state["posts_consumed"]))
            self.posts["R"] = int(p.get("R", self.feed_r.state["posts_consumed"]))
            n_sp = int(e.get("spikes", 0))
            self.spike_hist.append(n_sp)
            self.total_spikes += n_sp
            step = int(e["step"])
            if self.spikes is not None and 0 <= step < self.spikes["frames_packed"].shape[0]:
                row = np.unpackbits(self.spikes["frames_packed"][step])[: self.spikes["n"]].astype(bool)
                self.pip.deposit(row)
            self.last_step = step

    # ---- overlays ----------------------------------------------------------------------
    def draw_eye_cues(self, frame: np.ndarray, t: float) -> None:
        if t >= self.t_intro_cue:
            return
        a = 1.0 if t < self.t_intro_cue - 1.0 else clamp(self.t_intro_cue - t, 0, 1)
        a *= 0.85
        eyes = self.fly.eye_positions(t)
        k = self.k
        for side, eye in zip(("L", "R"), eyes):
            tgt = self.panel_centers[side]
            colr = COL_L if side == "L" else COL_R
            p0 = np.asarray(eye, np.float32)
            p1 = np.asarray(tgt, np.float32)
            d = p1 - p0
            L = float(np.hypot(*d))
            u = d / max(L, 1e-6)
            x0, y0 = int(min(p0[0], p1[0]) - 30 * k), int(min(p0[1], p1[1]) - 30 * k)
            x1, y1 = int(max(p0[0], p1[0]) + 30 * k), int(max(p0[1], p1[1]) + 30 * k)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(self.W, x1), min(self.H, y1)
            roi = frame[y0:y1, x0:x1]
            over = roi.copy()
            off = np.array([x0, y0], np.float32)
            dash, gap = 16 * k, 11 * k
            phase = (t * 90 * k) % (dash + gap)
            s = -phase
            while s < L:
                a0, a1 = max(s, 22 * k), min(s + dash, L - 26 * k)
                if a1 > a0:
                    q0 = _pt(p0 + u * a0 - off)
                    q1 = _pt(p0 + u * a1 - off)
                    cv2.line(over, q0, q1, tuple(int(c * 0.5) for c in colr), max(1, int(round(9 * k))), cv2.LINE_AA)
                    cv2.line(over, q0, q1, colr, max(1, int(round(3 * k))), cv2.LINE_AA)
                s += dash + gap
            c = _pt(p1 - off)
            cv2.circle(over, c, int(20 * k), colr, max(1, int(round(3 * k))), cv2.LINE_AA)
            cv2.circle(over, c, int(5 * k), colr, -1, cv2.LINE_AA)
            cv2.addWeighted(roi, 1 - a, over, a, 0, dst=roi)

    def draw_pip(self, frame: np.ndarray, t: float = 0.0) -> None:
        k = self.k
        x, y = int(PIP_XY[0] * k), int(PIP_XY[1] * k)
        S = int(PIP_SIZE * k)
        img = self.pip.render(t)
        # dark backing + thin rounded border
        b = max(2, int(3 * k))
        fill_panel(frame, x - b, y - b, x + S + b, y + S + b, int(29 * k), (8, 8, 12), 0.96, border=(74, 80, 98))
        roi = frame[y : y + S, x : x + S]
        cv2.copyTo(img, self.pip_mask, roi)
        # label strip: soft dark gradient instead of a hard bar so the cloud stays visible
        hs = int(64 * k)
        ramp = (1.0 - 0.72 * np.linspace(0, 1, hs, dtype=np.float32) ** 0.7)[:, None, None]
        strip = roi[S - hs : S]
        strip[:] = (strip.astype(np.float32) * ramp).astype(np.uint8)
        label = f"MaleCNS v1.0 · {self.n_neurons:,} neurons"
        self.text.draw(frame, label, PIP_XY[0] + 16, PIP_XY[1] + PIP_SIZE - 38, 24, COL_GREY)
        wl = self.text.draw(frame, "LIVE SPIKES", PIP_XY[0] + PIP_SIZE - 16, PIP_XY[1] + 14, 22, (120, 120, 235), True, align="right")
        cv2.circle(frame, (int((PIP_XY[0] + PIP_SIZE - 30) * k - wl), int((PIP_XY[1] + 28) * k)), int(6 * k), (90, 90, 250), -1, cv2.LINE_AA)

    def draw_top_left(self, frame: np.ndarray, t: float) -> None:
        T = self.text
        T.draw(frame, "MaleCNS v1.0 connectome · LIF spiking network", 40, 70, 22, COL_GREY)
        T.draw(frame, "DN BURST \u2192 SWIPE", 40, 118, 32, COL_WHITE, True)
        for i, side in enumerate(("L", "R")):
            y0 = 178 + i * 74
            colr = COL_L if side == "L" else COL_R
            v = self.readout_disp[side]
            thr = float(self.meta.get("burst_hz", 0) or 0) if self.meta else 0.0
            frac = clamp(v / (3.0 * thr), 0.0, 1.0) if thr > 0 else clamp((v + 1.0) / 5.0, 0.0, 1.0)
            x0, x1 = 88, 580
            fill_panel(frame, int(x0 * self.k), int(y0 * self.k), int(x1 * self.k), int((y0 + 44) * self.k), int(12 * self.k), (30, 30, 36), 0.9)
            if thr > 0:   # threshold tick at one third of the bar
                xt = int((x0 + 4 + (x1 - x0 - 8) / 3.0) * self.k)
                cv2.line(frame, (xt, int((y0 + 6) * self.k)), (xt, int((y0 + 38) * self.k)), (120, 120, 130), max(1, int(2 * self.k)), cv2.LINE_AA)
            w = int((x1 - x0 - 8) * frac)
            if w > 4:
                age = t - self.swipe_time[side]
                flash = clamp(1.0 - age / 0.5, 0, 1)
                cc = tuple(int(c * (1 - flash) + 245 * flash) for c in colr)
                fill_panel(frame, int((x0 + 4) * self.k), int((y0 + 4) * self.k), int((x0 + 4 + w) * self.k), int((y0 + 40) * self.k), int(10 * self.k), cc, 0.95)
            T.draw(frame, side, 40, y0 - 2, 40, colr, True)
            label = f"{max(v, 0.0):.1f} Hz" if thr > 0 else f"{(0.0 if abs(v) < 0.05 else v):+.1f}"
            # dark pill behind the value so it stays legible when the bar is full and bright
            fill_panel(frame, int((x1 - 118) * self.k), int((y0 + 5) * self.k), int((x1 - 6) * self.k), int((y0 + 39) * self.k), int(9 * self.k), (18, 18, 22), 0.85)
            T.draw(frame, label, x1 - 14, y0 + 6, 28, COL_WHITE, True, align="right")
        # legend
        x = 40
        y = 340
        for gi in (1, 2, 3, 4):
            cv2.circle(frame, (int((x + 9) * self.k), int((y + 16) * self.k)), int(7 * self.k), GROUP_COLORS[gi], -1, cv2.LINE_AA)
            w = T.draw(frame, GROUP_NAMES[gi], x + 26, y + 2, 22, COL_GREY)
            x += 26 + w / self.k + 18  # type: ignore[assignment]
        T.draw(frame, "left eye → left feed · right eye → right feed", 40, 392, 24, COL_DIM)
        T.draw(frame, "front-left leg swipes L · front-right swipes R", 40, 426, 24, COL_DIM)

    def draw_hud(self, frame: np.ndarray, t: float) -> None:
        T = self.text
        k = self.k
        pb_l, pb_r = self.device.panel_bottoms()
        # SWIPE flashes under each panel
        for side, pb in (("L", pb_l), ("R", pb_r)):
            age = t - self.swipe_time[side]
            if 0 <= age < 0.6:
                a = clamp(1.0 - age / 0.6, 0, 1)
                colr = COL_L if side == "L" else COL_R
                yy = 1286 - 12 * ease_out(age / 0.6)
                T.draw(frame, "↑ SWIPE", pb[0] / k, yy, 40, colr, True, align="center", alpha=a)
        # POSTS counters
        T.draw(frame, "POSTS L", 60, 1350, 30, COL_L)
        T.draw(frame, f"{self.posts['L']}", 60, 1382, 64, COL_WHITE, True)
        T.draw(frame, "POSTS R", 1020, 1350, 30, COL_R, align="right")
        T.draw(frame, f"{self.posts['R']}", 1020, 1382, 64, COL_WHITE, True, align="right")
        T.draw(frame, "SWIPES", 540, 1350, 30, COL_GREY, align="center")
        T.draw(frame, f"{self.total_swipes['L'] + self.total_swipes['R']}", 540, 1382, 64, COL_WHITE, True, align="center")
        # whole-brain activity
        hist = np.asarray(self.spike_hist, np.float32)
        recent = hist[-8:] if hist.size else np.zeros(1, np.float32)
        sps = float(recent.mean()) / CTRL_DT
        hz = sps / max(self.n_neurons, 1)
        T.draw(frame, "WHOLE-BRAIN ACTIVITY", 60, 1490, 30, COL_GREY)
        T.draw(frame, f"{sps:,.0f}", 60, 1522, 60, COL_WHITE, True)
        wnum = T.measure(f"{sps:,.0f}", 60, True) / k
        T.draw(frame, "spikes / s", 60 + wnum + 18, 1552, 30, COL_GREY)
        T.draw(frame, f"≈ {hz:.2f} Hz / neuron", 1020, 1548, 30, COL_GREY, align="right")
        # sparkline
        x0, y0, x1, y1 = int(60 * k), int(1612 * k), int(1020 * k), int(1750 * k)
        fill_panel(frame, x0, y0, x1, y1, int(18 * k), (14, 14, 18), 0.85, border=(50, 50, 60))
        if hist.size >= 2:
            n_show = self.spike_hist.maxlen or 1
            vals = hist
            vmax = max(float(vals.max()), 1.0)
            xs = x0 + int(14 * k) + (np.arange(vals.size) + (n_show - vals.size)) / (n_show - 1) * (x1 - x0 - int(28 * k))
            ys = y1 - int(12 * k) - vals / vmax * (y1 - y0 - int(30 * k))
            pts = np.int32(np.round(np.stack([xs, ys], axis=1))).reshape(-1, 1, 2)
            poly = np.vstack([pts, np.array([[[pts[-1, 0, 0], y1 - int(6 * k)]], [[pts[0, 0, 0], y1 - int(6 * k)]]], np.int32)])
            roi = frame[y0:y1, x0:x1]
            over = roi.copy()
            cv2.fillPoly(over, [poly - np.array([x0, y0], np.int32)], (90, 70, 40), cv2.LINE_AA)
            cv2.addWeighted(roi, 0.65, over, 0.35, 0, dst=roi)
            cv2.polylines(frame, [pts], False, (210, 190, 120), max(1, int(round(3 * k))), cv2.LINE_AA)
            cv2.circle(frame, tuple(pts[-1, 0]), int(6 * k), COL_WHITE, -1, cv2.LINE_AA)
        T.draw(frame, "spikes per 16 ms step · last 3 s", 80, 1620, 22, COL_DIM)
        T.draw(frame, "eyes see the feeds · legs swipe · nothing scripted", 540, 1800, 30, COL_DIM, align="center")

    # ---- storyboard cards ----------------------------------------------------------------
    def draw_card(self, frame: np.ndarray, blocks: List[Tuple[str, float, Tuple[int, int, int], bool]],
                  alpha: float, y_center: float, max_width: float = 900, pad: float = 48) -> None:
        """Centered translucent card; ``blocks`` = [(text, size, color, bold), ...] wrapped to width."""
        if alpha <= 0.001:
            return
        T = self.text
        lines: List[Tuple[str, float, Tuple[int, int, int], bool]] = []
        total_h = 0.0
        for text, size, color, bold in blocks:
            if text == "":
                lines.append(("", size, color, bold))
                total_h += size
                continue
            for ln in T.wrap(text, size, max_width - 2 * pad, bold):
                lines.append((ln, size, color, bold))
                total_h += T.line_height(size)
        card_h = total_h + 2 * pad
        x0 = (DESIGN_W - max_width) / 2
        y0 = y_center - card_h / 2
        k = self.k
        fill_panel(frame, int(x0 * k), int(y0 * k), int((x0 + max_width) * k), int((y0 + card_h) * k), int(36 * k),
                   (10, 10, 14), 0.82 * alpha, border=(int(80 * alpha), int(80 * alpha), int(92 * alpha)))
        y = y0 + pad
        for ln, size, color, bold in lines:
            if ln:
                T.draw(frame, ln, DESIGN_W / 2, y, size, color, bold, align="center", alpha=alpha)
                y += T.line_height(size)
            else:
                y += size

    def draw_cards(self, frame: np.ndarray, t: float) -> None:
        a = card_alpha(t, *self.t_title)
        if a > 0:
            # Title sits over the HUD band so the fly and the feeds are visible from frame one.
            self.draw_card(frame, [(TITLE, 68, COL_WHITE, True), ("", 18, COL_WHITE, False), (SUBTITLE, 34, COL_GREY, False)],
                           a, y_center=1585, max_width=980)
        a = card_alpha(t, *self.t_stats)
        if a > 0:
            blocks: List[Tuple[str, float, Tuple[int, int, int], bool]] = [
                (str(self.stats.get("headline", STATS_HEADLINE.format(n=int(self.n_neurons), e=float(self.meta.get("n_edges", 6242085)) / 1e6 if self.meta else 6.24))), 40, COL_WHITE, True), ("", 20, COL_WHITE, False)]
            items = self.stats_items()
            for key, val in items:
                blocks.append((f"{key}: {val}", 34, COL_GREY, False))
            self.draw_card(frame, blocks, a, y_center=1000, max_width=940)
        a = card_alpha(t, self.t_attrib[0], self.t_attrib[1] + 1.0, fade_out=0.0)
        if a > 0:
            self.draw_card(frame, [("CREDITS", 30, COL_GREY, True), ("", 12, COL_WHITE, False), (ATTRIBUTION, 30, COL_WHITE, False)],
                           a, y_center=1000, max_width=960, pad=40)

    def stats_items(self) -> List[Tuple[str, str]]:
        items: List[Tuple[str, str]] = []
        for key, val in self.stats.items():
            if key == "headline":
                continue
            items.append((str(key), f"{val:,}" if isinstance(val, (int, float)) and not isinstance(val, bool) else str(val)))
        if not any(k.lower().startswith("total swipes") for k, _ in items):
            items.append(("total swipes", f"{self.ep_swipes['L'] + self.ep_swipes['R']:,}  (L {self.ep_swipes['L']} · R {self.ep_swipes['R']})"))
        if not any(k.lower().startswith("total spikes") for k, _ in items):
            items.append(("total spikes", f"{self.ep_spikes:,}"))
        if not any(k.lower().startswith("posts") for k, _ in items):
            items.append(("posts scrolled", f"L {self.ep_posts['L']} · R {self.ep_posts['R']}"))
        if not any(k.lower().startswith("simulated") for k, _ in items):
            items.append(("simulated time", f"{self.ep_steps * CTRL_DT:.1f} s of brain time"))
        return items

    # ---- frame ----------------------------------------------------------------------------
    def render_frame(self, idx: int) -> np.ndarray:
        t = idx / FPS
        self.apply_events(t)
        for side in ("L", "R"):
            self.readout_disp[side] += (self.readout[side] - self.readout_disp[side]) * 0.45
        frame = self.bg.copy()
        self.device.draw(frame, self.feed_l.render(), self.feed_r.render())
        self.draw_eye_cues(frame, t)
        swipe_age = {}
        for side in ("L", "R"):
            age = t - self.swipe_time[side]
            swipe_age[side] = age if 0 <= age < SWIPE_ANIM_S else None
        self.fly.draw(frame, t, swipe_age)
        self.pip.decay()
        self.draw_pip(frame, t)
        self.draw_top_left(frame, t)
        self.draw_hud(frame, t)
        self.draw_cards(frame, t)
        return frame

    def step_feeds(self) -> None:
        self.feed_l.step(1.0 / FPS)
        self.feed_r.step(1.0 / FPS)

    def run(self) -> None:
        import imageio.v2 as imageio
        args = self.args
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        out_fps = FPS / self.frame_step
        stills = set()
        if args.stills:
            stills = {int(round(float(s) * FPS)) for s in args.stills.split(",") if s.strip()}
        png_dir = args.png_dir
        if png_dir:
            os.makedirs(png_dir, exist_ok=True)
        writer = imageio.get_writer(args.out, fps=out_fps, codec="libx264", quality=8, pixelformat="yuv420p",
                                    macro_block_size=1, ffmpeg_params=["-preset", args.preset])
        t_start = time.perf_counter()
        n_written = 0
        next_report = 5.0
        print(f"rendering {self.n_frames} frames ({self.duration:.1f} s) at {self.W}x{self.H}, "
              f"writing every {self.frame_step} frame(s) to {args.out}", flush=True)
        try:
            for idx in range(self.n_frames):
                self.step_feeds()
                if idx % self.frame_step != 0 and idx not in stills:
                    # still advance sim state / event application for skipped frames
                    self.apply_events(idx / FPS)
                    self.pip.decay()
                    continue
                frame = self.render_frame(idx)
                if idx in stills:
                    tsec = idx / FPS
                    name = f"render_still_{tsec:g}s.png"
                    cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(args.out)), name), frame)
                if idx % self.frame_step == 0:
                    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    n_written += 1
                    if png_dir and (idx // self.frame_step) % max(1, args.png_every) == 0:
                        cv2.imwrite(os.path.join(png_dir, f"frame_{idx:06d}.png"), frame)
                t_video = (idx + 1) / FPS
                if t_video >= next_report or idx == self.n_frames - 1:
                    now = time.perf_counter()
                    fps_wall = n_written / max(now - t_start, 1e-6)
                    eta = (self.n_frames - idx - 1) / self.frame_step / max(fps_wall, 1e-6)
                    print(f"  t={t_video:5.1f}s  frames={n_written}  {fps_wall:5.1f} fps  eta {eta:5.1f}s", flush=True)
                    next_report += 5.0
        finally:
            writer.close()
        elapsed = time.perf_counter() - t_start
        print(f"wrote {args.out}: {n_written} frames in {elapsed:.1f}s ({n_written / max(elapsed, 1e-6):.1f} fps)")


def card_alpha(t: float, t0: float, t1: float, fade_in: float = 0.45, fade_out: float = 0.6) -> float:
    if t < t0 or t >= t1:
        return 0.0
    a_in = ease_in_out((t - t0) / fade_in) if fade_in > 0 else 1.0
    a_out = ease_in_out((t1 - t) / fade_out) if fade_out > 0 else 1.0
    return min(a_in, a_out)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", default=os.path.join(ROOT, "out", "events.jsonl"))
    ap.add_argument("--spikes", default=os.path.join(ROOT, "out", "spikes.npz"))
    ap.add_argument("--positions", default=os.path.join(ROOT, "data", "graph", "positions.npy"))
    ap.add_argument("--groups", default=os.path.join(ROOT, "data", "graph", "groups.npy"))
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "final.mp4"))
    ap.add_argument("--duration", type=float, default=None, help="render at most this many seconds")
    ap.add_argument("--fixture", action="store_true", help="generate a synthetic episode when inputs are missing")
    ap.add_argument("--fixture-duration", type=float, default=50.0)
    ap.add_argument("--preview", action="store_true", help="540x960, every other frame (30 fps output)")
    ap.add_argument("--stats", default=None, help="JSON file with extra key/value pairs for the stats card")
    ap.add_argument("--png-dir", default=None, help="also dump PNG frames here")
    ap.add_argument("--png-every", type=int, default=60, help="dump every N written frames")
    ap.add_argument("--stills", default=None, help="comma-separated video times (s) saved as render_still_*.png next to --out")
    ap.add_argument("--preset", default="fast", help="x264 preset")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ev_path, sp_path, pos_path, grp_path = args.events, args.spikes, args.positions, args.groups
    missing = [p for p in (ev_path, sp_path, pos_path) if not os.path.exists(p)]
    if missing:
        if not args.fixture:
            print("missing inputs:\n  " + "\n  ".join(missing) + "\n(pass --fixture to render a synthetic episode)", file=sys.stderr)
            return 2
        out_dir = os.path.dirname(os.path.abspath(args.out))
        fx = {"events": os.path.join(out_dir, "fixture_events.jsonl"), "spikes": os.path.join(out_dir, "fixture_spikes.npz"),
              "positions": os.path.join(out_dir, "fixture_positions.npy"), "groups": os.path.join(out_dir, "fixture_groups.npy")}
        if all(os.path.exists(v) for v in fx.values()):
            print(f"inputs missing ({', '.join(os.path.basename(m) for m in missing)}); reusing fixture in {out_dir}")
        else:
            print(f"inputs missing ({', '.join(os.path.basename(m) for m in missing)}); generating fixture in {out_dir}")
            fx = make_fixture(out_dir, duration=args.fixture_duration)
        if not os.path.exists(ev_path):
            ev_path = fx["events"]
        if not os.path.exists(sp_path):
            sp_path = fx["spikes"]
        if not os.path.exists(pos_path):
            pos_path, grp_path = fx["positions"], fx["groups"]
    meta, events = load_events(ev_path)
    if not events:
        print("no events", file=sys.stderr)
        return 2
    spikes = load_spikes(sp_path)
    positions = np.load(pos_path) if os.path.exists(pos_path) else None
    groups = np.load(grp_path) if grp_path and os.path.exists(grp_path) else None
    if positions is not None and spikes is not None and positions.shape[0] != spikes["n"]:
        print(f"warning: positions rows ({positions.shape[0]}) != spikes n ({spikes['n']}); aligning by index", file=sys.stderr)
    stats = None
    if args.stats:
        with open(args.stats) as f:
            stats = json.load(f)
        if not isinstance(stats, dict):
            sys.exit(f"--stats must be a JSON object of label -> value, got {type(stats).__name__}")
    r = Renderer(args, meta, events, spikes, positions, groups, stats)
    r.run()
    return 0



def _pt(v) -> Tuple[int, int]:
    """Round a 2-vector to an integer pixel tuple for OpenCV."""
    r = np.round(np.asarray(v, dtype=np.float64))
    return int(r[0]), int(r[1])


if __name__ == "__main__":
    sys.exit(main())

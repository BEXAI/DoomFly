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
from typing import Dict, List, Optional, Sequence, Tuple

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
DEVICE_QUAD = np.float32([[68, 530], [1012, 530], [1068, 1265], [12, 1265]])
DEVICE_BEVEL = 26
DEVICE_HINGE = 14

FLY_CENTER = (540, 945)    # thorax centre (design px)
SWIPE_ANIM_S = 0.42        # reach 0.10 s, drag 0.22 s, return 0.10 s
SWIPE_DRAG_PX = 130        # tarsus travel (forward = toward the head = content scrolls up)

PIP_XY, PIP_SIZE = (640, 60), 400
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
STATS_HEADLINE = "166,691 neurons · 6.2 M connections (≥5 synapses) · 0 lines of hand-written behaviour"
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
                f = self._ImageFont.load_default(size=px)
            self._fonts[key] = f
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
    np.savez_compressed(paths["spikes"], **arrs)
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
    out = {"frames_packed": d["frames_packed"], "n": int(d["n"]), "total": d["total"]}
    return out


# --------------------------------------------------------------------------- #
# Background
# --------------------------------------------------------------------------- #
def build_background(W: int, H: int, k: float, quad: np.ndarray) -> np.ndarray:
    """Dark matte desk with gradient, vignette, grain and the device drop shadow."""
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    xs = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    top = np.array([34, 30, 27], np.float32)   # BGR warm dark
    bot = np.array([16, 15, 14], np.float32)
    col = top[None, None, :] * (1 - ys[..., None]) + bot[None, None, :] * ys[..., None]  # (H,1,3)
    yy = (ys - 0.5) * 2
    r2 = xs ** 2 * 0.9 + yy ** 2
    vig = 1.0 - 0.55 * np.clip(r2, 0, 1)                                                  # (H,W)
    bg = col * vig[..., None]                                                              # (H,W,3)
    rng = np.random.default_rng(3)
    grain = rng.normal(0, 3.5, (H, W)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), 0.8 * max(k, 0.5))
    bg += grain[..., None]
    # Soft spot light behind the device.
    cx, cy = W * 0.5, H * 0.47
    spot = np.exp(-(((np.arange(W) - cx) / (W * 0.55)) ** 2)[None, :] - (((np.arange(H) - cy) / (H * 0.28)) ** 2)[:, None])
    bg += (spot[..., None] * np.array([26, 24, 22], np.float32))
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    # Drop shadow of the device.
    sh = np.zeros((H, W), np.float32)
    q = quad.copy()
    q[:, 1] += 22 * k
    c = q.mean(axis=0)
    q = (q - c) * np.array([1.03, 1.04], np.float32) + c
    cv2.fillPoly(sh, [np.int32(q).reshape(-1, 1, 2)], 1.0)
    sh = cv2.GaussianBlur(sh, (0, 0), 28 * k)
    bg = (bg.astype(np.float32) * (1.0 - 0.75 * sh[..., None])).astype(np.uint8)
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
        src = np.float32([[0, 0], [Wc, 0], [Wc, Hc], [0, Hc]])
        dst = quad - np.float32([self.roi[0], self.roi[1]])
        self.Hm = cv2.getPerspectiveTransform(src, dst)
        self.roi_size = (self.roi[2] - self.roi[0], self.roi[3] - self.roi[1])
        wm = cv2.warpPerspective(self.mask, self.Hm, self.roi_size, flags=cv2.INTER_LINEAR)
        self.wmask = (wm >= 128).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(self.wmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.outline = cnts
        self.quad = quad

    def to_frame(self, pts_composite: Sequence[Tuple[float, float]]) -> np.ndarray:
        p = np.float32(pts_composite).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(p, self.Hm).reshape(-1, 2)
        return out + np.float32([self.roi[0], self.roi[1]])

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
class Fly:
    """Top-down stylised Drosophila: pre-rendered body sprite + procedural legs/antennae."""

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
        self.leg_color = (28, 36, 54)
        self.leg_edge = (10, 12, 18)
        self.rng = np.random.default_rng(11)
        self.twitch_phase = self.rng.uniform(0, 6.28, 2)

    # ---- sprite -----------------------------------------------------------------
    def _build_sprite(self) -> Tuple[np.ndarray, np.ndarray]:
        k = self.k
        ss = 2
        s = k * ss
        cw, ch = int(240 * s), int(420 * s)
        ox, oy = self.canvas_origin
        col = np.zeros((ch, cw, 3), np.float32)
        alp = np.zeros((ch, cw), np.float32)

        def P(x: float, y: float) -> Tuple[int, int]:
            return int(round((x + ox) * s)), int(round((y + oy) * s))

        def R(v: float) -> int:
            return max(1, int(round(v * s)))

        def paint(mask: np.ndarray, color, opacity: float = 1.0) -> None:
            a = (mask.astype(np.float32) / 255.0) * opacity
            col[:] = col * (1 - a[..., None]) + np.float32(color)[None, None, :] * a[..., None]
            alp[:] = alp + a * (1 - alp)

        def shape_mask() -> np.ndarray:
            return np.zeros((ch, cw), np.uint8)

        # --- abdomen: tapered ovoid with dark bands ------------------------------
        ys = np.linspace(22, 214, 40)
        left, right = [], []
        for y in ys:
            u = (y - 118) / 96.0
            w = 50 * math.sqrt(max(0.0, 1 - u * u)) * (1 - 0.30 * (y - 22) / 192)
            left.append(P(-w, y))
            right.append(P(w, y))
        abd_poly = np.int32(left + right[::-1]).reshape(-1, 1, 2)
        m = shape_mask()
        cv2.fillPoly(m, [abd_poly], 255, cv2.LINE_AA)
        paint(cv2.dilate(m, np.ones((R(4), R(4)), np.uint8)), (24, 32, 48))  # dark outline
        paint(m, (72, 124, 178))  # tan
        # shading: lighter middle stripe
        hl = shape_mask()
        cv2.fillPoly(hl, [abd_poly], 255, cv2.LINE_AA)
        hl2 = shape_mask()
        cv2.ellipse(hl2, P(-8, 100), (R(22), R(80)), 0, 0, 360, 255, -1, cv2.LINE_AA)
        paint(cv2.bitwise_and(hl, hl2), (110, 160, 205), 0.35)
        for yb in (54, 88, 122, 156, 188):
            band = shape_mask()
            cv2.rectangle(band, P(-70, yb - 7), P(70, yb + 7), 255, -1)
            band = cv2.bitwise_and(band, m)
            paint(band, (26, 40, 66), 0.92)
        # --- thorax -------------------------------------------------------------
        m = shape_mask()
        cv2.ellipse(m, P(0, 0), (R(56), R(52)), 0, 0, 360, 255, -1, cv2.LINE_AA)
        paint(cv2.dilate(m, np.ones((R(4), R(4)), np.uint8)), (24, 32, 48))
        paint(m, (86, 142, 194))
        hl = shape_mask()
        cv2.ellipse(hl, P(-14, -14), (R(28), R(24)), -20, 0, 360, 255, -1, cv2.LINE_AA)
        paint(hl, (130, 178, 220), 0.35)
        # scutellum (dark rounded bump at the rear of the thorax)
        sc = shape_mask()
        cv2.ellipse(sc, P(0, 40), (R(24), R(14)), 0, 0, 360, 255, -1, cv2.LINE_AA)
        paint(sc, (48, 78, 120), 0.9)
        # bristles
        for i in range(-3, 4):
            x = i * 13
            cv2.line(col, P(x, -40), P(x * 1.15, -58), (30, 40, 60), R(1.2), cv2.LINE_AA)
        # --- wings (translucent, over the abdomen) ------------------------------
        for side in (-1, 1):
            wm = shape_mask()
            cx, cy = P(side * 40, 118)
            cv2.ellipse(wm, (cx, cy), (R(30), R(134)), side * 9, 0, 360, 255, -1, cv2.LINE_AA)
            wm_in = cv2.erode(wm, np.ones((R(2.5), R(2.5)), np.uint8))
            paint(cv2.subtract(wm, wm_in), (170, 178, 196), 0.55)  # edge
            paint(wm_in, (214, 222, 236), 0.34)
            # veins
            base = P(side * 20, -4)
            for tx, ty in ((side * 34, 248), (side * 54, 236), (side * 68, 200), (side * 16, 240)):
                tip = P(tx, ty)
                vm = shape_mask()
                cv2.line(vm, base, tip, 255, R(1.6), cv2.LINE_AA)
                paint(cv2.bitwise_and(vm, wm), (120, 130, 155), 0.55)
            vm = shape_mask()
            cv2.line(vm, P(side * 30, 90), P(side * 62, 110), 255, R(1.4), cv2.LINE_AA)
            cv2.line(vm, P(side * 34, 150), P(side * 66, 160), 255, R(1.4), cv2.LINE_AA)
            paint(cv2.bitwise_and(vm, wm), (120, 130, 155), 0.5)
        # --- head ---------------------------------------------------------------
        m = shape_mask()
        cv2.circle(m, P(0, -78), R(36), 255, -1, cv2.LINE_AA)
        paint(cv2.dilate(m, np.ones((R(4), R(4)), np.uint8)), (24, 32, 48))
        paint(m, (78, 118, 168))
        # neck
        nk = shape_mask()
        cv2.rectangle(nk, P(-14, -50), P(14, -38), 255, -1)
        paint(nk, (40, 60, 90))
        # proboscis
        pm = shape_mask()
        cv2.ellipse(pm, P(0, -108), (R(8), R(12)), 0, 0, 360, 255, -1, cv2.LINE_AA)
        paint(pm, (44, 60, 92))
        # ocelli
        for ox_, oy_ in ((0, -100), (-7, -92), (7, -92)):
            om = shape_mask()
            cv2.circle(om, P(ox_, oy_), R(2.4), 255, -1, cv2.LINE_AA)
            paint(om, (60, 40, 120))
        # compound eyes with facets
        ex, ey = self.EYE_CENTER
        for side in (-1, 1):
            em = shape_mask()
            cx, cy = P(side * ex, ey)
            cv2.ellipse(em, (cx, cy), (R(21), R(29)), side * 14, 0, 360, 255, -1, cv2.LINE_AA)
            paint(cv2.dilate(em, np.ones((R(3), R(3)), np.uint8)), (20, 20, 90))
            paint(em, (36, 40, 214))
            # facets: hex grid of darker dots clipped to the eye
            fm = shape_mask()
            step = 4.6
            row = 0
            y = ey - 34
            while y < ey + 34:
                x = side * ex - 26 + (step / 2 if row % 2 else 0)
                while x < side * ex + 26:
                    cv2.circle(fm, P(x, y), R(1.5), 255, -1, cv2.LINE_AA)
                    x += step
                y += step * 0.87
                row += 1
            paint(cv2.bitwise_and(fm, em), (14, 14, 132), 0.85)
            # specular highlight
            hm = shape_mask()
            cv2.ellipse(hm, P(side * (ex - 6), ey - 12), (R(5), R(9)), side * 14, 0, 360, 255, -1, cv2.LINE_AA)
            paint(cv2.bitwise_and(hm, em), (230, 230, 245), 0.55)

        bgr = np.clip(col, 0, 255).astype(np.uint8)
        a = np.clip(alp * 255, 0, 255).astype(np.uint8)
        out_w, out_h = int(240 * k), int(420 * k)
        bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        a = cv2.resize(a, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return bgr, a

    # ---- per-frame ------------------------------------------------------------------
    def eye_positions(self, t: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        bob = self._bob(t)
        ex, ey = self.EYE_CENTER
        k = self.k
        return ((self.cx - ex * k, self.cy + (ey + bob) * k), (self.cx + ex * k, self.cy + (ey + bob) * k))

    def _bob(self, t: float) -> float:
        return 1.6 * math.sin(2 * math.pi * 1.1 * t) + 0.6 * math.sin(2 * math.pi * 2.7 * t + 1.0)

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

    def draw(self, frame: np.ndarray, t: float, swipe_age: Dict[str, Optional[float]]) -> None:
        k = self.k
        bob = self._bob(t)
        ox, oy = self.cx, self.cy + bob * k

        def F(x: float, y: float, with_bob: bool = True) -> Tuple[int, int]:
            return int(round(ox + x * k)), int(round((oy if with_bob else self.cy) + y * k))

        # ---- legs (under the body) ----
        for name in ("hind", "mid", "front"):
            ax, ay = self.LEG_ATTACH[name]
            for side in (-1, 1):
                A = np.float32(F(side * ax, ay))
                if name == "front":
                    age = swipe_age["L"] if side < 0 else swipe_age["R"]
                    tx, ty = self._front_tip(side, age)
                    T = np.float32(F(tx, ty, with_bob=False))
                else:
                    rx, ry = self.LEG_REST[name]
                    T = np.float32(F(side * rx, ry, with_bob=False))
                d = T - A
                L = float(np.hypot(*d)) + 1e-6
                u = d / L
                n = np.float32([-u[1], u[0]])
                if n[0] * side + n[1] * (-0.45) < 0:
                    n = -n
                bend = 0.36 if name != "front" else 0.30
                knee = A + u * (0.46 * L) + n * (bend * L)
                ankle = A + u * (0.86 * L) + n * (0.10 * L)
                pts = [A, knee, ankle, T]
                th = self.LEG_THICK[name]
                for i in range(3):
                    p0, p1 = tuple(np.int32(np.round(pts[i]))), tuple(np.int32(np.round(pts[i + 1])))
                    cv2.line(frame, p0, p1, self.leg_edge, max(1, int(round((th[i] + 2.5) * k))), cv2.LINE_AA)
                for i in range(3):
                    p0, p1 = tuple(np.int32(np.round(pts[i]))), tuple(np.int32(np.round(pts[i + 1])))
                    cv2.line(frame, p0, p1, self.leg_color, max(1, int(round(th[i] * k))), cv2.LINE_AA)
                cv2.circle(frame, tuple(np.int32(np.round(knee))), max(1, int(round(4.2 * k))), (40, 52, 76), -1, cv2.LINE_AA)
                cv2.circle(frame, tuple(np.int32(np.round(ankle))), max(1, int(round(3.2 * k))), (40, 52, 76), -1, cv2.LINE_AA)
                # tarsus contact ring while dragging
                if name == "front":
                    age = swipe_age["L"] if side < 0 else swipe_age["R"]
                    if age is not None and 0.08 <= age < 0.36:
                        a = 1.0 - abs((age - 0.22) / 0.14)
                        a = clamp(a, 0, 1)
                        colr = COL_L if side < 0 else COL_R
                        cc = tuple(int(c * a + 40 * (1 - a)) for c in colr)
                        cv2.circle(frame, tuple(np.int32(np.round(T))), int(round((16 + 14 * (1 - a)) * k)), cc, max(1, int(round(3.5 * k))), cv2.LINE_AA)

        # ---- antennae (idle twitch) ----
        for i, side in enumerate((-1, 1)):
            ph = self.twitch_phase[i]
            tw = 0.18 * math.sin(2 * math.pi * 0.6 * t + ph) + 0.10 * math.sin(2 * math.pi * 4.1 * t + ph * 2) * max(0.0, math.sin(2 * math.pi * 0.23 * t + ph))
            ang = -math.pi / 2 + side * (0.62 + tw)
            base = F(side * 12, -104)
            mid = F(side * 12 + 16 * math.cos(ang), -104 + 16 * math.sin(ang))
            tip = F(side * 12 + 30 * math.cos(ang) + side * 4, -104 + 30 * math.sin(ang) - 3)
            cv2.line(frame, base, mid, (30, 40, 62), max(1, int(round(3 * k))), cv2.LINE_AA)
            cv2.line(frame, mid, tip, (30, 40, 62), max(1, int(round(2 * k))), cv2.LINE_AA)
            # arista (feathery tip)
            for j in range(3):
                a2 = ang + side * (0.5 + 0.35 * j)
                p0 = mid
                p1 = (int(mid[0] + 9 * k * math.cos(a2)), int(mid[1] + 9 * k * math.sin(a2)))
                cv2.line(frame, p0, p1, (40, 52, 76), max(1, int(round(1.2 * k))), cv2.LINE_AA)

        # ---- body sprite ----
        sx = int(round(ox - self.canvas_origin[0] * k))
        sy = int(round(oy - self.canvas_origin[1] * k))
        alpha_blit(frame, self.sprite_bgr, self.sprite_a, sx, sy)


# --------------------------------------------------------------------------- #
# Brain picture-in-picture
# --------------------------------------------------------------------------- #
class BrainPiP:
    """Point cloud of soma positions; spikes lift brightness/size with exponential glow decay."""

    def __init__(self, positions: Optional[np.ndarray], groups: Optional[np.ndarray], n: int, size: int, k: float):
        self.size = size
        self.k = k
        self.n = n
        self.glow = np.zeros((size, size, 3), np.float32)
        if positions is None:
            self.valid_idx = np.zeros(0, np.int64)
            self.flat = np.zeros(0, np.int64)
            self.base = np.zeros((size, size, 3), np.float32)
            self.spike_cols = np.zeros((0, 3), np.float32)
            self.counts = {}
            return
        pos = positions[:n] if positions.shape[0] >= n else np.vstack([positions, np.full((n - positions.shape[0], 3), np.nan, np.float32)])
        if groups is None:
            groups = np.zeros(n, np.int8)
        groups = groups[:n] if groups.shape[0] >= n else np.concatenate([groups, np.zeros(n - groups.shape[0], np.int8)])
        valid = np.isfinite(pos).all(axis=1)
        self.valid_idx = np.flatnonzero(valid)
        P = pos[valid].astype(np.float64)
        P -= P.mean(axis=0)
        yaw, tilt = math.radians(18.0), math.radians(-10.0)
        # Mirror x: the volume's x axis runs from the fly's right to its left (anterior view),
        # while the fly in the scene is drawn head-up with its left at the viewer's left.
        x, y, z = -P[:, 0], P[:, 1], P[:, 2]
        x1 = x * math.cos(yaw) + z * math.sin(yaw)
        z1 = -x * math.sin(yaw) + z * math.cos(yaw)
        y1 = y * math.cos(tilt) - z1 * math.sin(tilt)
        pad = size * 0.07
        span_x = max(np.ptp(x1), 1e-6)
        span_y = max(np.ptp(y1), 1e-6)
        scale = (size - 2 * pad) / max(span_x, span_y)
        px = (x1 - x1.min()) * scale
        py = (y1 - y1.min()) * scale
        px += (size - np.ptp(px)) / 2
        py += (size - np.ptp(py)) / 2
        px = np.clip(np.round(px).astype(np.int64), 0, size - 1)
        py = np.clip(np.round(py).astype(np.int64), 0, size - 1)
        self.flat = py * size + px
        g = groups[valid].astype(np.int64)
        g[(g < 0) | (g > 4)] = 0
        self.groups_valid = g
        cols = np.array([GROUP_COLORS[i] for i in range(5)], np.float32) / 255.0
        # Base cloud: per-group density -> soft saturating brightness.
        base = np.zeros((size * size, 3), np.float32)
        self.counts = {}
        for gi in range(5):
            sel = g == gi
            if not sel.any():
                continue
            cnt = np.bincount(self.flat[sel], minlength=size * size).astype(np.float32)
            self.counts[gi] = int(sel.sum())
            inten = 1.0 - np.exp(-cnt / 1.6)
            gain = 0.55 if gi == 0 else 0.8
            base += inten[:, None] * cols[gi][None, :] * gain
        base = base.reshape(size, size, 3)
        base = cv2.GaussianBlur(base, (0, 0), 0.6)
        self.base = np.clip(base, 0, 1)
        # Spike colours: group colour lifted toward white.
        self.spike_cols = (cols * 0.55 + 0.45)[g]  # (n_valid, 3): group colour lifted toward white
        self.spike_cols[g == 0] = np.float32([0.95, 0.95, 0.98])

    def deposit(self, spiked: np.ndarray) -> None:
        """Add one control step of spikes (bool (N,)) to the glow buffer."""
        if self.flat.size == 0:
            return
        s = spiked[self.valid_idx]
        f = self.flat[s]
        if f.size == 0:
            return
        dep = np.zeros((self.size * self.size, 3), np.float32)
        np.add.at(dep, f, self.spike_cols[s])
        dep = dep.reshape(self.size, self.size, 3)
        dep = cv2.GaussianBlur(dep, (0, 0), 1.1 * max(self.k, 0.6)) * 4.0
        self.glow += dep

    def decay(self) -> None:
        self.glow *= PIP_DECAY

    def render(self) -> np.ndarray:
        img = np.clip(self.base + self.glow, 0, 1)
        return (img * 255).astype(np.uint8)


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
        self.n_neurons = spikes["n"] if spikes else (positions.shape[0] if positions is not None else 166691)

        seed_l = int(meta.get("seed_l", 1)) if meta else 1
        seed_r = int(meta.get("seed_r", 2)) if meta else 2
        self.feed_l, self.feed_r = make_pair(seed_l, seed_r, scale=FEED_SCALE * k)
        pw, ph = self.feed_l.out_w, self.feed_l.out_h

        self.text = TextRenderer(k)
        quad = DEVICE_QUAD * k
        self.bg = build_background(self.W, self.H, k, quad)
        self.device = Device(k, pw, ph, DEVICE_QUAD, self.feed_l.theme["hinge"])
        self.fly = Fly(k, (FLY_CENTER[0] * k, FLY_CENTER[1] * k))
        self.pip = BrainPiP(positions, groups, self.n_neurons, int(PIP_SIZE * k), k)
        self.pip_mask = rounded_mask(int(PIP_SIZE * k), int(PIP_SIZE * k), int(28 * k))
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
            p0 = np.float32(eye)
            p1 = np.float32(tgt)
            d = p1 - p0
            L = float(np.hypot(*d))
            u = d / max(L, 1e-6)
            x0, y0 = int(min(p0[0], p1[0]) - 30 * k), int(min(p0[1], p1[1]) - 30 * k)
            x1, y1 = int(max(p0[0], p1[0]) + 30 * k), int(max(p0[1], p1[1]) + 30 * k)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(self.W, x1), min(self.H, y1)
            roi = frame[y0:y1, x0:x1]
            over = roi.copy()
            off = np.float32([x0, y0])
            dash, gap = 16 * k, 11 * k
            phase = (t * 90 * k) % (dash + gap)
            s = -phase
            while s < L:
                a0, a1 = max(s, 22 * k), min(s + dash, L - 26 * k)
                if a1 > a0:
                    q0 = tuple(np.int32(np.round(p0 + u * a0 - off)))
                    q1 = tuple(np.int32(np.round(p0 + u * a1 - off)))
                    cv2.line(over, q0, q1, tuple(int(c * 0.5) for c in colr), max(1, int(round(9 * k))), cv2.LINE_AA)
                    cv2.line(over, q0, q1, colr, max(1, int(round(3 * k))), cv2.LINE_AA)
                s += dash + gap
            c = tuple(np.int32(np.round(p1 - off)))
            cv2.circle(over, c, int(20 * k), colr, max(1, int(round(3 * k))), cv2.LINE_AA)
            cv2.circle(over, c, int(5 * k), colr, -1, cv2.LINE_AA)
            cv2.addWeighted(roi, 1 - a, over, a, 0, dst=roi)

    def draw_pip(self, frame: np.ndarray) -> None:
        k = self.k
        x, y = int(PIP_XY[0] * k), int(PIP_XY[1] * k)
        S = int(PIP_SIZE * k)
        img = self.pip.render()
        # dark backing + border
        fill_panel(frame, x - int(3 * k), y - int(3 * k), x + S + int(3 * k), y + S + int(3 * k), int(30 * k), (12, 12, 16), 0.9, border=(70, 70, 80))
        roi = frame[y : y + S, x : x + S]
        cv2.copyTo(img, self.pip_mask, roi)
        cv2.rectangle(frame, (x, y + S - int(46 * k)), (x + S, y + S), (0, 0, 0), -1)
        label = f"MaleCNS v1.0 · {self.n_neurons:,} neurons"
        self.text.draw(frame, label, PIP_XY[0] + 16, PIP_XY[1] + PIP_SIZE - 38, 24, COL_GREY)
        wl = self.text.draw(frame, "LIVE SPIKES", PIP_XY[0] + PIP_SIZE - 16, PIP_XY[1] + 14, 22, (120, 120, 235), True, align="right")
        cv2.circle(frame, (int((PIP_XY[0] + PIP_SIZE - 30) * k - wl), int((PIP_XY[1] + 28) * k)), int(6 * k), (90, 90, 250), -1, cv2.LINE_AA)

    def draw_top_left(self, frame: np.ndarray, t: float) -> None:
        T = self.text
        T.draw(frame, "MaleCNS v1.0 connectome · LIF spiking network", 40, 68, 26, COL_GREY)
        T.draw(frame, "DN BURST \u2192 SWIPE", 40, 118, 32, COL_WHITE, True)
        for i, side in enumerate(("L", "R")):
            y0 = 178 + i * 74
            colr = COL_L if side == "L" else COL_R
            v = self.readout_disp[side]
            frac = clamp((v + 1.0) / 5.0, 0.0, 1.0)
            x0, x1 = 88, 596
            fill_panel(frame, int(x0 * self.k), int(y0 * self.k), int(x1 * self.k), int((y0 + 44) * self.k), int(12 * self.k), (30, 30, 36), 0.9)
            w = int((x1 - x0 - 8) * frac)
            if w > 4:
                age = t - self.swipe_time[side]
                flash = clamp(1.0 - age / 0.5, 0, 1)
                cc = tuple(int(c * (1 - flash) + 245 * flash) for c in colr)
                fill_panel(frame, int((x0 + 4) * self.k), int((y0 + 4) * self.k), int((x0 + 4 + w) * self.k), int((y0 + 40) * self.k), int(10 * self.k), cc, 0.95)
            T.draw(frame, side, 40, y0 - 2, 40, colr, True)
            T.draw(frame, f"{(0.0 if abs(v) < 0.05 else v):+.1f}", x1 - 14, y0 + 6, 28, COL_WHITE, True, align="right")
        # legend
        x = 40
        y = 340
        for gi in (1, 2, 3, 4):
            cv2.circle(frame, (int((x + 10) * self.k), int((y + 16) * self.k)), int(8 * self.k), GROUP_COLORS[gi], -1, cv2.LINE_AA)
            w = T.draw(frame, GROUP_NAMES[gi], x + 28, y, 26, COL_GREY)
            x += 28 + w / self.k + 30
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
            poly = np.vstack([pts, np.int32([[[pts[-1, 0, 0], y1 - int(6 * k)]], [[pts[0, 0, 0], y1 - int(6 * k)]]])])
            roi = frame[y0:y1, x0:x1]
            over = roi.copy()
            cv2.fillPoly(over, [poly - np.int32([x0, y0])], (90, 70, 40), cv2.LINE_AA)
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
                   (10, 10, 14), 0.82 * alpha, border=tuple(int(c * alpha) for c in (80, 80, 92)))
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
                (str(self.stats.get("headline", STATS_HEADLINE)), 40, COL_WHITE, True), ("", 20, COL_WHITE, False)]
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
            items.append(("total swipes", f"{self.total_swipes['L'] + self.total_swipes['R']:,}  (L {self.total_swipes['L']} · R {self.total_swipes['R']})"))
        if not any(k.lower().startswith("total spikes") for k, _ in items):
            items.append(("total spikes", f"{self.total_spikes:,}"))
        if not any(k.lower().startswith("posts") for k, _ in items):
            items.append(("posts scrolled", f"L {self.posts['L']} · R {self.posts['R']}"))
        if not any(k.lower().startswith("simulated") for k, _ in items):
            items.append(("simulated time", f"{(self.last_step + 1) * CTRL_DT:.1f} s of brain time"))
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
        self.draw_pip(frame)
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
        t_last = t_start
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
                    t_last = now
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
    r = Renderer(args, meta, events, spikes, positions, groups, stats)
    r.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

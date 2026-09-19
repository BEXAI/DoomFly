"""Synthetic two-panel social "feed" renderer for the DoomFly offline video.

Each :class:`Feed` models one portrait phone screen (one half of a foldable
inner display, 1335 x 1878 px at full resolution) showing an infinite,
deterministic, pseudo-random stack of "cards".  All content is original and
synthetic: colored header bands, avatar placeholders, grey rounded bars in
place of text, gradient/geometry image placeholders and grey action pills.
Roughly every 6-10 cards a high-contrast "novel post" card appears whose
bright saturated fill produces a strong luminance change when it scrolls in.

The renderer keeps all layout state (scroll offset, card tops, velocities) in
full-resolution pixels and rasterises at ``scale`` (default 0.25 -> 334 x 470)
for speed.  Card bitmaps are generated lazily by index from the seed and kept
in a small LRU cache, so a render is essentially a handful of array copies.

Two feed modes share the same control API (``swipe`` / ``set_velocity`` /
``step`` / ``render`` / ``luminance_grid`` / ``state``):

``mode="cards"`` (default)
    The card stack described above.  A swipe is an ease-out flick over
    ``0.55 * height`` px; cards have random heights.

``mode="reels"``
    A full-screen vertical short-video feed (TikTok / Reels style).  Posts are
    drawn from a pool of pre-generated clips (all ``*.mp4`` in ``clips_dir``,
    sorted by name, decoded once, resized to ``clip_res`` portrait frames and
    kept in a module-level cache shared by every Feed).  If the directory has
    no clips, six procedural placeholder clips (5 s @ 24 fps: moving gradients
    with drifting shapes, one high-contrast "novel" clip) are synthesised so
    the code path always works; ``state["clips_synthetic"]`` says so.  Post
    ``i`` is a deterministic shuffle of the pool per seed, every 6-10th post is
    "novel" (the novel-flagged or highest-contrast clip), every post occupies
    exactly one panel height and a swipe snaps (ease-out, 0.35 s) to the next
    post boundary.  The visible clips play and loop with time; the frame shown
    for post ``i`` is ``int((time_s + phase_i * duration) * fps) % n_frames``
    with a per-post phase from the seed, so the rendered image is a pure
    function of ``(seed, time_s, offset)`` and the video renderer replays it
    exactly.  Over the clip sits a light original overlay (bottom gradient,
    avatar disc, grey handle/caption bars, a column of three generic icon
    shapes and a thin progress bar); the overlay is pre-rendered once as an
    RGBA layer so a render is one resize plus one alpha blend.

Typical use::

    L, R = make_pair(seed_l=1, seed_r=2, scale=0.25)            # cards
    L, R = make_pair(seed_l=1, seed_r=2, scale=0.25, mode="reels", clips_dir="assets/clips")
    L.swipe()                    # flick the left panel
    for _ in range(60):
        L.step(1 / 60); R.step(1 / 60)
        frame_l = L.render()     # uint8 BGR (470, 334, 3)
        grid = L.luminance_grid(rows=8, cols=6)   # float32 0..1
    print(L.state)
"""
from __future__ import annotations

import glob
import math
import os
import time
from bisect import bisect_right
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Constants (full-resolution pixels unless noted)
# --------------------------------------------------------------------------- #
PANEL_W: int = 1335
PANEL_H: int = 1878
CARD_GAP: int = 24
CARD_MIN_H: int = 300
CARD_MAX_H: int = 900
NOVEL_MIN_H: int = 520
SIDE_MARGIN: int = 36
CARD_RADIUS: int = 40
CARD_BORDER: int = 4
CARD_PAD: int = 44
APPBAR_H: int = 112
HEADER_BAND_H: int = 14
NOVEL_MIN_GAP: int = 6
NOVEL_MAX_GAP: int = 10
SWIPE_FRACTION: float = 0.55
SWIPE_DURATION_S: float = 0.35
CACHE_CARDS: int = 96

# Reels mode.
MODES: Tuple[str, ...] = ("cards", "reels")
CLIP_RES: Tuple[int, int] = (270, 480)      # (width, height) of cached clip frames
CLIP_MAX_S: float = 20.0                     # decoded clips are truncated to this length
SYNTH_N_CLIPS: int = 6
SYNTH_FPS: int = 24
SYNTH_DURATION_S: float = 5.0
REEL_SNAP_EPS: float = 0.5                   # px tolerance for "offset is on a post boundary"
REEL_MARGIN: int = 48                        # overlay side margin (design px)

Color = Tuple[int, int, int]  # BGR


def _hex(s: str) -> Color:
    """Convert ``'#RRGGBB'`` to an OpenCV BGR tuple."""
    s = s.lstrip("#")
    return (int(s[4:6], 16), int(s[2:4], 16), int(s[0:2], 16))


THEMES: Dict[str, Dict[str, Color]] = {
    "dark": dict(
        bg=_hex("#0B0D12"),
        appbar=_hex("#12151D"),
        appbar_line=_hex("#222836"),
        card=_hex("#171B24"),
        border=_hex("#2A3140"),
        text=_hex("#465063"),
        text_dim=_hex("#2E3545"),
        pill=_hex("#212734"),
        pill_dot=_hex("#56607A"),
        hinge=_hex("#050608"),
    ),
    "light": dict(
        bg=_hex("#E6E8EE"),
        appbar=_hex("#F7F8FA"),
        appbar_line=_hex("#D5D9E2"),
        card=_hex("#FFFFFF"),
        border=_hex("#D0D5DF"),
        text=_hex("#B4BBC9"),
        text_dim=_hex("#D5DAE3"),
        pill=_hex("#EEF0F4"),
        pill_dot=_hex("#A2AABB"),
        hinge=_hex("#C8CCD5"),
    ),
}

# Accent palettes per panel: left = cool, right = warm.  Novel entries are
# (fill, shape, bar) triples: a bright saturated fill plus a bold shape.
PALETTES: Dict[str, Dict[str, list]] = {
    "L": dict(
        accent=[_hex("#4F8CFF"), _hex("#22D3EE"), _hex("#8B5CF6"), _hex("#34D399"), _hex("#60A5FA")],
        novel=[
            (_hex("#FF2D95"), _hex("#FFFFFF"), _hex("#5A0A33")),
            (_hex("#FFE600"), _hex("#111111"), _hex("#3A3300")),
            (_hex("#00E5FF"), _hex("#0B0D12"), _hex("#004A55")),
        ],
    ),
    "R": dict(
        accent=[_hex("#FB7185"), _hex("#F97316"), _hex("#FACC15"), _hex("#A3E635"), _hex("#F472B6")],
        novel=[
            (_hex("#FFE600"), _hex("#111111"), _hex("#3A3300")),
            (_hex("#FF2D95"), _hex("#FFFFFF"), _hex("#5A0A33")),
            (_hex("#7CFF00"), _hex("#101010"), _hex("#204400")),
            (_hex("#FF6A00"), _hex("#FFFFFF"), _hex("#5A2400")),
        ],
    ),
}


# --------------------------------------------------------------------------- #
# Small drawing helpers (operate at raster resolution)
# --------------------------------------------------------------------------- #
def _rounded_mask(h: int, w: int, r: int) -> np.ndarray:
    """uint8 (0/255) mask of a rounded rectangle filling an ``h x w`` box.

    Suitable as the ``mask`` argument of :func:`cv2.copyTo`.
    """
    m = np.zeros((max(h, 0), max(w, 0)), np.uint8)
    if h <= 0 or w <= 0:
        return m
    r = int(max(0, min(r, h // 2, w // 2)))
    cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 255, -1)
    if r > 0:
        for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
            cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return m


def _rrect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, r: int, color: Color) -> None:
    """Fill a rounded rectangle ``[x0,x1) x [y0,y1)`` on ``img`` in place."""
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return
    r = int(max(0, min(r, w // 2, h // 2)))
    if r == 0:
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), color, -1)
        return
    cv2.rectangle(img, (x0 + r, y0), (x1 - 1 - r, y1 - 1), color, -1)
    cv2.rectangle(img, (x0, y0 + r), (x1 - 1, y1 - 1 - r), color, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - 1 - r, y0 + r), (x0 + r, y1 - 1 - r), (x1 - 1 - r, y1 - 1 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def _gradient(h: int, w: int, c0: Color, c1: Color, angle: float) -> np.ndarray:
    """Linear two-color gradient of size ``h x w`` along ``angle`` (radians).

    Computed on a small grid and bilinearly upscaled (exact for a linear ramp).
    """
    gh, gw = min(h, 32), min(w, 32)
    ys = np.linspace(0.0, 1.0, gh, dtype=np.float32) * (h / max(w, h))
    xs = np.linspace(0.0, 1.0, gw, dtype=np.float32) * (w / max(w, h))
    t = xs[None, :] * math.cos(angle) + ys[:, None] * math.sin(angle)
    t -= t.min()
    t /= max(float(t.max()), 1e-6)
    a = np.asarray(c0, np.float32)
    b = np.asarray(c1, np.float32)
    small = (a * (1.0 - t[..., None]) + b * t[..., None]).astype(np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _mix(c: Color, k: float, towards: Color = (0, 0, 0)) -> Color:
    """Blend color ``c`` a fraction ``k`` towards ``towards``."""
    return tuple(int(round(ci * (1 - k) + ti * k)) for ci, ti in zip(c, towards))  # type: ignore[return-value]


def _star(cx: float, cy: float, r_out: float, r_in: float, n: int = 5, rot: float = -math.pi / 2) -> np.ndarray:
    """Vertices of an ``n``-pointed star as an int32 polygon for cv2."""
    pts = []
    for k in range(2 * n):
        r = r_out if k % 2 == 0 else r_in
        a = rot + k * math.pi / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts, np.int32).reshape(-1, 1, 2)


def _tri(cx: float, cy: float, r: float, rot: float) -> np.ndarray:
    """Vertices of an equilateral triangle (circumradius ``r``) for cv2."""
    return np.ascontiguousarray(_star(cx, cy, r, r, n=3, rot=rot)[::2])


def _ease_out(x: float) -> float:
    """Cubic ease-out on ``[0, 1]``."""
    x = min(max(x, 0.0), 1.0)
    return 1.0 - (1.0 - x) ** 3


# --------------------------------------------------------------------------- #
# Clip pool for reels mode
# --------------------------------------------------------------------------- #
class Clip:
    """One decoded short clip: ``frames`` is uint8 BGR ``(n_frames, h, w, 3)``."""

    def __init__(self, name: str, frames: np.ndarray, fps: float, synthetic: bool, novel: bool = False) -> None:
        self.name = name
        self.frames = frames
        self.fps = float(fps)
        self.n_frames = int(frames.shape[0])
        self.synthetic = bool(synthetic)
        self.novel = bool(novel)
        self.contrast = _clip_contrast(frames)

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps

    @property
    def info(self) -> Dict[str, object]:
        return dict(name=self.name, fps=self.fps, n_frames=self.n_frames, synthetic=self.synthetic, novel=self.novel, contrast=round(self.contrast, 4))


# Decoded clips shared across Feed instances, keyed by (absolute path or synthetic id, clip_res).
_CLIP_CACHE: Dict[Tuple[str, Tuple[int, int]], Clip] = {}


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_clips_dir(clips_dir: str) -> str:
    """Relative ``clips_dir`` is taken relative to the cwd, then the repo root."""
    if os.path.isabs(clips_dir) or os.path.isdir(clips_dir):
        return os.path.abspath(clips_dir)
    return os.path.join(_repo_root(), clips_dir)


def _clip_contrast(frames: np.ndarray) -> float:
    """Mean grey-level standard deviation (0..1) over up to 16 sampled frames."""
    n = frames.shape[0]
    idx = np.unique(np.linspace(0, n - 1, min(n, 16)).astype(int))
    return float(np.mean([cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).std() for i in idx]) / 255.0)


def _cover_resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize ``img`` to fill ``size = (w, h)`` (cover: scale to fill, centre crop)."""
    w, h = size
    ih, iw = img.shape[:2]
    if ih <= 0 or iw <= 0:
        return np.zeros((h, w, 3), np.uint8)
    k = max(w / iw, h / ih)
    rw, rh = max(w, int(round(iw * k))), max(h, int(round(ih * k)))
    interp = cv2.INTER_AREA if k < 1.0 else cv2.INTER_LINEAR
    r = cv2.resize(img, (rw, rh), interpolation=interp)
    x0, y0 = (rw - w) // 2, (rh - h) // 2
    return np.ascontiguousarray(r[y0 : y0 + h, x0 : x0 + w])


def _decode_clip(path: str, clip_res: Tuple[int, int]) -> Clip:
    """Decode ``path`` fully (cv2, falling back to imageio) into a :class:`Clip`.

    Clips longer than ``CLIP_MAX_S`` are truncated (they loop anyway).
    """
    frames: List[np.ndarray] = []
    fps = 0.0
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fps = fps if 1.0 <= fps <= 240.0 else 30.0
        limit = max(1, int(round(CLIP_MAX_S * fps)))
        while len(frames) < limit:
            ok, f = cap.read()
            if not ok or f is None:
                break
            frames.append(_cover_resize(f, clip_res))
    cap.release()
    if not frames:
        import imageio.v2 as imageio

        reader = imageio.get_reader(path)
        try:
            fps = float(reader.get_meta_data().get("fps", 30.0) or 30.0)
            limit = max(1, int(round(CLIP_MAX_S * fps)))
            for f in reader.iter_data():
                frames.append(_cover_resize(np.ascontiguousarray(f[..., 2::-1]), clip_res))
                if len(frames) >= limit:
                    break
        finally:
            reader.close()
    if not frames:
        raise RuntimeError(f"could not decode any frame from {path}")
    name = os.path.splitext(os.path.basename(path))[0]
    return Clip(name, np.stack(frames).astype(np.uint8), fps, synthetic=False)


# Placeholder palettes: (background c0, background c1, shape colours).  The last is the novel clip.
_SYNTH_PALETTES: List[Tuple[Color, Color, List[Color]]] = [
    (_hex("#0A1A3F"), _hex("#1FB8D8"), [_hex("#7FE3FF"), _hex("#FFFFFF"), _hex("#3B6CFF")]),
    (_hex("#2A0A4A"), _hex("#E0409A"), [_hex("#FFB3E0"), _hex("#8B5CF6"), _hex("#FFFFFF")]),
    (_hex("#062E2A"), _hex("#2BD48A"), [_hex("#B6FFDD"), _hex("#0F766E"), _hex("#FFFFFF")]),
    (_hex("#3F1200"), _hex("#F97316"), [_hex("#FFD1A6"), _hex("#FFFFFF"), _hex("#B91C1C")]),
    (_hex("#10122E"), _hex("#7C3AED"), [_hex("#C4B5FD"), _hex("#22D3EE"), _hex("#FFFFFF")]),
    (_hex("#FFE600"), _hex("#FFE600"), [_hex("#111111"), _hex("#FF2D95"), _hex("#FFFFFF")]),
]
_NOVEL_LOOKS: List[Tuple[Color, Color]] = [(_hex("#FFE600"), _hex("#111111")), (_hex("#FF2D95"), _hex("#FFFFFF")), (_hex("#0B0D12"), _hex("#00E5FF"))]


def _synth_clip(k: int, clip_res: Tuple[int, int]) -> Clip:
    """Procedural placeholder clip ``k`` (5 s @ 24 fps, seamless loop); the last one is the novel clip."""
    w, h = clip_res
    n = int(round(SYNTH_DURATION_S * SYNTH_FPS))
    novel = k == SYNTH_N_CLIPS - 1
    c0, c1, shape_cols = _SYNTH_PALETTES[k % len(_SYNTH_PALETTES)]
    rs = np.random.RandomState(1000 + k)
    # Drifting shapes: (kind, x amplitude, y amplitude, x cycles, y cycles, x phase, y phase, size, colour).
    params: List[Tuple[int, float, float, int, int, float, float, float, Color]] = [
        (
            int(rs.randint(3)),
            float(rs.uniform(0.18, 0.32)), float(rs.uniform(0.18, 0.32)),
            int(rs.randint(1, 3)), int(rs.randint(1, 3)),
            float(rs.uniform(0, 2 * math.pi)), float(rs.uniform(0, 2 * math.pi)),
            float(rs.uniform(0.16, 0.28)),
            shape_cols[j % len(shape_cols)],
        )
        for j in range(3)
    ]
    angle0 = float(rs.uniform(0, math.pi))
    frames = np.empty((n, h, w, 3), np.uint8)
    for f in range(n):
        t = f / n  # 0..1 over the loop
        if novel:
            # Hard cuts every 0.5 s between three high-contrast looks; one bold pulsing shape.
            cut = int(t * SYNTH_DURATION_S * 2) % len(_NOVEL_LOOKS)
            bg, fg = _NOVEL_LOOKS[cut]
            img = np.empty((h, w, 3), np.uint8)
            img[:] = bg
            pulse = 0.5 + 0.5 * math.sin(2 * math.pi * t * 10)
            cx = int(w * (0.5 + 0.22 * math.sin(2 * math.pi * t)))
            cy = int(h * (0.45 + 0.18 * math.cos(2 * math.pi * 2 * t)))
            r = int(min(w, h) * (0.22 + 0.10 * pulse))
            if cut == 1:
                box = cv2.boxPoints(((cx, cy), (r * 1.9, r * 1.9), 360.0 * t))
                cv2.fillPoly(img, [np.int32(box).reshape(-1, 1, 2)], fg, cv2.LINE_AA)
            elif cut == 2:
                cv2.fillPoly(img, [_star(cx, cy, r * 1.25, r * 0.55, rot=2 * math.pi * t)], fg, cv2.LINE_AA)
            else:
                cv2.circle(img, (cx, cy), r, fg, -1, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), int(r * 0.45), bg, -1, cv2.LINE_AA)
            for s_ in range(3):  # bold stripes sweeping upwards
                y = int(((1.0 - (t * 2 + s_ / 3.0)) % 1.0) * h)
                cv2.rectangle(img, (0, y), (w - 1, min(h - 1, y + 10)), fg, -1)
            frames[f] = img
            continue
        angle = angle0 + 0.9 * math.sin(2 * math.pi * t)
        img = _gradient(h, w, c0, c1, angle)
        cv2.convertScaleAbs(img, dst=img, alpha=1.0 + 0.08 * math.sin(2 * math.pi * t * 2), beta=0)
        layer = img.copy()
        for kind, ax, ay, mx, my, px, py, size, col in params:
            cx = int(w * (0.5 + ax * math.sin(2 * math.pi * (mx * t) + px)))
            cy = int(h * (0.5 + ay * math.cos(2 * math.pi * (my * t) + py)))
            sz = int(size * w)
            if kind == 0:
                cv2.circle(layer, (cx, cy), sz, col, -1, cv2.LINE_AA)
            elif kind == 1:
                box = cv2.boxPoints(((cx, cy), (sz * 1.5, sz * 1.1), 360.0 * t * mx))
                cv2.fillPoly(layer, [np.int32(box).reshape(-1, 1, 2)], col, cv2.LINE_AA)
            else:
                cv2.fillPoly(layer, [_tri(cx, cy, sz, 2 * math.pi * t)], col, cv2.LINE_AA)
        frames[f] = cv2.addWeighted(layer, 0.78, img, 0.22, 0)
    name = f"synthetic_{k:02d}" + ("_novel" if novel else "")
    return Clip(name, frames, SYNTH_FPS, synthetic=True, novel=novel)


def clip_pool(clips_dir: str = "assets/clips", clip_res: Tuple[int, int] = CLIP_RES) -> List[Clip]:
    """All ``*.mp4`` in ``clips_dir`` (sorted by name) as decoded clips, or placeholders.

    Decoding happens once per ``(path, clip_res)`` and is shared across Feed
    instances via a module-level cache.
    """
    res = (int(clip_res[0]), int(clip_res[1]))
    d = _resolve_clips_dir(clips_dir)
    paths = sorted(glob.glob(os.path.join(d, "*.mp4")))
    pool: List[Clip] = []
    if paths:
        for path in paths:
            key = (os.path.abspath(path), res)
            clip = _CLIP_CACHE.get(key)
            if clip is None:
                clip = _decode_clip(path, res)
                _CLIP_CACHE[key] = clip
            pool.append(clip)
        return pool
    for k in range(SYNTH_N_CLIPS):
        key = (f"<synthetic:{k}>", res)
        clip = _CLIP_CACHE.get(key)
        if clip is None:
            clip = _synth_clip(k, res)
            _CLIP_CACHE[key] = clip
        pool.append(clip)
    return pool


# --------------------------------------------------------------------------- #
# Feed
# --------------------------------------------------------------------------- #
class Feed:
    """A deterministic infinite synthetic feed for one foldable half-panel.

    Args:
        panel: ``'L'`` or ``'R'``; selects the accent palette.
        seed: Seed for the card stream; card ``i`` is identical for a given seed.
        width, height: Panel size in full-resolution pixels.
        scale: Raster scale (``0.25`` -> 334 x 470 output).
        theme: ``'dark'`` (default) or ``'light'``.
        autoplay*: Card-mode "autoplay" luminance cuts (see :meth:`_apply_autoplay`).
        mode: ``'cards'`` (default) or ``'reels'`` (full-screen short-video feed).
        clips_dir: Directory of ``*.mp4`` clips for reels mode (placeholders if empty).
        clip_res: ``(w, h)`` of the cached clip frames (portrait 9:16 by default).
        reels_seed: Seed of the clip shuffle in reels mode (defaults to ``seed``).
    """

    def __init__(
        self,
        panel: str,
        seed: int,
        width: int = PANEL_W,
        height: int = PANEL_H,
        scale: float = 0.25,
        theme: str = "dark",
            autoplay: bool = True,
        autoplay_amp: float = 0.8,
        autoplay_hz: float = 1.0,
        novel_amp: float = 0.6,
        novel_hz: float = 1.0,
        autoplay_whole: bool = True,
        mode: str = "cards",
        clips_dir: str = "assets/clips",
        clip_res: Tuple[int, int] = CLIP_RES,
        reels_seed: Optional[int] = None,
    ) -> None:
        if panel not in PALETTES:
            raise ValueError("panel must be 'L' or 'R'")
        if theme not in THEMES:
            raise ValueError(f"unknown theme {theme!r}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.panel = panel
        self.seed = int(seed)
        self.width = int(width)
        self.height = int(height)
        self.scale = float(scale)
        self.theme_name = theme
        self.theme = THEMES[theme]
        self.palette = PALETTES[panel]

        # Output raster size and supersampling factor for card rasterisation.
        self.out_w = int(round(self.width * self.scale))
        self.out_h = int(round(self.height * self.scale))
        self.ss = 2 if self.scale <= 0.5 else 1
        self._s = self.scale * self.ss  # raster scale for card generation

        # Scroll state (full-res px).
        self.offset = 0.0
        self.velocity = 0.0
        self.time_s = 0.0
        self._cont_v = 0.0
        self._anim: Optional[List[float]] = None  # [distance, duration, elapsed]

        # Lazily grown layout.
        self._specs: Dict[int, dict] = {}
        self._tops: List[float] = [float(APPBAR_H + CARD_GAP)]
        self._bottoms: List[float] = []
        self._novel_rs = np.random.RandomState((self.seed * 2654435761 + 0x5EED) & 0x7FFFFFFF)
        self._novel_last = int(self._novel_rs.randint(3, 7))
        self._novel_set = {self._novel_last}
        self._novel_seen: set = set()
        # "Autoplay": visible image blocks (and novel cards) pulse slowly in luminance,
        # like autoplaying video in a real feed. This is the ambient stimulus that lets
        # a motion-driven observer keep scrolling; it is a property of the content, not
        # of the brain. Disable with autoplay=False for a fully static feed.
        self.autoplay = bool(autoplay)
        self.autoplay_amp = float(autoplay_amp)
        self.autoplay_hz = float(autoplay_hz)
        self.novel_amp = float(novel_amp)
        self.novel_hz = float(novel_hz)
        self.autoplay_whole = bool(autoplay_whole)  # cut the whole card, not just its image block
        self._base_frame: Optional[np.ndarray] = None

        # Caches.
        self._cards: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._appbar = self._make_appbar()
        self._frame: Optional[np.ndarray] = None
        self._dirty = True
        # luminance_grid cache: (rows, cols, offset, visible autoplay cut indices) -> grid.
        # The rendered frame is a pure function of those inputs, so a hit is bit-identical.
        self._lum_cache: Optional[Tuple[tuple, np.ndarray]] = None

        # Reels mode: clip pool (decoded lazily on first use), post sequence and overlay.
        self.clips_dir = clips_dir
        self.clip_res = (int(clip_res[0]), int(clip_res[1]))
        self.reels_seed = self.seed if reels_seed is None else int(reels_seed)
        self._pool: Optional[List[Clip]] = None
        self._regular: List[int] = []
        self._novel_clip = 0
        self._reel_posts: Dict[int, dict] = {}
        self._reel_sig: Optional[tuple] = None
        self._ov_rgb: Optional[np.ndarray] = None
        self._ov_a: Optional[np.ndarray] = None
        self._ov_inv: Optional[np.ndarray] = None
        self._ov_bands: List[Tuple[int, int]] = []
        self._crop: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._av_c: Tuple[int, int] = (0, 0)
        self._av_r = 0
        self._prog: Tuple[int, int, int, int] = (0, 0, 0, 0)

        self._update_counters()

    # ------------------------------------------------------------------ #
    # Public state / control
    # ------------------------------------------------------------------ #
    @property
    def state(self) -> Dict[str, object]:
        """Current scroll state as a plain dict.

        Reels mode adds ``current_post``, ``clip_name``, ``clip_frame`` (frame
        index of the current post's clip), ``n_clips`` and ``clips_synthetic``.
        """
        st: Dict[str, object] = dict(
            offset_px=float(self.offset),
            velocity=float(self.velocity),
            posts_consumed=int(self._posts_consumed),
            novel_visible=bool(self._novel_visible),
            novel_count=int(len(self._novel_seen)),
            mode=self.mode,
        )
        if self.mode == "reels":
            pool = self._clip_pool()
            cur = self._reel_current()
            post = self._reel_post(cur)
            st.update(
                current_post=int(cur),
                clip_name=pool[post["clip"]].name,
                clip_frame=int(self._reel_frame_index(cur)),
                n_clips=len(pool),
                clips_synthetic=bool(pool[0].synthetic),
            )
        return st

    @property
    def clip_info(self) -> List[Dict[str, object]]:
        """Per-clip metadata (name, fps, n_frames, synthetic, novel, contrast) in reels mode; ``[]`` otherwise."""
        if self.mode != "reels":
            return []
        return [c.info for c in self._clip_pool()]

    def swipe(self, strength: float = 1.0) -> None:
        """Start an ease-out flick scrolling ``0.55 * height * strength`` px.

        A swipe that lands mid-animation carries the remaining distance of the
        previous flick over into the new one.

        In reels mode a swipe instead snaps (ease-out, 0.35 s) to the next post
        boundary after where the current animation would land; ``strength``
        >= 2 skips ``int(strength)`` posts.
        """
        remaining = 0.0
        if self._anim is not None:
            dist, dur, el = self._anim
            remaining = dist * (1.0 - _ease_out(el / dur))
        if self.mode == "reels":
            landing = self.offset + remaining
            n_posts = max(1, int(float(strength) + 1e-9))
            target = (math.floor((landing + REEL_SNAP_EPS) / self.height) + n_posts) * self.height
            self._anim = [target - self.offset, SWIPE_DURATION_S, 0.0]
            return
        distance = remaining + SWIPE_FRACTION * self.height * float(strength)
        self._anim = [distance, SWIPE_DURATION_S, 0.0]

    def set_velocity(self, px_per_s: float) -> None:
        """Set a continuous scroll velocity (full-res px/s) applied every step."""
        self._cont_v = float(px_per_s)

    def step(self, dt_s: float) -> None:
        """Advance the scroll by ``dt_s`` seconds and refresh counters."""
        dt = max(float(dt_s), 0.0)
        self.time_s += dt
        d = 0.0
        if self._anim is not None:
            dist, dur, el = self._anim
            el2 = min(dur, el + dt)
            d += dist * (_ease_out(el2 / dur) - _ease_out(el / dur))
            if el2 >= dur:
                self._anim = None
            else:
                self._anim[2] = el2
        d += self._cont_v * dt
        new_off = max(0.0, self.offset + d)
        if self.mode == "reels" and self._anim is None and self._cont_v == 0.0:
            # Land exactly on the post boundary (kills float drift from the eased sum).
            snapped = round(new_off / self.height) * self.height
            if abs(snapped - new_off) <= REEL_SNAP_EPS:
                new_off = float(snapped)
        self.velocity = (new_off - self.offset) / dt if dt > 0 else 0.0
        if new_off != self.offset:
            self._dirty = True
            self._base_frame = None
        if (self.autoplay or self.mode == "reels") and dt > 0:
            self._dirty = True
        self.offset = new_off
        self._update_counters()

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self) -> np.ndarray:
        """Return the current frame as uint8 BGR ``(H*scale, W*scale, 3)``."""
        if self._frame is not None and not self._dirty:
            return self._frame
        if self.mode == "reels":
            return self._render_reels()
        if self._base_frame is not None:
            frame = self._apply_autoplay(self._base_frame)
            self._frame = frame
            self._dirty = False
            return frame
        frame = np.empty((self.out_h, self.out_w, 3), np.uint8)
        frame[:] = self.theme["bg"]
        off = self.offset
        bottom_edge = off + self.height
        self._ensure_layout_to(bottom_edge)
        i = bisect_right(self._bottoms, off)
        while True:
            self._ensure_layout_to(max(bottom_edge, self._tops[i]))
            top = self._tops[i]
            if top >= bottom_edge:
                break
            strip = self._card_bitmap(i)
            y = int(round((top - off) * self.scale))
            h = strip.shape[0]
            ya, yb = max(0, y), min(self.out_h, y + h)
            if yb > ya:
                frame[ya:yb] = strip[ya - y : yb - y]
            i += 1
        ab = self._appbar
        frame[: ab.shape[0]] = ab
        self._base_frame = frame
        frame = self._apply_autoplay(frame)
        self._frame = frame
        self._dirty = False
        return frame

    def _apply_autoplay(self, base: np.ndarray) -> np.ndarray:
        """'Autoplay': every visible image block (and novel card) behaves like a muted
        autoplaying clip with hard cuts: its brightness jumps to a new random level
        every `period` seconds (period and phase fixed per card). Between cuts the
        frame is static, so an adapting observer only sees the cuts. Returns a new frame."""
        if not self.autoplay:
            return base
        frame = base.copy()
        off = self.offset
        bottom_edge = off + self.height
        i = bisect_right(self._bottoms, off)
        ab_h = self._appbar.shape[0]
        while i < len(self._tops) and self._tops[i] < bottom_edge:
            spec = self._spec(i)
            top = self._tops[i]
            period, h_i = self._autoplay_cut_period(i, spec)
            if spec["kind"] == "novel":
                y0, y1 = top, top + spec["h"]
                amp = self.novel_amp
            elif spec.get("image") is not None:
                if self.autoplay_whole:
                    y0, y1 = top, top + spec["h"]
                else:
                    y0, y1 = top + spec["image"]["y0"], top + spec["image"]["y1"]
                amp = self.autoplay_amp
            else:
                i += 1
                continue
            if period <= 0.0:   # hz <= 0 means this content never cuts
                i += 1
                continue
            phase = period * (((h_i >> 4) % 1000) / 1000.0)
            cut = int((self.time_s + phase) // period)
            u = (((h_i ^ (cut * 2654435761)) * 2246822519) & 0xFFFFFFFF) / 0xFFFFFFFF
            m = 1.0 + amp * (2.0 * u - 1.0)
            ya = max(ab_h, int(round((y0 - off) * self.scale)))
            yb = min(self.out_h, int(round((y1 - off) * self.scale)))
            if yb > ya:
                pad = int(round((CARD_PAD + 20) * self.scale))
                region = frame[ya:yb, pad:self.out_w - pad]
                cv2.convertScaleAbs(region, dst=region, alpha=m, beta=0)
            i += 1
        return frame

    def _autoplay_cut_period(self, i: int, spec: dict) -> Tuple[float, int]:
        """(cut period in s, per-card hash) of card ``i``; period 0 = this card never cuts."""
        h_i = (self.seed * 1_000_003 + i * 7919) & 0x7FFFFFFF
        if spec["kind"] == "novel":
            period = 1.0 / self.novel_hz if self.novel_hz > 0 else 0.0
        elif spec.get("image") is not None:
            period = (1.0 / self.autoplay_hz if self.autoplay_hz > 0 else 0.0) * (0.7 + 0.6 * ((h_i >> 8) % 1000) / 1000.0)
        else:
            period = 0.0
        return period, h_i

    def _frame_signature(self) -> tuple:
        """Everything the rendered frame depends on: the scroll offset and, with autoplay,
        the current cut index of every visible autoplaying card (see _apply_autoplay)."""
        if self.mode == "reels":
            return self._reel_signature()
        sig: List[object] = [self.offset]
        if not self.autoplay:
            return tuple(sig)
        off = self.offset
        bottom_edge = off + self.height
        i = bisect_right(self._bottoms, off)
        while i < len(self._tops) and self._tops[i] < bottom_edge:
            period, h_i = self._autoplay_cut_period(i, self._spec(i))
            if period > 0.0:
                phase = period * (((h_i >> 4) % 1000) / 1000.0)
                sig.append((i, int((self.time_s + phase) // period)))
            i += 1
        return tuple(sig)

    def luminance_grid(self, rows: int, cols: int) -> np.ndarray:
        """Mean luminance (0..1) of the current frame on a ``rows x cols`` grid.

        Cached on the frame signature: with autoplay the frame only changes at a scroll or a
        cut, so between those the last grid is returned (a copy) without re-rendering."""
        key = (int(rows), int(cols)) + self._frame_signature()
        c = self._lum_cache
        if c is not None and c[0] == key:
            return c[1].copy()
        gray = cv2.cvtColor(self.render(), cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (int(cols), int(rows)), interpolation=cv2.INTER_AREA)
        grid = small.astype(np.float32) / 255.0
        self._lum_cache = (key, grid)
        return grid.copy()

    # ------------------------------------------------------------------ #
    # Layout bookkeeping
    # ------------------------------------------------------------------ #
    def _is_novel(self, i: int) -> bool:
        while self._novel_last < i:
            self._novel_last += int(self._novel_rs.randint(NOVEL_MIN_GAP, NOVEL_MAX_GAP + 1))
            self._novel_set.add(self._novel_last)
        return i in self._novel_set

    def _ensure_layout_to(self, y: float) -> None:
        """Grow the card layout until the last known top exceeds ``y``.

        Invariant: ``len(self._tops) == len(self._bottoms) + 1``.
        """
        while self._tops[-1] <= y:
            i = len(self._bottoms)
            h = self._spec(i)["h"]
            self._bottoms.append(self._tops[i] + h)
            self._tops.append(self._bottoms[i] + CARD_GAP)

    def _update_counters(self) -> None:
        if self.mode == "reels":
            self._posts_consumed = self._reel_current()
            novel = False
            for i, _ in self._reel_visible():
                if self._reel_post(i)["novel"]:
                    novel = True
                    self._novel_seen.add(i)
            self._novel_visible = novel
            return
        off = self.offset
        bottom_edge = off + self.height
        self._ensure_layout_to(bottom_edge)
        consumed = bisect_right(self._bottoms, off)
        self._posts_consumed = consumed
        visible_novel = False
        i = consumed
        while True:
            self._ensure_layout_to(max(bottom_edge, self._tops[i]))
            if self._tops[i] >= bottom_edge:
                break
            if self._spec(i)["kind"] == "novel":
                visible_novel = True
                self._novel_seen.add(i)
            i += 1
        self._novel_visible = visible_novel

    # ------------------------------------------------------------------ #
    # Reels mode
    # ------------------------------------------------------------------ #
    def _clip_pool(self) -> List[Clip]:
        """The decoded clip pool (loaded on first use) plus the per-seed post order."""
        if self._pool is None:
            pool = clip_pool(self.clips_dir, self.clip_res)
            n = len(pool)
            flagged = [k for k, c in enumerate(pool) if c.novel]
            self._novel_clip = flagged[0] if flagged else int(np.argmax([c.contrast for c in pool]))
            order = [int(k) for k in np.random.RandomState(self.reels_seed & 0x7FFFFFFF).permutation(n)]
            # Non-novel posts cycle through the shuffled pool minus the novel clip (if the pool allows).
            self._regular = [k for k in order if k != self._novel_clip] if n >= 3 else order
            self._pool = pool
        return self._pool

    def _reel_post(self, i: int) -> dict:
        """Post ``i``: ``clip`` index, ``novel`` flag, ``phase`` (0..1 start point) and ``accent``."""
        post = self._reel_posts.get(i)
        if post is None:
            self._clip_pool()
            rs = np.random.RandomState((self.seed * 1_000_003 + i * 7919 + 777) & 0x7FFFFFFF)
            novel = self._is_novel(i)
            clip = self._novel_clip if novel else self._regular[i % len(self._regular)]
            accents = self.palette["accent"]
            post = dict(clip=clip, novel=novel, phase=float(rs.uniform(0.0, 1.0)), accent=accents[rs.randint(len(accents))])
            self._reel_posts[i] = post
        return post

    def _reel_current(self) -> int:
        """Index of the post whose top is at (or just above) the screen top."""
        return int(math.floor((self.offset + REEL_SNAP_EPS) / self.height))

    def _reel_visible(self) -> List[Tuple[int, int]]:
        """``(post index, raster y of its top)`` for every post intersecting the screen."""
        cur = self._reel_current()
        out: List[Tuple[int, int]] = []
        for i in (cur, cur + 1):
            y_full = i * self.height - self.offset
            if y_full >= self.height - REEL_SNAP_EPS:
                break
            out.append((i, int(round(y_full * self.scale))))
        return out

    def _reel_frame_index(self, i: int) -> int:
        """Frame of post ``i``'s clip at the current time (loops; pure function of seed and time)."""
        post = self._reel_post(i)
        clip = self._clip_pool()[post["clip"]]
        return int(math.floor(self.time_s * clip.fps + post["phase"] * clip.n_frames)) % clip.n_frames

    def _reel_signature(self) -> tuple:
        """Everything a reels frame depends on: offset and the clip frame of each visible post."""
        return (round(self.offset, 3),) + tuple((i, self._reel_frame_index(i)) for i, _ in self._reel_visible())

    def _render_reels(self) -> np.ndarray:
        sig = self._reel_signature()
        if self._frame is not None and sig == self._reel_sig:
            self._dirty = False
            return self._frame
        if self._ov_rgb is None:
            self._build_reel_overlay()
        frame = np.empty((self.out_h, self.out_w, 3), np.uint8)
        frame[:] = self.theme["bg"]
        for i, y in self._reel_visible():
            panel = self._reel_panel(i)
            ya, yb = max(0, y), min(self.out_h, y + self.out_h)
            if yb > ya:
                frame[ya:yb] = panel[ya - y : yb - y]
        self._frame = frame
        self._reel_sig = sig
        self._dirty = False
        return frame

    def _reel_panel(self, i: int) -> np.ndarray:
        """Full panel image for post ``i``: clip frame (cover) + overlay + dynamic bits."""
        post = self._reel_post(i)
        clip = self._clip_pool()[post["clip"]]
        fi = self._reel_frame_index(i)
        x0, x1, y0, y1 = self._crop
        src = clip.frames[fi, y0:y1, x0:x1]
        img: np.ndarray = cv2.resize(src, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)
        assert self._ov_rgb is not None and self._ov_a is not None and self._ov_inv is not None
        for b0, b1 in self._ov_bands:
            img[b0:b1] = cv2.blendLinear(img[b0:b1], self._ov_rgb[b0:b1], self._ov_inv[b0:b1], self._ov_a[b0:b1])
        # Avatar fill (per-post accent) and progress-bar fill (per-frame).
        cv2.circle(img, self._av_c, self._av_r, post["accent"], -1, cv2.LINE_AA)
        px0, py0, px1, py1 = self._prog
        xe = px0 + int(round((px1 - px0) * (fi + 1) / clip.n_frames))
        if xe > px0 and py1 > py0:
            cv2.rectangle(img, (px0, py0), (xe - 1, py1 - 1), (245, 245, 245), -1)
        return img

    def _build_reel_overlay(self) -> None:
        """Pre-render the static overlay as an RGBA layer at raster size (and the cover crop)."""
        # Cover crop of the clip frame for the panel aspect (in clip pixels).
        cw, ch = self.clip_res
        panel_aspect = self.width / self.height
        if cw / ch < panel_aspect:       # clip taller than panel -> crop rows
            keep = max(1, int(round(cw / panel_aspect)))
            y0 = (ch - keep) // 2
            self._crop = (0, cw, y0, y0 + keep)
        else:                            # clip wider -> crop columns
            keep = max(1, int(round(ch * panel_aspect)))
            x0 = (cw - keep) // 2
            self._crop = (x0, x0 + keep, 0, ch)

        S = self._S
        H, W = S(self.height), S(self.width)
        premul: np.ndarray = np.zeros((H, W, 3), np.float32)
        alpha: np.ndarray = np.zeros((H, W), np.float32)

        def over(mask: np.ndarray, color: Color, a: float) -> None:
            """Composite a shape (coverage ``mask`` 0..255, ``color``, opacity ``a``) over the layer."""
            a_s = mask.astype(np.float32) * (a / 255.0)
            premul[:] = premul * (1.0 - a_s)[..., None] + np.asarray(color, np.float32) * a_s[..., None]
            alpha[:] = a_s + alpha * (1.0 - a_s)

        def shape(draw, color: Color, a: float, dx: int = 0, dy: int = 0) -> None:  # type: ignore[no-untyped-def]
            mk = np.zeros((H, W), np.uint8)
            draw(mk, dx, dy)
            over(mk, color, a)

        def rr(mk: np.ndarray, x0: int, y0: int, x1: int, y1: int, r: int) -> None:
            """Rounded rectangle ``[x0,x1) x [y0,y1)`` into coverage mask ``mk`` (supersampled px)."""
            if x1 > x0 and y1 > y0:
                mk[y0:y1, x0:x1] = np.maximum(mk[y0:y1, x0:x1], _rounded_mask(y1 - y0, x1 - x0, r))

        # Bottom and top gradients (black).
        ys = np.arange(H, dtype=np.float32) / max(H - 1, 1)
        g_bot = np.clip((ys - 0.52) / 0.48, 0.0, 1.0) ** 1.6 * 0.78
        g_top = np.clip(1.0 - ys / 0.12, 0.0, 1.0) ** 1.5 * 0.35
        over(np.broadcast_to((np.maximum(g_bot, g_top) * 255.0)[:, None], (H, W)), (0, 0, 0), 1.0)

        m = REEL_MARGIN
        white: Color = (255, 255, 255)
        grey: Color = (215, 218, 224)
        grey_dim: Color = (185, 190, 200)
        shadow: Color = (0, 0, 0)

        # --- Bottom-left: avatar ring + handle bars + caption bars ----------------------
        av_r = 52
        av_cx, av_cy = m + av_r, self.height - 330
        self._av_c = (int(round(av_cx * self.scale)), int(round(av_cy * self.scale)))
        self._av_r = max(1, int(round((av_r - 6) * self.scale)))
        shape(lambda mk, dx, dy: cv2.circle(mk, (S(av_cx + dx), S(av_cy + dy)), S(av_r + 4), 255, -1, cv2.LINE_AA), shadow, 0.35, 4, 6)
        shape(lambda mk, dx, dy: cv2.circle(mk, (S(av_cx), S(av_cy)), S(av_r), 255, -1, cv2.LINE_AA), white, 0.95)
        hx = av_cx + av_r + 30
        col_x = self.width - m - 140          # left edge of the right icon column
        shape(lambda mk, dx, dy: rr(mk, S(hx), S(av_cy - 40), S(hx + 330), S(av_cy - 6), S(17)), grey, 0.92)
        shape(lambda mk, dx, dy: rr(mk, S(hx), S(av_cy + 10), S(hx + 200), S(av_cy + 36), S(13)), grey_dim, 0.8)
        cap_w = col_x - m - 40
        shape(lambda mk, dx, dy: rr(mk, S(m), S(self.height - 236), S(m + int(cap_w * 0.92)), S(self.height - 206), S(15)), grey, 0.85)
        shape(lambda mk, dx, dy: rr(mk, S(m), S(self.height - 190), S(m + int(cap_w * 0.55)), S(self.height - 160), S(15)), grey, 0.7)
        # Small "sound" pill under the caption: a disc with a short bar.
        shape(lambda mk, dx, dy: cv2.circle(mk, (S(m + 22), S(self.height - 112)), S(22), 255, -1, cv2.LINE_AA), grey_dim, 0.75)
        shape(lambda mk, dx, dy: rr(mk, S(m + 60), S(self.height - 124), S(m + 260), S(self.height - 100), S(12)), grey_dim, 0.6)

        # --- Right column: heart / comment / share + count bars -------------------------
        icx = col_x + 70
        icon_ys = [self.height - 800, self.height - 590, self.height - 380]

        def heart(mk: np.ndarray, dx: int, dy: int) -> None:
            cx, cy = icx + dx, icon_ys[0] + dy
            cv2.circle(mk, (S(cx - 24), S(cy - 16)), S(28), 255, -1, cv2.LINE_AA)
            cv2.circle(mk, (S(cx + 24), S(cy - 16)), S(28), 255, -1, cv2.LINE_AA)
            pts = np.array([(S(cx - 51), S(cy - 6)), (S(cx + 51), S(cy - 6)), (S(cx), S(cy + 54))], np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mk, [pts], 255, cv2.LINE_AA)

        def comment(mk: np.ndarray, dx: int, dy: int) -> None:
            cx, cy = icx + dx, icon_ys[1] + dy
            cv2.ellipse(mk, (S(cx), S(cy - 8)), (S(50), S(40)), 0, 0, 360, 255, -1, cv2.LINE_AA)
            pts = np.array([(S(cx - 34), S(cy + 18)), (S(cx - 10), S(cy + 30)), (S(cx - 46), S(cy + 50))], np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mk, [pts], 255, cv2.LINE_AA)

        def share(mk: np.ndarray, dx: int, dy: int) -> None:
            cx, cy = icx + dx, icon_ys[2] + dy
            pts = np.array([(S(cx + 6), S(cy - 52)), (S(cx + 54), S(cy - 12)), (S(cx + 6), S(cy + 28))], np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mk, [pts], 255, cv2.LINE_AA)
            cv2.ellipse(mk, (S(cx - 6), S(cy + 30)), (S(42), S(42)), 0, 90, 300, 255, max(1, S(16)), cv2.LINE_AA)

        for draw in (heart, comment, share):
            shape(draw, shadow, 0.4, 4, 6)
            shape(draw, white, 0.94)
        for cy in icon_ys:
            shape(lambda mk, dx, dy, cy=cy: rr(mk, S(icx - 40), S(cy + 76), S(icx + 40), S(cy + 96), S(10)), grey, 0.9)
        # Disc at the bottom of the column (a "sound cover" placeholder).
        shape(lambda mk, dx, dy: cv2.circle(mk, (S(icx), S(self.height - 200)), S(46), 255, -1, cv2.LINE_AA), grey_dim, 0.7)
        shape(lambda mk, dx, dy: cv2.circle(mk, (S(icx), S(self.height - 200)), S(18), 255, -1, cv2.LINE_AA), (40, 40, 44), 0.9)

        # --- Progress bar track (fill is drawn per frame) --------------------------------
        pb_y0, pb_y1 = self.height - 16, self.height - 8
        shape(lambda mk, dx, dy: cv2.rectangle(mk, (0, S(pb_y0)), (W - 1, S(pb_y1) - 1), 255, -1), white, 0.3)
        self._prog = (0, int(round(pb_y0 * self.scale)), self.out_w, int(round(pb_y1 * self.scale)))

        # Downsample (premultiplied) to raster size and split into RGB + alpha.
        if self.ss != 1:
            premul = cv2.resize(premul, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(alpha, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        a = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        rgb = premul / np.maximum(a, 1e-3)[..., None]
        self._ov_rgb = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
        self._ov_a = np.ascontiguousarray(a)
        self._ov_inv = np.ascontiguousarray(1.0 - a)
        # Row bands where the overlay has any coverage (blend only there).
        rows = np.flatnonzero(a.max(axis=1) > 0.5 / 255.0)
        bands: List[Tuple[int, int]] = []
        if rows.size:
            start = prev = int(rows[0])
            for r in rows[1:]:
                if int(r) != prev + 1:
                    bands.append((start, prev + 1))
                    start = int(r)
                prev = int(r)
            bands.append((start, prev + 1))
        self._ov_bands = bands

    # ------------------------------------------------------------------ #
    # Card specification (pure function of seed and index)
    # ------------------------------------------------------------------ #
    def _spec(self, i: int) -> dict:
        spec = self._specs.get(i)
        if spec is None:
            spec = self._make_spec(i)
            self._specs[i] = spec
        return spec

    def _make_spec(self, i: int) -> dict:
        rs = np.random.RandomState((self.seed * 1_000_003 + i * 7919 + 12345) & 0x7FFFFFFF)
        accents = self.palette["accent"]
        if self._is_novel(i):
            fill, shape, bar = self.palette["novel"][rs.randint(len(self.palette["novel"]))]
            return dict(
                kind="novel",
                h=int(rs.randint(NOVEL_MIN_H, CARD_MAX_H + 1)),
                fill=fill,
                shape=shape,
                bar=bar,
                shape_kind=int(rs.randint(4)),
                shape_rot=float(rs.uniform(0, math.pi)),
                bar_w=float(rs.uniform(0.45, 0.7)),
            )
        h = int(rs.randint(CARD_MIN_H, CARD_MAX_H + 1))
        text_y0 = HEADER_BAND_H + 160
        footer_h = 116
        avail_lines = max(1, (h - text_y0 - footer_h + 20) // 48)
        n_lines = int(min(rs.randint(1, 4), avail_lines))
        text_end = text_y0 + n_lines * 48 - 20
        img_y0 = text_end + 28
        img_y1 = h - 92 - 28
        has_image = (img_y1 - img_y0) >= 180 and rs.rand() < 0.8
        if not has_image:
            # No image: fill the card with a longer paragraph instead of leaving
            # dead space between the text and the footer.
            n_lines = int(min(avail_lines, max(n_lines, rs.randint(max(1, avail_lines - 1), avail_lines + 1))))
        fracs = [1.0] * n_lines
        for k in range(n_lines):
            if k == n_lines - 1 or (k > 0 and rs.rand() < 0.22):
                fracs[k] = float(rs.uniform(0.35, 0.9))
        image = None
        if has_image:
            c0 = accents[rs.randint(len(accents))]
            c1 = accents[rs.randint(len(accents))]
            dark = _mix(c0, 0.55)
            c1 = _mix(c1, 0.35)
            shapes = []
            for _ in range(int(rs.randint(2, 5))):
                shapes.append(
                    dict(
                        kind=int(rs.randint(4)),
                        cx=float(rs.uniform(0.1, 0.9)),
                        cy=float(rs.uniform(0.15, 0.85)),
                        size=float(rs.uniform(0.12, 0.32)),
                        rot=float(rs.uniform(0, math.pi)),
                        color=_mix(accents[rs.randint(len(accents))], float(rs.uniform(0.0, 0.5)), (255, 255, 255)),
                    )
                )
            image = dict(y0=img_y0, y1=img_y1, c0=dark, c1=c1, angle=float(rs.uniform(0, math.pi)), shapes=shapes)
        return dict(
            kind="post",
            h=h,
            accent=accents[rs.randint(len(accents))],
            avatar=accents[rs.randint(len(accents))],
            name_w=int(rs.randint(220, 421)),
            handle_w=int(rs.randint(130, 261)),
            n_lines=n_lines,
            fracs=fracs,
            text_y0=text_y0,
            image=image,
            counts=[int(rs.randint(40, 110)) for _ in range(3)],
        )

    # ------------------------------------------------------------------ #
    # Rasterisation
    # ------------------------------------------------------------------ #
    def _S(self, v: float) -> int:
        """Full-res -> supersampled raster px."""
        return int(round(v * self._s))

    def _card_bitmap(self, i: int) -> np.ndarray:
        """Panel-wide strip (out-scale) for card ``i`` from the LRU cache."""
        strip = self._cards.get(i)
        if strip is not None:
            self._cards.move_to_end(i)
            return strip
        strip = self._raster_card(self._spec(i))
        self._cards[i] = strip
        while len(self._cards) > CACHE_CARDS:
            self._cards.popitem(last=False)
        return strip

    def _raster_card(self, spec: dict) -> np.ndarray:
        th = self.theme
        S = self._S
        h_px = S(spec["h"])
        strip_w = S(self.width)
        x0, x1 = S(SIDE_MARGIN), S(self.width - SIDE_MARGIN)
        cw = x1 - x0
        strip: np.ndarray = np.empty((h_px, strip_w, 3), np.uint8)
        strip[:] = th["bg"]

        if spec["kind"] == "novel":
            content = self._draw_novel(spec, h_px, cw)
        else:
            content = self._draw_post(spec, h_px, cw)

        r = S(CARD_RADIUS)
        b = max(1, S(CARD_BORDER))
        outer = _rounded_mask(h_px, cw, r)
        inner = np.zeros((h_px, cw), np.uint8)
        inner[b:-b, b:-b] = _rounded_mask(h_px - 2 * b, cw - 2 * b, r - b)
        border = np.empty_like(content)
        border[:] = th["border"]
        region = strip[:, x0:x1]
        cv2.copyTo(border, outer, region)
        cv2.copyTo(content, inner, region)

        if self.ss != 1:
            out_h = int(round(spec["h"] * self.scale))
            strip = cv2.resize(strip, (self.out_w, max(1, out_h)), interpolation=cv2.INTER_AREA)
        return strip

    def _draw_post(self, spec: dict, h_px: int, cw: int) -> np.ndarray:
        th = self.theme
        S = self._S
        img = np.empty((h_px, cw, 3), np.uint8)
        img[:] = th["card"]
        pad = S(CARD_PAD)

        # Colored header band.
        cv2.rectangle(img, (0, 0), (cw - 1, S(HEADER_BAND_H) - 1), spec["accent"], -1)

        # Avatar placeholder: filled disc with a generic silhouette.
        acx, acy, ar = S(CARD_PAD + 44), S(HEADER_BAND_H + 78), S(46)
        cv2.circle(img, (acx, acy), ar, spec["avatar"], -1, cv2.LINE_AA)
        sil = _mix(spec["avatar"], 0.55)
        cv2.circle(img, (acx, acy - S(12)), S(17), sil, -1, cv2.LINE_AA)
        cv2.ellipse(img, (acx, acy + S(40)), (S(30), S(24)), 0, 180, 360, sil, -1, cv2.LINE_AA)

        # Name / handle bars.
        nx = S(CARD_PAD + 112)
        _rrect(img, nx, S(HEADER_BAND_H + 50), nx + S(spec["name_w"]), S(HEADER_BAND_H + 76), S(13), th["text"])
        _rrect(img, nx, S(HEADER_BAND_H + 92), nx + S(spec["handle_w"]), S(HEADER_BAND_H + 110), S(9), th["text_dim"])
        # Small accent dot ("verified"-like badge, abstract).
        cv2.circle(img, (nx + S(spec["name_w"]) + S(22), S(HEADER_BAND_H + 63)), S(9), spec["accent"], -1, cv2.LINE_AA)
        # Overflow dots at the right.
        for k in range(3):
            cv2.circle(img, (cw - pad - S(8) - k * S(20), S(HEADER_BAND_H + 62)), S(5), th["text"], -1, cv2.LINE_AA)

        # Body "text" lines.
        y = spec["text_y0"]
        full_w = self.width - 2 * SIDE_MARGIN - 2 * CARD_PAD
        for frac in spec["fracs"]:
            _rrect(img, pad, S(y), pad + S(full_w * frac), S(y + 28), S(14), th["text"])
            y += 48

        # Optional image placeholder block.
        im = spec["image"]
        if im is not None:
            iy0, iy1 = S(im["y0"]), S(im["y1"])
            ih, iw = iy1 - iy0, cw - 2 * pad
            if ih > 2 and iw > 2:
                block = _gradient(ih, iw, im["c0"], im["c1"], im["angle"])
                shapes = block.copy()
                for sh in im["shapes"]:
                    cx, cy = int(sh["cx"] * iw), int(sh["cy"] * ih)
                    sz = int(sh["size"] * min(iw, ih))
                    col = sh["color"]
                    if sh["kind"] == 0:
                        cv2.circle(shapes, (cx, cy), sz, col, -1, cv2.LINE_AA)
                    elif sh["kind"] == 1:
                        box = cv2.boxPoints(((cx, cy), (sz * 1.6, sz * 1.1), math.degrees(sh["rot"])))
                        cv2.fillPoly(shapes, [np.int32(box).reshape(-1, 1, 2)], col, cv2.LINE_AA)
                    elif sh["kind"] == 2:
                        cv2.fillPoly(shapes, [_tri(cx, cy, sz, sh["rot"])], col, cv2.LINE_AA)
                    else:
                        cv2.circle(shapes, (cx, cy), sz, col, max(2, sz // 4), cv2.LINE_AA)
                block = cv2.addWeighted(shapes, 0.82, block, 0.18, 0)
                m = _rounded_mask(ih, iw, S(24))
                cv2.copyTo(block, m, img[iy0:iy1, pad : pad + iw])

        # Footer: three grey action pills with a dot and a short count bar.
        py0 = S(spec["h"] - 92)
        py1 = S(spec["h"] - 44)
        pw = S(168)
        for k in range(3):
            px = pad + k * S(196)
            _rrect(img, px, py0, px + pw, py1, S(24), th["pill"])
            cv2.circle(img, (px + S(34), (py0 + py1) // 2), S(11), th["pill_dot"], -1, cv2.LINE_AA)
            cnt = S(spec["counts"][k])
            _rrect(img, px + S(62), (py0 + py1) // 2 - S(8), px + S(62) + cnt, (py0 + py1) // 2 + S(8), S(8), th["pill_dot"])
        return img

    def _draw_novel(self, spec: dict, h_px: int, cw: int) -> np.ndarray:
        S = self._S
        img = np.empty((h_px, cw, 3), np.uint8)
        img[:] = spec["fill"]
        cx, cy = cw // 2, int(h_px * 0.44)
        r = int(min(cw, h_px) * 0.30)
        col = spec["shape"]
        kind = spec["shape_kind"]
        if kind == 0:
            cv2.circle(img, (cx, cy), r, col, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), int(r * 0.55), spec["fill"], -1, cv2.LINE_AA)
        elif kind == 1:
            box = cv2.boxPoints(((cx, cy), (r * 1.6, r * 1.6), 45.0))
            cv2.fillPoly(img, [np.int32(box).reshape(-1, 1, 2)], col, cv2.LINE_AA)
        elif kind == 2:
            cv2.fillPoly(img, [_star(cx, cy, r * 1.1, r * 0.5)], col, cv2.LINE_AA)
        else:
            cv2.fillPoly(img, [_tri(cx, cy, r * 1.15, -math.pi / 2)], col, cv2.LINE_AA)
        # Bold "headline" bars near the bottom.
        pad = S(CARD_PAD + 20)
        bw = int((cw - 2 * pad) * spec["bar_w"])
        by = h_px - S(150)
        _rrect(img, pad, by, pad + bw, by + S(44), S(22), spec["bar"])
        _rrect(img, pad, by + S(64), pad + int(bw * 0.55), by + S(64) + S(30), S(15), spec["bar"])
        # Small pill row on the right to echo the normal card footer.
        px = cw - pad - S(160)
        _rrect(img, px, by + S(6), px + S(160), by + S(6) + S(52), S(26), col)
        return img

    def _make_appbar(self) -> np.ndarray:
        """Fixed top app bar with abstract controls and an L/R label."""
        th = self.theme
        S = self._S
        h = S(APPBAR_H)
        w = S(self.width)
        bar: np.ndarray = np.empty((h, w, 3), np.uint8)
        bar[:] = th["appbar"]
        cv2.rectangle(bar, (0, h - max(1, S(3))), (w - 1, h - 1), th["appbar_line"], -1)
        accent = self.palette["accent"][0]
        cv2.circle(bar, (S(SIDE_MARGIN + 40), h // 2), S(28), accent, -1, cv2.LINE_AA)
        cv2.circle(bar, (S(SIDE_MARGIN + 40), h // 2), S(11), th["appbar"], -1, cv2.LINE_AA)
        _rrect(bar, S(SIDE_MARGIN + 100), h // 2 - S(14), S(SIDE_MARGIN + 420), h // 2 + S(14), S(14), th["text"])
        for k in range(2):
            cx = w - S(SIDE_MARGIN + 130) + k * S(70)
            cv2.circle(bar, (cx, h // 2), S(20), th["pill"], -1, cv2.LINE_AA)
            cv2.circle(bar, (cx, h // 2), S(7), th["pill_dot"], -1, cv2.LINE_AA)
        fs = 1.9 * self._s
        cv2.putText(bar, self.panel, (w - S(SIDE_MARGIN + 32), h // 2 + S(22)), cv2.FONT_HERSHEY_SIMPLEX, fs, th["text"], max(1, S(5)), cv2.LINE_AA)
        if self.ss != 1:
            bar = cv2.resize(bar, (self.out_w, max(1, int(round(APPBAR_H * self.scale)))), interpolation=cv2.INTER_AREA)
        return bar


# --------------------------------------------------------------------------- #
# Module API
# --------------------------------------------------------------------------- #
def make_pair(seed_l: int = 1, seed_r: int = 2, scale: float = 0.25, mode: str = "cards", clips_dir: str = "assets/clips", **kw) -> Tuple[Feed, Feed]:
    """Create the (left, right) feed pair with independent palettes.

    ``mode`` / ``clips_dir`` select the feed type (see the module docstring);
    remaining keyword arguments (autoplay*, theme, clip_res, reels_seed, ...) are
    forwarded to both :class:`Feed` instances.
    """
    return Feed("L", seed_l, scale=scale, mode=mode, clips_dir=clips_dir, **kw), Feed("R", seed_r, scale=scale, mode=mode, clips_dir=clips_dir, **kw)


def _demo(mode: str = "cards") -> None:
    """Render a 6 s / 60 fps side-by-side demo video and a still frame (``python feeds.py [cards|reels]``)."""
    import imageio.v2 as imageio

    out_dir = os.path.join(_repo_root(), "out")
    os.makedirs(out_dir, exist_ok=True)
    tag = "feeds_demo" if mode == "cards" else f"{mode}_demo"
    mp4_path = os.path.join(out_dir, f"{tag}.mp4")
    png_path = os.path.join(out_dir, f"{tag}.png")

    fps = 60
    n_frames = 6 * fps
    left, right = make_pair(mode=mode)
    swipes_l = {int(round(t * fps)) for t in (1.0, 2.2, 4.0)}
    swipes_r = {int(round(t * fps)) for t in (1.5, 3.1, 5.0)}
    hinge = np.empty((left.out_h, 8, 3), np.uint8)
    hinge[:] = left.theme["hinge"]

    timings: List[float] = []
    writer = imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=1)
    still: Optional[np.ndarray] = None
    try:
        for f in range(n_frames):
            if f in swipes_l:
                left.swipe()
            if f in swipes_r:
                right.swipe()
            left.step(1.0 / fps)
            right.step(1.0 / fps)
            t0 = time.perf_counter()
            fl = left.render()
            fr = right.render()
            timings.append((time.perf_counter() - t0) * 1000.0 / 2.0)
            frame = np.hstack([fl, hinge, fr])
            writer.append_data(np.ascontiguousarray(frame[..., ::-1]))
            if f == int(2.35 * fps):
                still = frame.copy()
    finally:
        writer.close()
    if still is None:
        still = frame
    cv2.imwrite(png_path, still)

    arr = np.asarray(timings)
    print(f"wrote {mp4_path}")
    print(f"wrote {png_path}")
    print(f"render ms per panel: mean {arr.mean():.3f}  median {np.median(arr):.3f}  max {arr.max():.3f}  (first {arr[0]:.3f})")
    print("L state:", left.state)
    print("R state:", right.state)
    print("L luminance grid 6x4:\n", np.round(left.luminance_grid(6, 4), 3))


if __name__ == "__main__":
    import sys

    _demo(sys.argv[1] if len(sys.argv) > 1 else "cards")

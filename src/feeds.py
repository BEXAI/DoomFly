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

Typical use::

    L, R = make_pair(seed_l=1, seed_r=2, scale=0.25)
    L.swipe()                    # flick the left panel
    for _ in range(60):
        L.step(1 / 60); R.step(1 / 60)
        frame_l = L.render()     # uint8 BGR (470, 334, 3)
        grid = L.luminance_grid(rows=8, cols=6)   # float32 0..1
    print(L.state)
"""
from __future__ import annotations

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
    ) -> None:
        if panel not in PALETTES:
            raise ValueError("panel must be 'L' or 'R'")
        if theme not in THEMES:
            raise ValueError(f"unknown theme {theme!r}")
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

        self._update_counters()

    # ------------------------------------------------------------------ #
    # Public state / control
    # ------------------------------------------------------------------ #
    @property
    def state(self) -> Dict[str, object]:
        """Current scroll state as a plain dict."""
        return dict(
            offset_px=float(self.offset),
            velocity=float(self.velocity),
            posts_consumed=int(self._posts_consumed),
            novel_visible=bool(self._novel_visible),
            novel_count=int(len(self._novel_seen)),
        )

    def swipe(self, strength: float = 1.0) -> None:
        """Start an ease-out flick scrolling ``0.55 * height * strength`` px.

        A swipe that lands mid-animation carries the remaining distance of the
        previous flick over into the new one.
        """
        remaining = 0.0
        if self._anim is not None:
            dist, dur, el = self._anim
            remaining = dist * (1.0 - _ease_out(el / dur))
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
        self.velocity = (new_off - self.offset) / dt if dt > 0 else 0.0
        if new_off != self.offset:
            self._dirty = True
            self._base_frame = None
        if self.autoplay and dt > 0:
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
            h_i = (self.seed * 1_000_003 + i * 7919) & 0x7FFFFFFF
            if spec["kind"] == "novel":
                y0, y1 = top, top + spec["h"]
                period, amp = (1.0 / self.novel_hz if self.novel_hz > 0 else 0.0), self.novel_amp
            elif spec.get("image") is not None:
                if self.autoplay_whole:
                    y0, y1 = top, top + spec["h"]
                else:
                    y0, y1 = top + spec["image"]["y0"], top + spec["image"]["y1"]
                period = (1.0 / self.autoplay_hz if self.autoplay_hz > 0 else 0.0) * (0.7 + 0.6 * ((h_i >> 8) % 1000) / 1000.0)
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

    def luminance_grid(self, rows: int, cols: int) -> np.ndarray:
        """Mean luminance (0..1) of the current frame on a ``rows x cols`` grid."""
        gray = cv2.cvtColor(self.render(), cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (int(cols), int(rows)), interpolation=cv2.INTER_AREA)
        return small.astype(np.float32) / 255.0

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
def make_pair(seed_l: int = 1, seed_r: int = 2, scale: float = 0.25, **kw) -> Tuple[Feed, Feed]:
    """Create the (left, right) feed pair with independent palettes."""
    return Feed("L", seed_l, scale=scale, **kw), Feed("R", seed_r, scale=scale, **kw)


def _demo() -> None:
    """Render a 6 s / 60 fps side-by-side demo video and a still frame."""
    import imageio.v2 as imageio

    out_dir = "/home/user/DoomFly/out"
    os.makedirs(out_dir, exist_ok=True)
    mp4_path = os.path.join(out_dir, "feeds_demo.mp4")
    png_path = os.path.join(out_dir, "feeds_demo.png")

    fps = 60
    n_frames = 6 * fps
    left, right = make_pair()
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
    _demo()

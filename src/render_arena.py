#!/usr/bin/env python3
"""DoomFly arena renderer: composites an arena run (``out/arena_<cond>_s<seed>.jsonl``) into a
1080x1920 @ 60 fps mp4 (see docs/ARENA_DESIGN.md, "Video").

Layout (design px, 1080 x 1920):
    0 .. 1080     top-down plan view of the 400 x 400 mm room (flat grid, walls), the open Duo
                  lying flat at the centre with the two live replayed feeds inside its bevel, the
                  fading 15 s trail, the fly sprite rotated to the logged heading, contact rings
                  when a swipe is applied, a soft flash on the phone on reward events
    1080 .. 1920  HUD (left: condition, time on phone, visits / swipes / rewards, KC->MBON weight
                  bar) and the brain PiP (right), then the distance-to-phone plot
Storyboard: title card (0-4 s), 4-bar comparison card from ``--summary`` (omitted when absent),
attribution card.  Timings scale like render.py for runs shorter than 50 s.

Everything reusable comes from ``src/render.py`` (TextRenderer, fill_panel, Device, Fly (image
mode, for the sprite), BrainPiP, cards, spike handling); this module adds the plan-view scene.

Usage::

    python3 src/render_arena.py --events out/arena_dopamine_s0.jsonl \
        [--spikes out/arena_dopamine_s0_spikes.npz] [--summary out/arena_summary.json] \
        [--positions data/graph/positions.npy --groups data/graph/groups.npy] \
        --out out/arena_dopamine_s0.mp4 [--duration S] [--preview] [--stills 1,20,40]
    python3 src/render_arena.py --fixture [--fixture-cond dopamine --fixture-duration 60] --preview

``--fixture`` writes a schema-exact synthetic log (correlated random walk that visits the phone)
to ``arena_fixture.jsonl`` (plus ``arena_fixture_summary.json``) next to ``--out`` (default ``out/``)
and renders it; it never touches real logs.  ``--preview`` renders 540x960 and every other frame.

Conventions: world x right / y up in mm, room centred at 0; heading theta in radians, 0 = +x,
CCW positive; the sprite's head-up direction maps to the heading.  Screen y is down, so a world
direction (cos th, sin th) is drawn as (cos th, -sin th).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
try:
    from src.feeds import make_pair  # noqa: E402
    from src.render import (  # noqa: E402
        ATTRIBUTION, COL_DIM, COL_GREY, COL_L, COL_R, COL_WHITE, FLY_IMAGE_BODY_LEN, FLY_SPRITE_JSON,
        FLY_SPRITE_PNG, PIP_SIZE, BrainPiP, Device, Fly, TextRenderer, _pt, card_alpha, clamp, ease_in_out,
        ease_out, fill_panel, load_events, load_spikes, parse_stills, premul_blit, rounded_mask)
except ImportError:  # running from inside src/
    from feeds import make_pair  # type: ignore  # noqa: E402
    from render import (  # type: ignore  # noqa: E402
        ATTRIBUTION, COL_DIM, COL_GREY, COL_L, COL_R, COL_WHITE, FLY_IMAGE_BODY_LEN, FLY_SPRITE_JSON,
        FLY_SPRITE_PNG, PIP_SIZE, BrainPiP, Device, Fly, TextRenderer, _pt, card_alpha, clamp, ease_in_out,
        ease_out, fill_panel, load_events, load_spikes, parse_stills, premul_blit, rounded_mask)

# --------------------------------------------------------------------------- #
# Design constants (design px at 1080 x 1920, scaled by k)
# --------------------------------------------------------------------------- #
DESIGN_W, DESIGN_H = 1080, 1920
FPS = 60
CTRL_DT = 0.016
NOMINAL_LEN_S = 50.0        # storyboard timings are specified for >= 50 s runs (like render.py)

ARENA_PX = 1080             # the plan view is the top square
ROOM_MM = 400.0
ROOM_INSET = 26.0           # design px between the frame edge and the wall's outer face
PX_PER_MM = (ARENA_PX - 2 * ROOM_INSET) / ROOM_MM      # 2.57 design px / mm
ARENA_CX, ARENA_CY = ARENA_PX / 2.0, ARENA_PX / 2.0
PHONE_W_MM, PHONE_H_MM = 158.0, 111.0
PANEL_W_MM, PANEL_H_MM = 79.0, 111.0
WALL_MM = 4.0
FLY_BODY_MM = 30.0
LEG_REACH_MM = 22.0
LEG_ANGLE_RAD = math.radians(35.0)
TRAIL_S = 15.0
TRAIL_BUCKETS = 30
VISIT_DEBOUNCE_S = 0.5      # analyze_arena.DEBOUNCE_S: off-phone gaps shorter than this do not end a visit
SWIPE_ANIM_S = 0.42
DEVICE_K_RATIO = 0.41       # bevel / hinge / corner radius of the small phone relative to render.py's

PIP_XY = (600, 1110)
HUD_X0, HUD_X1 = 40, 560
PLOT_Y0, PLOT_Y1 = 1610, 1800

TITLE = "A fly brain is free to ignore the phone"
SUBTITLE = "MaleCNS v1.0 connectome · walking simulation · nothing scripted"
COND_LABELS = {
    "real": ("REAL WIRING", "brain steers the body · swipes only when a leg is on a panel · no dopamine"),
    "dopamine": ("REAL WIRING + DOPAMINE", "a novel post seen from the phone drives PAM → KC→MBON depression"),
    "shuffled": ("SHUFFLED WIRING", "same neurons, edges shuffled · brain steers · no dopamine"),
    "random": ("RANDOM WALK", "brain simulated but disconnected from the body · swipes never applied"),
    "trained": ("TRAINED WIRING", "KC→MBON weights from 10 min of forced, rewarded scrolling · plasticity frozen · no dopamine"),
}
COND_ORDER = ("real", "dopamine", "shuffled", "random", "trained")
COND_SHORT = {"real": "real", "dopamine": "dopamine", "shuffled": "shuffled", "random": "random", "trained": "trained"}
COL_ACCENT = (80, 200, 255)          # BGR amber: highlights, time-on-phone
COL_REWARD = (90, 200, 255)          # BGR gold: reward flash
COL_DOPA = (220, 80, 240)            # BGR magenta: dopamine / plasticity
COL_TRAIL = (200, 215, 235)          # BGR warm white
FLOOR = (28, 15, 9)                  # BGR navy (#090f1c)
FLOOR_OUTSIDE = (16, 9, 6)
GRID_COLOR = (178, 168, 132)
WALL_COLOR = (150, 140, 110)


def _as_int(v: object, default: int) -> int:
    if isinstance(v, bool) or v is None:
        return default
    try:
        return int(v)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def world_to_screen(x_mm: float, y_mm: float, k: float) -> Tuple[float, float]:
    """World mm (x right, y up, room centred) -> frame px (y down)."""
    return (ARENA_CX + x_mm * PX_PER_MM) * k, (ARENA_CY - y_mm * PX_PER_MM) * k


def leg_tips(x: float, y: float, th: float) -> Dict[str, Tuple[float, float]]:
    """Front-leg tip positions (mm) per docs/ARENA_DESIGN.md: 22 mm at +-35 deg from heading; L = +35."""
    return {
        "L": (x + LEG_REACH_MM * math.cos(th + LEG_ANGLE_RAD), y + LEG_REACH_MM * math.sin(th + LEG_ANGLE_RAD)),
        "R": (x + LEG_REACH_MM * math.cos(th - LEG_ANGLE_RAD), y + LEG_REACH_MM * math.sin(th - LEG_ANGLE_RAD)),
    }


def panel_under(px: float, py: float) -> Optional[str]:
    if abs(py) > PANEL_H_MM / 2:
        return None
    if -PANEL_W_MM <= px < 0:
        return "L"
    if 0 <= px <= PANEL_W_MM:
        return "R"
    return None


def on_phone(x: float, y: float) -> bool:
    return abs(x) <= PHONE_W_MM / 2 and abs(y) <= PHONE_H_MM / 2


def dist_to_phone(x: float, y: float) -> float:
    dx = max(abs(x) - PHONE_W_MM / 2, 0.0)
    dy = max(abs(y) - PHONE_H_MM / 2, 0.0)
    return math.hypot(dx, dy)


# --------------------------------------------------------------------------- #
# Fixture: schema-exact fake log (correlated random walk that visits the phone)
# --------------------------------------------------------------------------- #
def make_fixture(path: str, cond: str = "dopamine", seed: int = 0, duration: float = 60.0,
                 summary_path: Optional[str] = None) -> None:
    """Write a fake ``arena_<cond>_s<seed>.jsonl``-shaped log to ``path``.

    The walker is an OU-turning correlated random walk with an approach bias toward the phone
    during "interested" bouts, so the log contains visits, contact-gated swipes, blocked swipes,
    reward events (novel rising edge while on the phone) and a decaying w_ratio (dopamine only).
    Never writes to a real ``arena_<cond>_s<seed>.jsonl`` path.
    """
    if os.path.basename(path).startswith("arena_") and "fixture" not in os.path.basename(path):
        raise ValueError(f"refusing to write a fixture over a real-looking log: {path}")
    rng = np.random.default_rng(seed)
    dt = CTRL_DT
    T = int(round(duration / dt))
    # start pose: >= 120 mm from the phone centre
    while True:
        x, y = (float(v) for v in rng.uniform(-180, 180, 2))
        if math.hypot(x, y) >= 120:
            break
    th = float(rng.uniform(-math.pi, math.pi))
    omega = 0.0
    bout_left = 0.0
    bout_kind = "wander"
    stand_left = 0.0
    w_ratio = 1.0
    dopamine = cond == "dopamine"
    steer = cond != "random"
    novel = {"L": False, "R": False}
    novel_left = {"L": rng.uniform(2, 6), "R": rng.uniform(2, 6)}
    posts = {"L": 0, "R": 0}
    prev_novel = dict(novel)
    reward_left = 0.0
    meta = {
        "condition": cond, "seed": seed, "fixture": True, "dt": dt, "duration_s": duration,
        "room": {"w_mm": ROOM_MM, "h_mm": ROOM_MM, "wall_mm": WALL_MM},
        "phone": {"L": [-79.0, 0.0, -55.5, 55.5], "R": [0.0, 79.0, -55.5, 55.5]},
        "body": {"v0": 12.0, "g_v": 0.0 if not steer else 6.0, "g_omega": 0.0 if not steer else 3.0, "scale": 10,
                 "body_len_mm": FLY_BODY_MM, "leg_reach_mm": LEG_REACH_MM, "leg_angle_deg": 35.0},
        "eye": {"grid": [24, 18], "d_min_mm": 5.0, "d_max_mm": 300.0},
        "lif": {"dt_ms": 1.0}, "plasticity": {"enabled": dopamine, "eta": 0.05, "tau_trace_s": 1.0},
        "reward": {"pam_hz": 100.0, "dur_s": 0.3}, "n_neurons": 166700,
        "sets": {"kc": 4064, "mbon": 97, "pam": 316, "ppl1": 16},
        "seed_l": 1 + seed, "seed_r": 2 + seed, "burst_hz": 1.5,
    }
    lines = [json.dumps({"meta": meta})]
    for s in range(T):
        t = s * dt
        if bout_left <= 0:
            bout_kind = "approach" if rng.random() < 0.45 else "wander"
            bout_left = float(rng.uniform(4, 12))
        bout_left -= dt
        # OU turning noise
        omega += (-omega / 0.5) * dt + 1.2 * math.sqrt(2 * dt / 0.5) * float(rng.normal())
        steer_w = 0.0
        if bout_kind == "approach" and not on_phone(x, y):
            ang = math.atan2(-y, -x)
            d = (ang - th + math.pi) % (2 * math.pi) - math.pi
            steer_w = 2.5 * d
        elif on_phone(x, y) and bout_kind == "approach":
            steer_w = -1.5 * omega   # linger: damp the turning
        w = omega + steer_w
        v0 = 12.0
        if stand_left > 0:
            v0 = 3.0
            stand_left -= dt
        elif on_phone(x, y) and rng.random() < 0.01:
            stand_left = float(rng.uniform(0.5, 2.5))
        r_dn = float(max(0.0, rng.normal(0.4, 0.3)))
        v = float(np.clip(v0 + (6.0 * r_dn if steer else 0.0) * 0.3, 0, 60))
        th += w * dt
        th = (th + math.pi) % (2 * math.pi) - math.pi
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        lim = ROOM_MM / 2 - WALL_MM - 8
        if abs(x) > lim or abs(y) > lim:
            nx_, ny_ = (-1.0 if x > lim else 1.0 if x < -lim else 0.0), (-1.0 if y > lim else 1.0 if y < -lim else 0.0)
            hx, hy = math.cos(th), math.sin(th)
            dot = hx * nx_ + hy * ny_
            hx, hy = hx - 2 * dot * nx_, hy - 2 * dot * ny_
            th = math.atan2(hy, hx) + float(rng.uniform(-0.52, 0.52))
            x, y = float(np.clip(x, -lim, lim)), float(np.clip(y, -lim, lim))
        tips = leg_tips(x, y, th)
        reach = {"L": panel_under(*tips["L"]), "R": panel_under(*tips["R"])}
        # novel cards: rising edges every few seconds per panel, visible ~1 s
        for side in ("L", "R"):
            novel_left[side] -= dt
            if novel_left[side] <= 0:
                novel[side] = not novel[side]
                novel_left[side] = float(rng.uniform(0.8, 1.4)) if novel[side] else float(rng.uniform(3, 8))
        # DN burst -> swipe
        burst = float(max(0.0, rng.normal(0.6, 0.6)))
        swipe: Optional[str] = None
        blocked: Optional[str] = None
        side_ev = float(rng.normal(0, 1))
        if burst > 1.5 and steer:
            leg = "L" if side_ev < 0 else "R"
            under = reach[leg]
            if under is not None:
                swipe = under
                posts[under] += int(rng.integers(1, 3))
                novel[under] = True if rng.random() < 0.3 else novel[under]
            else:
                blocked = leg
        onp = on_phone(x, y)
        # reward: rising edge of novel_visible on either panel while on the phone
        reward = onp and any(novel[sd] and not prev_novel[sd] for sd in ("L", "R"))
        if reward:
            reward_left = 0.3
        prev_novel = dict(novel)
        if dopamine and reward_left > 0:
            w_ratio = max(0.2, w_ratio * (1 - 0.012))
        reward_left -= dt
        spikes = int(rng.poisson(1400 + (900 if reward_left > 0 else 0)))
        pops = {"eye_L": float(rng.uniform(0.5, 4)), "eye_R": float(rng.uniform(0.5, 4)),
                "dn_L": float(rng.uniform(0, 1.5)), "dn_R": float(rng.uniform(0, 1.5)),
                "dna_L": float(rng.uniform(0, 1.5)), "dna_R": float(rng.uniform(0, 1.5)),
                "leg_L": float(rng.uniform(0, 1)), "leg_R": float(rng.uniform(0, 1)),
                "kc": float(rng.uniform(0.2, 1.0)), "mbon": float(rng.uniform(0.5, 3)),
                "pam": 100.0 if reward_left > 0 else float(rng.uniform(0, 0.5))}
        lines.append(json.dumps({
            "step": s, "t": round(t, 4), "pose": {"x": round(x, 3), "y": round(y, 3), "th": round(th, 4), "v": round(v, 3)},
            "on_phone": onp, "dist_mm": round(dist_to_phone(x, y), 3), "reach": reach, "swipe": swipe,
            "swipe_blocked": blocked, "burst_hz": round(burst, 3), "side_ev": round(side_ev, 3), "spikes": spikes,
            "pops": {kk: round(vv, 3) for kk, vv in pops.items()}, "reward": reward, "w_ratio": round(w_ratio, 5),
            "novel": dict(novel), "posts": dict(posts)}))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    if summary_path:
        vals = {"real": 0.21, "dopamine": 0.34, "shuffled": 0.12, "random": 0.09}
        summ = {"fixture": True, "per_condition": {
            c: {"time_on_phone_frac": {"mean": m, "ci": [max(0.0, m - 0.06), m + 0.07], "n": 5}} for c, m in vals.items()}}
        with open(summary_path, "w") as f:
            json.dump(summ, f, indent=1)


# --------------------------------------------------------------------------- #
# Background: flat plan view of the room + HUD band
# --------------------------------------------------------------------------- #
def build_arena_background(k: float, phone_rect_px: Tuple[int, int, int, int]) -> np.ndarray:
    """Dark navy floor with a flat 20 mm grid (brighter every 100 mm), 4 mm walls, vignette, grain,
    a soft spot light and the phone's flat drop shadow; a plain darker band below the arena."""
    W, H = int(DESIGN_W * k), int(DESIGN_H * k)
    A = int(ARENA_PX * k)
    bg = np.empty((H, W, 3), np.float32)
    bg[:] = FLOOR_OUTSIDE
    # HUD band: very dark with a faint vertical gradient
    ys = np.linspace(0, 1, H - A, dtype=np.float32)[:, None, None]
    bg[A:] = np.array([14, 12, 10], np.float32) * (1 - ys) + np.array([10, 9, 8], np.float32) * ys
    # room floor
    x0 = y0 = int(round(ROOM_INSET * k))
    x1 = y1 = int(round((ARENA_PX - ROOM_INSET) * k))
    floor = np.empty((y1 - y0, x1 - x0, 3), np.float32)
    floor[:] = FLOOR
    yy = np.linspace(-1, 1, y1 - y0, dtype=np.float32)[:, None]
    xx = np.linspace(-1, 1, x1 - x0, dtype=np.float32)[None, :]
    vig = 1.0 - 0.42 * np.clip(xx ** 2 + yy ** 2, 0, 1.4) / 1.4
    spot = np.exp(-(xx ** 2 + yy ** 2) / 0.35)
    floor *= vig[..., None]
    floor += (spot[..., None] * np.array([22, 18, 14], np.float32))
    grid = np.zeros((y1 - y0, x1 - x0), np.float32)
    for mm in range(0, int(ROOM_MM) + 1, 20):
        p = mm * PX_PER_MM * k
        major = 1.0 if mm % 100 == 0 else 0.45
        pi = int(round(p))
        cv2.line(grid, (pi, 0), (pi, y1 - y0), major, 1, cv2.LINE_AA)
        cv2.line(grid, (0, pi), (x1 - x0, pi), major, 1, cv2.LINE_AA)
    gcol = np.array(GRID_COLOR, np.float32)
    bloom = cv2.GaussianBlur(grid, (0, 0), 2.5 * max(k, 0.5))
    floor += grid[..., None] * gcol * 0.13 + bloom[..., None] * gcol * 0.10
    # centre cross-hair (phone origin) is implied by the major lines; add faint crease marker later
    bg[y0:y1, x0:x1] = floor
    # walls: a 4 mm band just outside the floor
    wall_px = max(2, int(round(WALL_MM * PX_PER_MM * k)))
    wall = np.zeros((H, W), np.float32)
    cv2.rectangle(wall, (x0 - wall_px, y0 - wall_px), (x1 + wall_px - 1, y1 + wall_px - 1), 1.0, -1)
    cv2.rectangle(wall, (x0, y0), (x1 - 1, y1 - 1), 0.0, -1)
    wcol = np.array(WALL_COLOR, np.float32)
    bg[:A] = bg[:A] * (1 - wall[:A, :, None] * 0.85) + wall[:A, :, None] * wcol * 0.55
    glow = cv2.GaussianBlur(wall, (0, 0), 6 * max(k, 0.5))
    bg[:A] += glow[:A, :, None] * wcol * 0.16
    # grain
    rng = np.random.default_rng(3)
    grain: np.ndarray = rng.normal(0, 2.6, (H, W)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), 0.8 * max(k, 0.5))
    bg += grain[..., None]
    # phone drop shadow (flat object: tight, slightly offset toward the bottom-right)
    px0, py0, px1, py1 = phone_rect_px
    sh = np.zeros((H, W), np.float32)
    off = int(round(4 * k))
    cv2.rectangle(sh, (px0 + off - int(3 * k), py0 + off - int(3 * k)), (px1 + off + int(3 * k), py1 + off + int(3 * k)), 1.0, -1)
    sh_w = cv2.GaussianBlur(sh, (0, 0), 16 * k)
    sh_c = cv2.GaussianBlur(sh, (0, 0), 5 * k)
    shadow = np.clip(0.5 * sh_w + 0.4 * sh_c, 0, 1)
    bg *= (1.0 - 0.75 * shadow[..., None])
    # separator between arena and HUD band
    cv2.line(bg, (0, A), (W, A), (40, 40, 48), max(1, int(round(2 * k))), cv2.LINE_AA)
    return np.clip(bg, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Fly in the plan view: rotated premultiplied sprite + procedural front legs
# --------------------------------------------------------------------------- #
class ArenaFly:
    """The photoreal sprite (via render.Fly in image mode) scaled to a 30 mm body and rotated per
    frame with cv2.warpAffine on the premultiplied colour + alpha stacked as one 4-channel image."""

    def __init__(self, k: float):
        self.k = k
        body_px = FLY_BODY_MM * PX_PER_MM * k                      # frame px, head top -> abdomen tip
        self.kf = body_px / FLY_IMAGE_BODY_LEN                       # render.Fly's k that yields that body length
        self.base = Fly(self.kf, (0.0, 0.0), style="image")
        pm, a8 = self.base.sprite_bgr, self.base.sprite_a
        h, w = a8.shape
        self.origin = np.array(self.base.img_origin, np.float32)    # thorax centre in sprite px
        self.rgba = np.dstack([pm, a8]).astype(np.float32)           # premultiplied colour + alpha
        # rotation canvas: big enough for any angle
        self.R = int(math.ceil(math.hypot(max(self.origin[0], w - self.origin[0]), max(self.origin[1], h - self.origin[1])))) + 2
        self.canvas = 2 * self.R + 1
        self._rot_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.mm2local = FLY_IMAGE_BODY_LEN / FLY_BODY_MM * self.kf   # mm -> frame px in the sprite's head-up frame
        # prothoracic legs are slimmer than the photo's (removed) forelegs at this scale
        self.leg_thick = tuple(v * 0.55 for v in self.base.leg_thick_front)

    def rotated(self, th: float) -> Tuple[np.ndarray, np.ndarray]:
        """(premultiplied bgr uint8, alpha uint8) of the sprite rotated to heading ``th``, thorax at the
        canvas centre.  Cached per 1 degree."""
        deg = int(round(math.degrees(th))) % 360
        hit = self._rot_cache.get(deg)
        if hit is not None:
            return hit
        # sprite head-up = heading +90 deg; positive cv2 angle rotates CCW on screen
        M = cv2.getRotationMatrix2D((float(self.origin[0]), float(self.origin[1])), deg - 90.0, 1.0)
        M[0, 2] += self.R - self.origin[0]
        M[1, 2] += self.R - self.origin[1]
        out = cv2.warpAffine(self.rgba, M, (self.canvas, self.canvas), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        pm = np.clip(out[..., :3], 0, 255).astype(np.uint8)
        a = np.clip(out[..., 3], 0, 255).astype(np.uint8)
        if len(self._rot_cache) > 400:
            self._rot_cache.clear()
        self._rot_cache[deg] = (pm, a)
        return pm, a

    def local_to_frame(self, pts: np.ndarray, th: float, cx: float, cy: float) -> np.ndarray:
        """Head-up sprite frame (x = fly's right, y = toward the abdomen, px) -> screen px at heading th."""
        al = th - math.pi / 2                                        # CCW-on-screen rotation
        ca, sa = math.cos(al), math.sin(al)
        x, y = pts[:, 0], pts[:, 1]
        return np.stack([cx + x * ca + y * sa, cy - x * sa + y * ca], axis=1).astype(np.float32)

    def tip_local(self, side: int, age: Optional[float]) -> Tuple[float, float]:
        """Front-leg tip in the head-up frame (px): rest = contract geometry (22 mm at +-35 deg);
        during a swipe: plant a little further out, drag toward the head, return."""
        s = self.mm2local
        rest = (side * LEG_REACH_MM * math.sin(LEG_ANGLE_RAD) * s, -LEG_REACH_MM * math.cos(LEG_ANGLE_RAD) * s)
        if age is None or age < 0 or age >= SWIPE_ANIM_S:
            return rest
        t_reach, t_drag = 0.10, 0.32
        plant = (rest[0] + side * 1.5 * s, rest[1] + 3.0 * s)     # reach a little forward/outward
        end = (plant[0] - side * 1.0 * s, plant[1] - 9.0 * s)     # drag ~9 mm toward the head (content scrolls up)
        if age < t_reach:
            p = ease_in_out(age / t_reach)
            return (rest[0] + (plant[0] - rest[0]) * p, rest[1] + (plant[1] - rest[1]) * p)
        if age < t_drag:
            p = ease_out((age - t_reach) / (t_drag - t_reach))
            return (plant[0] + (end[0] - plant[0]) * p, plant[1] + (end[1] - plant[1]) * p)
        p = ease_in_out((age - t_drag) / (SWIPE_ANIM_S - t_drag))
        return (end[0] + (rest[0] - end[0]) * p, end[1] + (rest[1] - end[1]) * p)

    def leg_points_local(self, side: int, tip: Tuple[float, float]) -> np.ndarray:
        """Attach (from the sprite JSON) -> knee -> ankle -> tip in the head-up frame; the knee bows
        outward/forward like a real prothoracic leg."""
        # attach at the front corners of the thorax (design px in render.Fly's head-up frame)
        A = np.array([side * 30.0 * self.kf, -22.0 * self.kf], np.float32)
        T = np.array(tip, np.float32)
        d = T - A
        L = float(np.hypot(*d)) + 1e-6
        u = d / L
        n = np.array([-u[1], u[0]], np.float32)
        if n[0] * side < 0:
            n = -n
        knee = A + u * (0.40 * L) + n * (0.22 * L)
        ankle = A + u * (0.78 * L) + n * (0.06 * L)
        return np.stack([A, knee, ankle, T])

    def draw(self, frame: np.ndarray, x_mm: float, y_mm: float, th: float,
             swipe_age: Dict[str, Optional[float]], contact: Dict[str, Optional[str]],
             blocked_age: Dict[str, Optional[float]]) -> None:
        k = self.k
        cx, cy = world_to_screen(x_mm, y_mm, k)
        pm, a = self.rotated(th)
        sx, sy = int(round(cx)) - self.R, int(round(cy)) - self.R
        # contact shadow (rotated alpha, blurred, offset)
        sh = cv2.GaussianBlur(a, (0, 0), 2.2 * max(k, 0.5))
        H, W = frame.shape[:2]
        ox, oy = sx + int(round(2 * k)), sy + int(round(3 * k))
        x0, y0, x1, y1 = max(0, ox), max(0, oy), min(W, ox + self.canvas), min(H, oy + self.canvas)
        if x1 > x0 and y1 > y0:
            roi = frame[y0:y1, x0:x1]
            wgt = sh[y0 - oy:y1 - oy, x0 - ox:x1 - ox].astype(np.float32) * (0.45 / 255.0)
            roi[:] = (roi.astype(np.float32) * (1.0 - wgt[..., None])).astype(np.uint8)
        # front legs (under the body)
        legs = {}
        for side, key in ((-1, "L"), (1, "R")):
            pts_l = self.leg_points_local(side, self.tip_local(side, swipe_age[key]))
            pts = self.local_to_frame(pts_l, th, cx, cy)
            legs[key] = pts
            self.base._draw_leg(frame, [pts[i] for i in range(4)], self.leg_thick)
        premul_blit(frame, pm, a, sx, sy)
        # contact rings (on top) in the swiped panel's colour; blocked swipe = small grey cross
        for key in ("L", "R"):
            age = swipe_age[key]
            tip = legs[key][3]
            if age is not None and 0.06 <= age < 0.40:
                al = clamp(1.0 - abs((age - 0.22) / 0.18), 0, 1)
                panel = contact.get(key) or key
                colr = COL_L if panel == "L" else COL_R
                cc = tuple(int(c * al + 40 * (1 - al)) for c in colr)
                cv2.circle(frame, _pt(tip), int(round((7 + 9 * (1 - al)) * k)), cc, max(1, int(round(2.2 * k))), cv2.LINE_AA)
                cv2.circle(frame, _pt(tip), max(1, int(round(2.2 * k))), colr, -1, cv2.LINE_AA)
            bage = blocked_age[key]
            if bage is not None and 0 <= bage < 0.5:
                al = clamp(1.0 - bage / 0.5, 0, 1)
                cc = tuple(int(c * al + 30 * (1 - al)) for c in (120, 120, 130))
                r = int(round(6 * k))
                p = _pt(tip)
                cv2.line(frame, (p[0] - r, p[1] - r), (p[0] + r, p[1] + r), cc, max(1, int(round(2 * k))), cv2.LINE_AA)
                cv2.line(frame, (p[0] - r, p[1] + r), (p[0] + r, p[1] - r), cc, max(1, int(round(2 * k))), cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #
class ArenaRenderer:
    def __init__(self, args, meta: dict, events: List[dict], spikes: Optional[dict],
                 positions: Optional[np.ndarray], groups: Optional[np.ndarray], summary: Optional[dict]):
        self.args = args
        self.k = 0.5 if args.preview else 1.0
        k = self.k
        self.W, self.H = int(DESIGN_W * k), int(DESIGN_H * k)
        self.frame_step = 2 if args.preview else 1
        self.events = events
        self.spikes = spikes
        self.meta = meta or {}
        self.summary = summary
        self.cond = str(self.meta.get("condition", "real"))
        self.seed = self.meta.get("seed", None)
        ev_end = (events[-1]["t"] + CTRL_DT) if events else 0.0
        dur = ev_end if args.duration is None else min(args.duration, ev_end)
        self.duration = max(dur, 0.5)
        self.n_frames = int(round(self.duration * FPS))
        n_default = int(positions.shape[0]) if positions is not None else 166700
        self.n_neurons = int(spikes["n"]) if spikes else _as_int(self.meta.get("n_neurons"), n_default)

        # ---- phone + feeds (to scale: panels 79 x 111 mm each, crease at x = 0) ----
        self.kd = k * DEVICE_K_RATIO
        G = max(3, int(round(14 * self.kd)))
        phone_w_px = PHONE_W_MM * PX_PER_MM * k
        fscale = ((phone_w_px - G) / 2.0) / 1335.0
        fmeta: Dict[str, object] = dict(self.meta["feeds"]) if isinstance(self.meta.get("feeds"), dict) else {}

        seed_l = _as_int(fmeta.get("seed_l", self.meta.get("seed_l")), 1)
        seed_r = _as_int(fmeta.get("seed_r", self.meta.get("seed_r")), 2)
        ap_kw = {key: (fmeta.get(key, self.meta.get(key))) for key in ("autoplay", "autoplay_amp", "autoplay_hz", "autoplay_whole")
                 if key in fmeta or key in self.meta}
        feed_mode = str(meta.get("feed_mode", "cards")) if meta else "cards"
        clips_dir = str(meta.get("clips_dir", "assets/clips")) if meta else "assets/clips"
        if not os.path.isabs(clips_dir) and not os.path.isdir(clips_dir):
            clips_dir = os.path.join(ROOT, clips_dir)
        self.feed_l, self.feed_r = make_pair(seed_l, seed_r, scale=fscale, mode=feed_mode, clips_dir=clips_dir, **ap_kw)
        pw, ph = self.feed_l.out_w, self.feed_l.out_h
        B = max(4, int(round(26 * self.kd)))
        Wc, Hc = 2 * pw + G + 2 * B, ph + 2 * B
        cx, cy = world_to_screen(0.0, 0.0, k)
        qx0, qy0 = int(round(cx - Wc / 2.0)), int(round(cy - Hc / 2.0))
        quad_frame = np.array([[qx0, qy0], [qx0 + Wc, qy0], [qx0 + Wc, qy0 + Hc], [qx0, qy0 + Hc]], np.float32)
        self.device = Device(self.kd, pw, ph, quad_frame / self.kd, self.feed_l.theme["hinge"])
        self.phone_rect = (qx0, qy0, qx0 + Wc, qy0 + Hc)
        # reward flash mask over the phone (soft rounded glow, a bit larger than the device)
        pad = int(round(60 * k))
        fx0, fy0 = qx0 - pad, qy0 - pad
        fw, fh = Wc + 2 * pad, Hc + 2 * pad
        m = np.zeros((fh, fw), np.float32)
        m[pad:pad + Hc, pad:pad + Wc] = rounded_mask(Hc, Wc, int(round(48 * self.kd))).astype(np.float32) / 255.0
        glow = cv2.GaussianBlur(m, (0, 0), 14 * k)
        # halo around the device + a faint tint on it (the feeds must stay readable)
        self.flash_mask = np.clip(glow * (1.0 - m) * 0.9 + m * 0.16, 0, 1)
        self.flash_roi = (fx0, fy0, fx0 + fw, fy0 + fh)

        self.text = TextRenderer(k)
        self.bg = build_arena_background(k, self.phone_rect)
        self.fly = ArenaFly(k)
        self.pip = BrainPiP(positions, groups, self.n_neurons, int(PIP_SIZE * k), k)
        self.pip_mask = rounded_mask(int(PIP_SIZE * k), int(PIP_SIZE * k), int(26 * k))

        # ---- storyboard timings scale with the run length ----
        f = min(1.0, self.duration / NOMINAL_LEN_S)
        D = self.duration
        self.t_title = (0.0, 4.0 * f)
        self.has_compare = self._summary_values() is not None
        self.t_compare = (D - 12.0 * f, D - 4.0 * f)
        self.t_attrib = (D - 4.0 * f, D)
        self.fade = max(f, 0.3)      # card fades shrink with the cards (never below 0.3x), as in render.py

        # ---- dynamic state ----
        self.ev_ptr = 0
        self.swipe_time = {"L": -1e9, "R": -1e9}          # per LEG
        self.swipe_panel: Dict[str, Optional[str]] = {"L": None, "R": None}
        self.blocked_time = {"L": -1e9, "R": -1e9}
        self.reward_time = -1e9
        self.pose = {"x": 0.0, "y": 0.0, "th": 0.0, "v": 0.0}
        self.on_phone = False
        self.dist_mm = 0.0
        self.time_on = 0.0
        self.visits = 0
        self.n_swipes = 0
        self.n_blocked = 0
        self.n_rewards = 0
        self.w_ratio = 1.0
        self.w_disp = 1.0
        self.last_t = 0.0
        self.trail: Deque[Tuple[float, float, float]] = deque()
        self.dist_series: List[float] = []
        self.dist_t: List[float] = []
        self.on_series: List[bool] = []            # on_phone per step, for the plot's on-phone marks
        self._off_steps: Optional[int] = None      # off-phone steps since the last on-phone step (None before any visit)
        self.visit_gap_steps = max(1, int(round(VISIT_DEBOUNCE_S / CTRL_DT)))
        self.dist_max = max(1.0, float(max((float(e.get("dist_mm", 0.0)) for e in events), default=200.0)))
        self.dist_max = max(self.dist_max, 120.0)

    # ---- summary ---------------------------------------------------------------------------
    def _summary_values(self) -> Optional[List[Tuple[str, float, Optional[Tuple[float, float]]]]]:
        if not self.summary or not isinstance(self.summary, dict):
            return None
        pc = self.summary.get("per_condition")
        if not isinstance(pc, dict):
            return None
        out: List[Tuple[str, float, Optional[Tuple[float, float]]]] = []
        for cond in COND_ORDER:
            d = pc.get(cond)
            if not isinstance(d, dict):
                continue
            tf = d.get("time_on_phone_frac", d.get("time_on_phone"))
            if not isinstance(tf, dict) or "mean" not in tf:
                continue
            try:
                mean = float(tf["mean"])
            except (TypeError, ValueError):
                continue
            ci: Optional[Tuple[float, float]] = None
            c = tf.get("ci", tf.get("ci95"))
            try:
                if isinstance(c, (list, tuple)) and len(c) == 2:
                    ci = (float(c[0]), float(c[1]))
                elif isinstance(c, (int, float)):
                    ci = (mean - float(c), mean + float(c))
            except (TypeError, ValueError):
                ci = None
            if math.isfinite(mean):
                out.append((cond, mean, ci))
        return out if out else None

    # ---- event application ------------------------------------------------------------------
    def apply_events(self, t: float) -> None:
        while self.ev_ptr < len(self.events) and self.events[self.ev_ptr]["t"] <= t + 1e-9:
            e = self.events[self.ev_ptr]
            self.ev_ptr += 1
            te = float(e["t"])
            pose = e.get("pose") or {}
            self.pose = {"x": float(pose.get("x", 0.0)), "y": float(pose.get("y", 0.0)),
                         "th": float(pose.get("th", 0.0)), "v": float(pose.get("v", 0.0))}
            onp = bool(e.get("on_phone", on_phone(self.pose["x"], self.pose["y"])))
            if onp:
                # a visit as analyze_arena.visits_debounced counts it: an on-phone run that starts
                # after an off-phone gap of >= 0.5 s (shorter gaps continue the previous visit)
                if self._off_steps is None or self._off_steps >= self.visit_gap_steps:
                    self.visits += 1
                self._off_steps = 0
                self.time_on += CTRL_DT
            elif self._off_steps is not None:
                self._off_steps += 1
            self.on_phone = onp
            self.dist_mm = float(e.get("dist_mm", dist_to_phone(self.pose["x"], self.pose["y"])))
            self.dist_series.append(self.dist_mm)
            self.dist_t.append(te)
            self.on_series.append(onp)
            reach = e.get("reach") or {}
            sw = e.get("swipe")
            if sw in ("L", "R"):
                (self.feed_l if sw == "L" else self.feed_r).swipe()
                # the leg whose tip is over the swiped panel (geometry decides), else same side
                legs = [lg for lg in ("L", "R") if reach.get(lg) == sw]
                leg = legs[0] if len(legs) == 1 else (sw if not legs or sw in legs else legs[0])
                self.swipe_time[leg] = te
                self.swipe_panel[leg] = sw
                self.n_swipes += 1
            bl = e.get("swipe_blocked")
            if bl in ("L", "R"):
                self.blocked_time[bl] = te
                self.n_blocked += 1
            if e.get("reward"):
                self.n_rewards += 1
                self.reward_time = te
            self.w_ratio = float(e.get("w_ratio", 1.0))
            sx, sy = world_to_screen(self.pose["x"], self.pose["y"], self.k)
            self.trail.append((te, sx, sy))
            while self.trail and self.trail[0][0] < te - TRAIL_S:
                self.trail.popleft()
            step = int(e["step"])
            if self.spikes is not None and 0 <= step < self.spikes["frames_packed"].shape[0]:
                row = np.unpackbits(self.spikes["frames_packed"][step])[: self.spikes["n"]].astype(bool)
                self.pip.deposit(row)
            self.last_t = te

    # ---- arena layers -----------------------------------------------------------------------
    def draw_trail(self, frame: np.ndarray, t: float) -> None:
        if len(self.trail) < 2:
            return
        k = self.k
        pts = np.array([(p[1], p[2]) for p in self.trail], np.float32)
        ages = t - np.array([p[0] for p in self.trail], np.float32)
        x0, y0 = np.floor(pts.min(axis=0)).astype(int) - int(6 * k)
        x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + int(6 * k)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.W, x1), min(int(ARENA_PX * k), y1)
        if x1 <= x0 or y1 <= y0:
            return
        mask = np.zeros((y1 - y0, x1 - x0), np.float32)
        local = pts - np.array([x0, y0], np.float32)
        # oldest first so the newest, most opaque segments win
        edges = np.linspace(TRAIL_S, 0.0, TRAIL_BUCKETS + 1)
        for b in range(TRAIL_BUCKETS):
            hi, lo = edges[b], edges[b + 1]
            sel = np.flatnonzero((ages <= hi) & (ages > lo - 1e-9))
            if sel.size == 0:
                continue
            i0, i1 = int(sel[0]), int(min(sel[-1] + 2, len(pts)))   # overlap one point with the next bucket
            a_mid = 1.0 - (hi + lo) / (2 * TRAIL_S)
            alpha = 0.85 * a_mid ** 1.6 + 0.03
            seg = np.round(local[i0:i1]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [seg], False, float(alpha), max(1, int(round(2.0 * k))), cv2.LINE_AA)
        roi = frame[y0:y1, x0:x1]
        col = np.array(COL_TRAIL, np.float32)
        roi[:] = np.clip(roi.astype(np.float32) * (1 - mask[..., None]) + col * mask[..., None], 0, 255).astype(np.uint8)

    def draw_reward_flash(self, frame: np.ndarray, t: float) -> None:
        age = t - self.reward_time
        if not (0 <= age < 0.9):
            return
        a = clamp(1.0 - age / 0.9, 0, 1) ** 1.5 * (0.5 + 0.5 * math.exp(-age / 0.12))
        x0, y0, x1, y1 = self.flash_roi
        H, W = frame.shape[:2]
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        m = self.flash_mask[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] * a
        roi = frame[cy0:cy1, cx0:cx1]
        roi[:] = np.clip(roi.astype(np.float32) + m[..., None] * np.array(COL_REWARD, np.float32) * 0.9, 0, 255).astype(np.uint8)
        # label above the phone
        px0, py0, px1, _ = self.phone_rect
        self.text.draw(frame, "REWARD · PAM 100 Hz", (px0 + px1) / 2 / self.k, py0 / self.k - 46 - 10 * ease_out(age / 0.9), 26,
                       COL_REWARD, True, align="center", alpha=clamp(1.0 - age / 0.9, 0, 1))

    def draw_arena_labels(self, frame: np.ndarray) -> None:
        T = self.text
        T.draw(frame, "400 × 400 mm arena · top view", 40, 18, 22, COL_DIM)
        T.draw(frame, f"fly scaled 10× (body {FLY_BODY_MM:.0f} mm)", 1040, 18, 22, COL_DIM, align="right")
        px0, py0, px1, py1 = self.phone_rect
        T.draw(frame, "L", (px0 + (px1 - px0) * 0.25) / self.k, py1 / self.k + 8, 22, COL_L, True, align="center")
        T.draw(frame, "R", (px0 + (px1 - px0) * 0.75) / self.k, py1 / self.k + 8, 22, COL_R, True, align="center")

    # ---- PiP + HUD --------------------------------------------------------------------------
    def draw_pip(self, frame: np.ndarray, t: float) -> None:
        k = self.k
        x, y = int(PIP_XY[0] * k), int(PIP_XY[1] * k)
        S = int(PIP_SIZE * k)
        img = self.pip.render(t)
        b = max(2, int(3 * k))
        fill_panel(frame, x - b, y - b, x + S + b, y + S + b, int(29 * k), (8, 8, 12), 0.96, border=(74, 80, 98))
        roi = frame[y:y + S, x:x + S]
        cv2.copyTo(img, self.pip_mask, roi)
        hs = int(64 * k)
        ramp = (1.0 - 0.72 * np.linspace(0, 1, hs, dtype=np.float32) ** 0.7)[:, None, None]
        strip = roi[S - hs:S]
        strip[:] = (strip.astype(np.float32) * ramp).astype(np.uint8)
        label = f"MaleCNS v1.0 · {self.n_neurons:,} neurons"
        self.text.draw(frame, label, PIP_XY[0] + 16, PIP_XY[1] + PIP_SIZE - 38, 24, COL_GREY)
        if self.spikes is not None:
            wl = self.text.draw(frame, "LIVE SPIKES", PIP_XY[0] + PIP_SIZE - 16, PIP_XY[1] + 14, 22, (120, 120, 235), True, align="right")
            cv2.circle(frame, (int((PIP_XY[0] + PIP_SIZE - 30) * k - wl), int((PIP_XY[1] + 28) * k)), int(6 * k), (90, 90, 250), -1, cv2.LINE_AA)
        else:
            self.text.draw(frame, "SOMA POSITIONS", PIP_XY[0] + PIP_SIZE - 16, PIP_XY[1] + 14, 22, COL_DIM, True, align="right")

    def draw_hud(self, frame: np.ndarray, t: float) -> None:
        T = self.text
        k = self.k
        label, sub = COND_LABELS.get(self.cond, (self.cond.upper(), ""))
        seed_txt = f" · seed {self.seed}" if self.seed is not None else ""
        T.draw(frame, "CONDITION" + seed_txt, HUD_X0, 1108, 22, COL_DIM)
        T.draw(frame, label, HUD_X0, 1134, 34, COL_WHITE, True)
        y = 1178
        for ln in T.wrap(sub, 20, HUD_X1 - HUD_X0)[:2]:
            T.draw(frame, ln, HUD_X0, y, 20, COL_GREY)
            y += 26
        # time on phone
        tt = self.last_t + CTRL_DT          # elapsed brain time incl. the current step (analysis: T = last t + dt)
        pct = 100.0 * self.time_on / tt
        T.draw(frame, "TIME ON PHONE", HUD_X0, 1240, 24, COL_GREY)
        colr = COL_ACCENT if self.on_phone else COL_WHITE
        w = T.draw(frame, f"{self.time_on:.1f} s", HUD_X0, 1266, 60, colr, True)
        T.draw(frame, f"{pct:.0f} %", HUD_X0 + w / k + 22, 1288, 34, COL_GREY, True)
        if self.on_phone:
            cv2.circle(frame, (int((HUD_X1 - 12) * k), int(1256 * k)), int(7 * k), COL_ACCENT, -1, cv2.LINE_AA)
            T.draw(frame, "ON PHONE", HUD_X1 - 28, 1244, 20, COL_ACCENT, True, align="right")
        # counters row
        row_y = 1350
        cols = ((HUD_X0, "VISITS", f"{self.visits}", COL_WHITE),
                (HUD_X0 + 170, "SWIPES", f"{self.n_swipes}", COL_WHITE),
                (HUD_X0 + 380, "REWARDS", f"{self.n_rewards}", COL_WHITE))
        for xx, lab, val, cc in cols:
            T.draw(frame, lab, xx, row_y, 22, COL_GREY)
            if lab == "REWARDS":
                age = t - self.reward_time
                fl = clamp(1.0 - age / 0.6, 0, 1) if age >= 0 else 0.0
                cc = tuple(int(c * (1 - fl) + r * fl) for c, r in zip(cc, COL_REWARD))
            if lab == "SWIPES":
                fl = max(clamp(1.0 - (t - self.swipe_time[s]) / 0.5, 0, 1) for s in ("L", "R"))
                cc = tuple(int(c * (1 - fl) + a * fl) for c, a in zip(cc, COL_ACCENT))
            wv = T.draw(frame, val, xx, row_y + 24, 46, cc, True)
            if lab == "SWIPES":
                T.draw(frame, f"· {self.n_blocked} blocked", xx + wv / k + 10, row_y + 42, 20, COL_DIM)
        # KC->MBON weight ratio bar
        self.w_disp += (self.w_ratio - self.w_disp) * 0.3
        by0 = 1450
        T.draw(frame, "KC→MBON WEIGHT  w / w₀", HUD_X0, by0, 22, COL_GREY)
        plastic = str(self.meta.get("condition", "")) == "dopamine" or self.w_ratio < 0.999
        T.draw(frame, f"{self.w_disp:.3f}" if plastic else "plasticity off", HUD_X1, by0, 22, COL_DOPA if plastic else COL_DIM, True, align="right")
        bx0, bx1, bh = HUD_X0, HUD_X1, 26
        fill_panel(frame, int(bx0 * k), int((by0 + 30) * k), int(bx1 * k), int((by0 + 30 + bh) * k), int(8 * k), (30, 30, 36), 0.9)
        frac = clamp(self.w_disp, 0.0, 1.0)
        wpx = int((bx1 - bx0 - 6) * frac)
        cc = COL_DOPA if plastic else (90, 90, 100)
        if wpx > 4:
            fill_panel(frame, int((bx0 + 3) * k), int((by0 + 33) * k), int((bx0 + 3 + wpx) * k), int((by0 + 27 + bh) * k), int(6 * k), cc, 0.95)
        # floor tick at 0.2 w0 (rule bound)
        xt = int((bx0 + 3 + (bx1 - bx0 - 6) * 0.2) * k)
        cv2.line(frame, (xt, int((by0 + 32) * k)), (xt, int((by0 + 28 + bh) * k)), (120, 120, 130), max(1, int(2 * k)), cv2.LINE_AA)
        T.draw(frame, "0.2 (floor)", bx0 + (bx1 - bx0) * 0.2 - 4, by0 + 30 + bh + 4, 18, COL_DIM, align="center")
        T.draw(frame, "1.0", bx1, by0 + 30 + bh + 4, 18, COL_DIM, align="right")
        # distance plot
        self.draw_distance_plot(frame, t)
        T.draw(frame, "eyes see the room · the brain steers · swipes only from a leg on a panel", 540, 1840, 26, COL_DIM, align="center")

    def draw_distance_plot(self, frame: np.ndarray, t: float) -> None:
        T = self.text
        k = self.k
        x0, y0, x1, y1 = int(60 * k), int(PLOT_Y0 * k), int(1020 * k), int(PLOT_Y1 * k)
        fill_panel(frame, x0, y0, x1, y1, int(18 * k), (14, 14, 18), 0.85, border=(50, 50, 60))
        T.draw(frame, f"DISTANCE TO PHONE · 0–{self.dist_max:.0f} mm · on-phone stretches marked", 80, PLOT_Y0 + 8, 22, COL_GREY)
        T.draw(frame, f"{self.dist_mm:.0f} mm", 1000, PLOT_Y0 + 8, 22, COL_WHITE, True, align="right")
        px0, px1 = x0 + int(14 * k), x1 - int(14 * k)
        py0, py1 = y0 + int(40 * k), y1 - int(14 * k)
        n = len(self.dist_series)
        if n >= 2:
            vals = np.asarray(self.dist_series, np.float32)
            ts = np.asarray(self.dist_t, np.float32)
            ons = np.asarray(self.on_series, bool)
            xs = px0 + ts / max(self.duration, 1e-6) * (px1 - px0)
            ys = py1 - np.clip(vals / self.dist_max, 0, 1) * (py1 - py0)
            # subsample to at most one point per pixel column
            if n > (px1 - px0):
                idx = np.unique(np.round(np.linspace(0, n - 1, px1 - px0)).astype(int))
                xs, ys, vals, ons = xs[idx], ys[idx], vals[idx], ons[idx]
            pts = np.int32(np.round(np.stack([xs, ys], axis=1))).reshape(-1, 1, 2)
            # on-phone band: fill under the curve where distance == 0 reads as time on the phone
            roi = frame[y0:y1, x0:x1]
            over = roi.copy()
            poly = np.vstack([pts, np.array([[[pts[-1, 0, 0], py1]], [[pts[0, 0, 0], py1]]], np.int32)])
            cv2.fillPoly(over, [poly - np.array([x0, y0], np.int32)], (70, 60, 44), cv2.LINE_AA)
            cv2.addWeighted(roi, 0.7, over, 0.3, 0, dst=roi)
            cv2.polylines(frame, [pts], False, (210, 190, 140), max(1, int(round(2 * k))), cv2.LINE_AA)
            # highlight on-phone stretches along the baseline (the logged flag: dist_mm is the distance
            # to the phone centre in real logs, so it never reaches 0 on the phone)
            onp = ons
            if onp.any():
                bl = np.zeros(pts.shape[0], bool)
                bl[:] = onp
                xs_i = pts[:, 0, 0]
                start = None
                for i in range(len(bl) + 1):
                    if i < len(bl) and bl[i] and start is None:
                        start = i
                    elif (i == len(bl) or not bl[i]) and start is not None:
                        cv2.line(frame, (int(xs_i[start]), py1 + int(3 * k)), (int(xs_i[i - 1]) + 1, py1 + int(3 * k)),
                                 COL_ACCENT, max(1, int(round(4 * k))), cv2.LINE_AA)
                        start = None
            cv2.circle(frame, tuple(int(v) for v in pts[-1, 0]), int(5 * k), COL_WHITE, -1, cv2.LINE_AA)
        T.draw(frame, "0 s", 80, PLOT_Y1 - 30, 16, COL_DIM)
        T.draw(frame, f"{self.duration:.0f} s", 1000, PLOT_Y1 - 30, 16, COL_DIM, align="right")

    # ---- cards -------------------------------------------------------------------------------
    def draw_card(self, frame: np.ndarray, blocks: List[Tuple[str, float, Tuple[int, int, int], bool]],
                  alpha: float, y_center: float, max_width: float = 900, pad: float = 48) -> None:
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

    def draw_compare_card(self, frame: np.ndarray, alpha: float) -> None:
        vals = self._summary_values()
        if not vals or alpha <= 0.001:
            return
        T = self.text
        k = self.k
        W_card, pad = 940, 44
        row_h, n = 74, len(vals)
        card_h = pad * 2 + 46 + 34 + 24 + n * row_h + 30
        x0 = (DESIGN_W - W_card) / 2
        y0 = 1000 - card_h / 2
        fill_panel(frame, int(x0 * k), int(y0 * k), int((x0 + W_card) * k), int((y0 + card_h) * k), int(36 * k),
                   (10, 10, 14), 0.86 * alpha, border=(int(80 * alpha), int(80 * alpha), int(92 * alpha)))
        T.draw(frame, "Time on the phone", DESIGN_W / 2, y0 + pad, 44, COL_WHITE, True, align="center", alpha=alpha)
        n_seeds: Optional[int] = None
        pc = (self.summary or {}).get("per_condition", {})
        for cnd, _, _ in vals:
            d = pc.get(cnd, {}) if isinstance(pc, dict) else {}
            tf = d.get("time_on_phone_frac", d.get("time_on_phone")) if isinstance(d, dict) else None
            for cand in ((tf or {}).get("n") if isinstance(tf, dict) else None, d.get("n_runs") if isinstance(d, dict) else None,
                         d.get("n") if isinstance(d, dict) else None):
                if isinstance(cand, int) and not isinstance(cand, bool):
                    n_seeds = cand
                    break
        sub = "% of the run · mean ± 95 % bootstrap CI" + (f" across {n_seeds} seeds" if n_seeds else " across seeds")
        T.draw(frame, sub, DESIGN_W / 2, y0 + pad + 54, 26, COL_GREY, align="center", alpha=alpha)
        # bars: one axis, thin marks anchored at 0, direct value labels in text ink, CI whiskers
        lx = x0 + pad
        bx0 = lx + 230
        bx1 = x0 + W_card - pad - 130
        vmax = max(max(v for _, v, _ in vals), max((c[1] for _, _, c in vals if c), default=0.0), 0.05)
        axis_max = math.ceil(vmax * 100 / 10.0) * 10.0 / 100.0
        axis_max = min(max(axis_max, 0.1), 1.0)
        ytop = y0 + pad + 54 + 34 + 24
        for i, (cnd, mean, ci) in enumerate(vals):
            ry = ytop + i * row_h
            is_cur = cnd == self.cond
            lab_col = COL_WHITE if is_cur else COL_GREY
            T.draw(frame, COND_SHORT.get(cnd, cnd), lx, ry + 14, 30, lab_col, is_cur, alpha=alpha)
            if is_cur:
                T.draw(frame, "▶ this run", lx, ry + 46, 18, COL_ACCENT, alpha=alpha)
            fill_panel(frame, int(bx0 * k), int((ry + 12) * k), int(bx1 * k), int((ry + 42) * k), int(6 * k), (26, 26, 32), 0.9 * alpha)
            frac = clamp(mean / axis_max, 0, 1)
            wpx = (bx1 - bx0) * frac
            bar_col = COL_ACCENT if is_cur else (160, 160, 170)
            if wpx > 2:
                fill_panel(frame, int(bx0 * k), int((ry + 12) * k), int((bx0 + wpx) * k), int((ry + 42) * k), int(6 * k), bar_col, 0.95 * alpha)
            if ci is not None:
                cx0 = bx0 + (bx1 - bx0) * clamp(ci[0] / axis_max, 0, 1)
                cx1 = bx0 + (bx1 - bx0) * clamp(ci[1] / axis_max, 0, 1)
                cy = ry + 27
                wc = tuple(int(c * alpha) for c in (235, 235, 240))
                cv2.line(frame, (int(cx0 * k), int(cy * k)), (int(cx1 * k), int(cy * k)), wc, max(1, int(round(2 * k))), cv2.LINE_AA)
                for cx_ in (cx0, cx1):
                    cv2.line(frame, (int(cx_ * k), int((cy - 8) * k)), (int(cx_ * k), int((cy + 8) * k)), wc, max(1, int(round(2 * k))), cv2.LINE_AA)
            txt = f"{100 * mean:.0f} %"
            if ci is not None:
                txt += f"  [{100 * ci[0]:.0f}–{100 * ci[1]:.0f}]"
            T.draw(frame, txt, bx1 + 14, ry + 15, 24, COL_WHITE if is_cur else COL_GREY, is_cur, alpha=alpha)
        # axis
        ay = ytop + n * row_h + 4
        ac = tuple(int(c * alpha) for c in (70, 70, 80))
        cv2.line(frame, (int(bx0 * k), int(ay * k)), (int(bx1 * k), int(ay * k)), ac, max(1, int(round(1.5 * k))), cv2.LINE_AA)
        for frac_t in (0.0, 0.5, 1.0):
            xx = bx0 + (bx1 - bx0) * frac_t
            T.draw(frame, f"{100 * axis_max * frac_t:.0f} %", xx, ay + 6, 18, COL_DIM, align="center", alpha=alpha)

    def draw_cards(self, frame: np.ndarray, t: float) -> None:
        fi, fo = 0.45 * self.fade, 0.6 * self.fade
        a = card_alpha(t, *self.t_title, fade_in=fi, fade_out=fo)
        if a > 0:
            self.draw_card(frame, [(TITLE, 68, COL_WHITE, True), ("", 18, COL_WHITE, False), (SUBTITLE, 34, COL_GREY, False)],
                           a, y_center=1500, max_width=980)
        if self.has_compare:
            a = card_alpha(t, *self.t_compare, fade_in=fi, fade_out=fo)
            if a > 0:
                self.draw_compare_card(frame, a)
        a = card_alpha(t, self.t_attrib[0], self.t_attrib[1] + 1.0, fade_in=fi, fade_out=0.0)
        if a > 0:
            self.draw_card(frame, [("CREDITS", 30, COL_GREY, True), ("", 12, COL_WHITE, False), (ATTRIBUTION, 30, COL_WHITE, False)],
                           a, y_center=1000, max_width=960, pad=40)

    # ---- frame -------------------------------------------------------------------------------
    def render_frame(self, idx: int) -> np.ndarray:
        t = idx / FPS
        self.apply_events(t)
        frame = self.bg.copy()
        self.device.draw(frame, self.feed_l.render(), self.feed_r.render())
        self.draw_reward_flash(frame, t)
        self.draw_trail(frame, t)
        swipe_age: Dict[str, Optional[float]] = {}
        blocked_age: Dict[str, Optional[float]] = {}
        for side in ("L", "R"):
            age = t - self.swipe_time[side]
            swipe_age[side] = age if 0 <= age < SWIPE_ANIM_S else None
            bage = t - self.blocked_time[side]
            blocked_age[side] = bage if 0 <= bage < 0.5 else None
        self.fly.draw(frame, self.pose["x"], self.pose["y"], self.pose["th"], swipe_age, self.swipe_panel, blocked_age)
        self.draw_arena_labels(frame)
        self.pip.decay()
        self.draw_pip(frame, t)
        self.draw_hud(frame, t)
        self.draw_cards(frame, t)
        return frame

    def step_feeds(self) -> None:
        self.feed_l.step(1.0 / FPS)
        self.feed_r.step(1.0 / FPS)

    def iter_frames(self, stills=frozenset()):
        """Advance feeds / events / PiP exactly as the video loop does and yield (idx, frame) for every
        rendered frame (skipped preview frames and, with --no-video, all non-still frames only step the state)."""
        no_video = bool(getattr(self.args, "no_video", False))
        for idx in range(self.n_frames):
            self.step_feeds()
            if idx == self.n_frames - 1:
                # the last control step ends inside the final frame: apply it (events with t < n_frames / FPS)
                self.apply_events(self.n_frames / FPS - 1e-6)
            if (idx % self.frame_step != 0 or no_video) and idx not in stills:
                self.apply_events(idx / FPS)
                self.pip.decay()
                continue
            yield idx, self.render_frame(idx)

    def run(self) -> None:
        import imageio.v2 as imageio
        args = self.args
        out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
        os.makedirs(out_dir, exist_ok=True)
        out_fps = FPS / self.frame_step
        stills = parse_stills(args.stills, self.n_frames, self.duration)
        writer = None if args.no_video else imageio.get_writer(
            args.out, fps=out_fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=1,
            ffmpeg_params=["-preset", args.preset])
        t_start = time.perf_counter()
        n_written = n_rendered = 0
        next_report = 5.0
        print(f"rendering {self.n_frames} frames ({self.duration:.1f} s) at {self.W}x{self.H}, condition={self.cond}, "
              f"writing every {self.frame_step} frame(s) to {args.out}", flush=True)
        try:
            for idx, frame in self.iter_frames(stills):
                n_rendered += 1
                if idx in stills:
                    name = f"arena_still_{idx / FPS:g}s.png"
                    cv2.imwrite(os.path.join(out_dir, name), frame)
                if writer is not None and idx % self.frame_step == 0:
                    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    n_written += 1
                t_video = (idx + 1) / FPS
                if t_video >= next_report or idx == self.n_frames - 1:
                    now = time.perf_counter()
                    fps_wall = n_rendered / max(now - t_start, 1e-6)
                    eta = (self.n_frames - idx - 1) / self.frame_step / max(fps_wall, 1e-6)
                    print(f"  t={t_video:5.1f}s  frames={n_rendered}  {fps_wall:5.1f} fps  eta {eta:5.1f}s", flush=True)
                    next_report += 5.0
        finally:
            if writer is not None:
                writer.close()
        elapsed = time.perf_counter() - t_start
        if writer is not None:
            print(f"wrote {args.out}: {n_written} frames in {elapsed:.1f}s ({n_rendered / max(elapsed, 1e-6):.1f} fps)")
        else:
            print(f"rendered {n_rendered} still(s) in {elapsed:.1f}s")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", default=None, help="out/arena_<cond>_s<seed>.jsonl (required unless --fixture)")
    ap.add_argument("--spikes", default=None, help="optional out/arena_<cond>_s<seed>_spikes.npz for the PiP")
    ap.add_argument("--positions", default=os.path.join(ROOT, "data", "graph", "positions.npy"))
    ap.add_argument("--groups", default=os.path.join(ROOT, "data", "graph", "groups.npy"))
    ap.add_argument("--summary", default=os.path.join(ROOT, "out", "arena_summary.json"),
                    help="analyze_arena.py summary for the comparison card (card omitted when missing)")
    ap.add_argument("--out", default=None, help="mp4 path (default out/arena_<cond>_s<seed>.mp4)")
    ap.add_argument("--duration", type=float, default=None, help="render at most this many seconds")
    ap.add_argument("--preview", action="store_true", help="540x960, every other frame (30 fps output)")
    ap.add_argument("--stills", default=None, help="comma-separated video times (s) saved as arena_still_*.png next to --out")
    ap.add_argument("--no-video", action="store_true", help="only write --stills (no mp4)")
    ap.add_argument("--fixture", action="store_true", help="generate + render a synthetic log (out/arena_fixture.jsonl)")
    ap.add_argument("--fixture-cond", default="dopamine", choices=COND_ORDER)
    ap.add_argument("--fixture-duration", type=float, default=60.0)
    ap.add_argument("--fixture-seed", type=int, default=0)
    ap.add_argument("--preset", default="fast", help="x264 preset")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = os.path.join(ROOT, "out")
    ev_path = args.events
    summary_path = args.summary
    if args.fixture:
        fx_dir = os.path.dirname(os.path.abspath(args.out)) if args.out else out_dir
        os.makedirs(fx_dir, exist_ok=True)
        ev_path = os.path.join(fx_dir, "arena_fixture.jsonl")
        fx_summary = os.path.join(fx_dir, "arena_fixture_summary.json")
        print(f"generating fixture {ev_path} ({args.fixture_cond}, {args.fixture_duration:.0f} s)")
        make_fixture(ev_path, cond=args.fixture_cond, seed=args.fixture_seed, duration=args.fixture_duration,
                     summary_path=fx_summary)
        if not (summary_path and os.path.exists(summary_path)):
            summary_path = fx_summary
        if args.out is None:
            args.out = os.path.join(fx_dir, "arena_fixture.mp4")
    if not ev_path or not os.path.exists(ev_path):
        print(f"missing events log {ev_path!r} (pass --events or --fixture)", file=sys.stderr)
        return 2
    meta, events = load_events(ev_path)
    if not events:
        print("no events", file=sys.stderr)
        return 2
    if args.out is None:
        base = os.path.splitext(os.path.basename(ev_path))[0]
        args.out = os.path.join(out_dir, base + ".mp4")
    spikes = load_spikes(args.spikes)
    if args.spikes and spikes is None:
        print(f"warning: spikes file {args.spikes} not found; PiP shows the base cloud only", file=sys.stderr)
    positions = np.load(args.positions) if args.positions and os.path.exists(args.positions) else None
    groups = np.load(args.groups) if args.groups and os.path.exists(args.groups) else None
    if positions is None:
        print("warning: positions missing; PiP will be empty", file=sys.stderr)
    if not (os.path.exists(FLY_SPRITE_PNG) and os.path.exists(FLY_SPRITE_JSON)):
        print(f"missing fly sprite {FLY_SPRITE_PNG} / .json (run src/build_fly_sprite.py)", file=sys.stderr)
        return 2
    summary: Optional[dict] = None
    if summary_path and os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                loaded = json.load(f)
            summary = loaded if isinstance(loaded, dict) else None
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: cannot read summary {summary_path}: {exc}; comparison card omitted", file=sys.stderr)
    r = ArenaRenderer(args, meta, events, spikes, positions, groups, summary)
    if not r.has_compare:
        print("note: no usable --summary; comparison card omitted")
    r.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

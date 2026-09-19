#!/usr/bin/env python3
"""Build the photoreal fly sprite used by ``src/render.py`` (``--fly image``).

Input:  assets/fly_reference.jpg   photoreal top-down Drosophila render on a near-black
                                   background (head up, wings folded, legs spread)
Output: assets/fly_sprite.png      RGBA, **premultiplied over black**: the colour channels
                                   are the (background-corrected) source pixels, so a
                                   compositor must use  out = rgb + (1 - alpha) * background
        assets/fly_sprite.json     anchors in sprite pixel coordinates (see ``anchors`` below)
        out/fly_sprite_check.png   sprite composited over a grey/blue checker (matte check)
        out/fly_sprite_debug.png   matte / inpaint band / anchors overlaid on the source

Steps
  a. alpha matte: alpha = 1 for the saturated tan body, legs and eyes (saturation + luminance
     threshold, small morphological close, small-hole fill so dark bristles stay opaque),
     luminance-driven alpha on a 3 px ring outside that mask (anti-aliased edges, dark leg
     rims), and clip(lum / 190, 0.35, 1) for the neutral grey, translucent wing membrane;
     everything is feathered by ~1.5 px.
  b. the FRONT legs (the upper pair, along the wing edges to the image sides) are painted out
     with cv2.inpaint along hand-picked polylines so the renderer can draw animated ones from
     the recorded thorax attach points.
  c. crop to the alpha bounding box (+ margin) and save.

Coordinates below are pixels of the 944 x 2096 source.  The symmetry axis is x = 473.5.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "assets", "fly_reference.jpg")
OUT_PNG = os.path.join(ROOT, "assets", "fly_sprite.png")
OUT_JSON = os.path.join(ROOT, "assets", "fly_sprite.json")
OUT_DIR = os.path.join(ROOT, "out")

AXIS_X = 473.5                      # mirror: x' = 947 - x

# --- hand-picked anchors (source px) --------------------------------------------------
HEAD_CENTER = (473.5, 668.0)        # between the eyes
BODY_CENTER = (473.5, 745.0)        # thorax (notum) centre; maps onto the renderer's FLY_CENTER
# thorax attach points of the front legs (hidden under the wing bases)
FRONT_ATTACH_L = (302.0, 903.0)
# the visible front leg, base -> tip (x, y, mask radius).  Coxa knob, femur along the wing
# edge, tibia, tarsus out to the image side.
FRONT_LEG_L: List[Tuple[float, float, float]] = [
    (262, 916, 15), (245, 936, 19), (222, 958, 17), (208, 990, 16), (201, 1015, 16),
    (190, 1050, 14), (177, 1085, 13), (168, 1115, 12), (155, 1150, 13), (132, 1180, 13),
    (113, 1202, 13), (90, 1222, 13), (66, 1240, 13), (52, 1254, 14),
]
# joints of the visible front leg: knee (femur/tibia) and ankle (tibia/tarsus) and tip
FRONT_KNEE_L = (201.0, 1015.0)
FRONT_ANKLE_L = (155.0, 1150.0)
FRONT_TIP_L = (52.0, 1254.0)
# leg widths measured on the image (px): femur, tibia, tarsus
FRONT_LEG_WIDTH = (24.0, 14.0, 10.0)

MARGIN = 6                          # crop margin (px)


def mirror(p):
    return (2 * AXIS_X - p[0], p[1])


def fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    """Fill enclosed background components smaller than ``max_area`` (dark bristles, bands)."""
    inv = (mask == 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = mask.copy()
    h, w = mask.shape
    for i in range(1, n):
        x, y, ww, hh, area = st[i]
        touches = x == 0 or y == 0 or x + ww >= w or y + hh >= h
        if area <= max_area and not touches:
            out[lab == i] = 1
    return out


def leg_mask(shape: Tuple[int, int], pts: List[Tuple[float, float, float]]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for (x0, y0, r0), (x1, y1, r1) in zip(pts[:-1], pts[1:]):
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) / 2))
        for t in np.linspace(0, 1, n):
            cv2.circle(m, (int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t))),
                       int(round(r0 + (r1 - r0) * t)), 255, -1, cv2.LINE_AA)
    return m


def main() -> int:
    img = cv2.imread(SRC, cv2.IMREAD_COLOR)
    if img is None:
        print(f"missing {SRC}", file=sys.stderr)
        return 2
    H, W = img.shape[:2]
    os.makedirs(OUT_DIR, exist_ok=True)

    # background level (the "black" is ~ (16, 12, 11) BGR)
    corners = np.concatenate([img[:60, :60].reshape(-1, 3), img[:60, -60:].reshape(-1, 3),
                              img[-60:, :60].reshape(-1, 3), img[-60:, -60:].reshape(-1, 3)])
    bg_level = corners.mean(0).astype(np.float32)

    # ---- b. paint out the front legs -------------------------------------------------
    legs_l = FRONT_LEG_L
    legs_r = [(2 * AXIS_X - x, y, r) for x, y, r in FRONT_LEG_L]
    band = cv2.bitwise_or(leg_mask((H, W), legs_l), leg_mask((H, W), legs_r))
    # widen by 5 px so the legs' bristle fringe and dark rims go too (remnants would otherwise
    # show as tan flecks along the wing edge)
    band_bin = cv2.dilate((band > 0).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    # The legs run just OUTSIDE the wings' beaded rims, so keep the wing itself out of the band:
    # per row, scanning inward from the leg's centre line, the wing interior starts where the
    # bright, neutral membrane starts (>= 3 consecutive px with low saturation); we keep the
    # rim by backing off 4 px.
    lum_src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sat_src = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    membrane = (lum_src > 110) & (sat_src < 70)
    wing_interior = np.zeros((H, W), np.uint8)

    def leg_cx(y: float) -> float:
        for (xa, ya, _), (xb, yb, _) in zip(legs_l[:-1], legs_l[1:]):
            if ya <= y <= yb:
                return xa + (xb - xa) * (y - ya) / (yb - ya)
        return -1.0

    for y in range(int(legs_l[0][1]) - 6, int(legs_l[-1][1]) + 7):
        c = leg_cx(y)
        if c < 0:
            c = legs_l[0][0] if y < legs_l[0][1] else legs_l[-1][0]
        c_i = int(round(c))
        for side in (-1, 1):
            x_start = c_i if side < 0 else int(round(2 * AXIS_X - c))
            edge = -1
            for x in range(x_start, x_start + (66 if side < 0 else -66), 1 if side < 0 else -1):
                if 0 <= x < W and 0 <= x + 2 * (1 if side < 0 else -1) < W:
                    seg = membrane[y, min(x, x + 2 * (1 if side < 0 else -1)) : max(x, x + 2 * (1 if side < 0 else -1)) + 1]
                    if seg.all():
                        edge = x
                        break
            if edge >= 0:
                if side < 0:
                    wing_interior[y, max(0, edge - 4) : int(AXIS_X)] = 1
                else:
                    wing_interior[y, int(AXIS_X) : min(W, edge + 5)] = 1
    band_bin[wing_interior > 0] = 0
    clean = cv2.inpaint(img, band_bin, 5, cv2.INPAINT_TELEA)

    # ---- a. alpha matte --------------------------------------------------------------
    hsv = cv2.cvtColor(clean, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    lum = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    lum_f = np.clip(lum.astype(np.float32) - float(bg_level.mean()), 0, 255)

    body: np.ndarray = ((sat > 90) & (lum > 40)).astype(np.uint8)
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    body = fill_small_holes(body, max_area=600)
    # drop specks (bristle tips, noise) that are not connected to anything sizeable
    n, lab, st, _ = cv2.connectedComponentsWithStats(body, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, 4] >= 60
    body = keep[lab].astype(np.uint8)

    # silhouette: the source luminance; the (background-side) leg band is not part of it
    fg = ((lum_src > 32) & (band_bin == 0)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    fg = fill_small_holes(fg, max_area=400)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, 4] >= 200
    fg = keep[lab].astype(np.uint8)

    ring = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) & (1 - body)
    wing = fg & (1 - body) & (1 - ring)

    alpha = np.zeros((H, W), np.float32)
    alpha[body > 0] = 1.0
    alpha[ring > 0] = np.clip(lum_f[ring > 0] / 70.0, 0.0, 1.0)
    alpha[wing > 0] = np.clip(lum_f[wing > 0] / 190.0, 0.35, 1.0)
    # dark, low-saturation pixels well inside the silhouette are shadowed body (neck, wing
    # hinges, the gap under the scutellum), not membrane: the darker, the more opaque
    interior = cv2.erode(fill_small_holes(fg, max_area=10 ** 7),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    dark = (wing > 0) & (interior > 0)
    alpha[dark] = np.maximum(alpha[dark], np.clip(1.0 - lum_f[dark] / 80.0, 0.0, 1.0))
    # the leg band we painted out must not leave an alpha imprint outside the wing
    alpha[(band_bin > 0) & (fg == 0)] = 0.0

    # premultiplied colour over black: C = a*col + (1-a)*bg  ->  a*col = C - (1-a)*bg
    pm = clean.astype(np.float32) - (1.0 - alpha)[..., None] * bg_level[None, None, :]
    pm = np.clip(pm, 0, 255)
    # feather ~1.5 px (same kernel on colour and alpha keeps the premultiplied pair consistent)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0.7)
    pm = cv2.GaussianBlur(pm, (5, 5), 0.7)
    pm = np.minimum(pm, 255.0 * alpha[..., None] + 1e-3)   # premultiplied invariant: rgb <= alpha

    # ---- c. crop ---------------------------------------------------------------------
    a_ys, a_xs = np.where(alpha > 0.02)
    x0, x1 = max(0, int(a_xs.min()) - MARGIN), min(W, int(a_xs.max()) + MARGIN + 1)
    y0, y1 = max(0, int(a_ys.min()) - MARGIN), min(H, int(a_ys.max()) + MARGIN + 1)
    a8 = np.clip(alpha[y0:y1, x0:x1] * 255 + 0.5, 0, 255).astype(np.uint8)
    c8 = np.clip(pm[y0:y1, x0:x1] + 0.5, 0, 255).astype(np.uint8)
    rgba = np.dstack([c8, a8])          # BGRA in memory -> PNG RGBA
    cv2.imwrite(OUT_PNG, rgba)

    def sp(p) -> List[float]:
        return [round(float(p[0]) - x0, 1), round(float(p[1]) - y0, 1)]

    # eye centres from the orange hue (measured, not guessed)
    hue = hsv[..., 0]
    eyes: np.ndarray = (((hue < 14) | (hue > 170)) & (sat > 150) & (hsv[..., 2] > 120)).astype(np.uint8)
    eyes = cv2.morphologyEx(eyes, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(eyes)
    big = sorted(range(1, n), key=lambda i: -st[i, 4])[:2]
    if len(big) < 2:
        print(f"eye detection found {len(big)} orange blob(s) in {SRC}; need two compound eyes "
              "(check the hue/saturation thresholds for this reference image)", file=sys.stderr)
        return 2
    eye_pts = sorted((tuple(cen[i]) for i in big), key=lambda p: p[0])
    eye_l, eye_r = eye_pts[0], eye_pts[1]

    # body bbox: opaque body mask near the axis (head to abdomen tip, no legs / wings)
    bys, bxs = np.where(body[:, int(AXIS_X) - 130 : int(AXIS_X) + 130] > 0)
    body_bbox = [int(bxs.min() + int(AXIS_X) - 130), int(bys.min()), int(bxs.max() + int(AXIS_X) - 130), int(bys.max())]
    body_len = body_bbox[3] - body_bbox[1]

    # leg colours sampled from the remaining (mid/hind) legs: lit core, mid tone, dark rim
    leg_px = clean[1290:1520, 130:260][body[1290:1520, 130:260] > 0].astype(np.float32)
    lum_leg = leg_px @ np.array([0.114, 0.587, 0.299], np.float32)
    order = np.argsort(lum_leg)
    n_ = len(order)
    col_dark = leg_px[order[: n_ // 5]].mean(0)
    col_mid = leg_px[order[n_ // 3 : 2 * n_ // 3]].mean(0)
    col_light = leg_px[order[-n_ // 6 :]].mean(0)

    anchors: Dict[str, object] = {
        "source": os.path.relpath(SRC, ROOT),
        "source_size": [W, H],
        "alpha_convention": "premultiplied-over-black: out = rgb + (1 - alpha) * background",
        "crop_origin": [int(x0), int(y0)],
        "sprite_size": [int(x1 - x0), int(y1 - y0)],
        "axis_x": round(AXIS_X - x0, 1),
        "head_center": sp(HEAD_CENTER),
        "eye_left": sp(eye_l),
        "eye_right": sp(eye_r),
        "body_center": sp(BODY_CENTER),
        "body_bbox": [int(body_bbox[0] - x0), int(body_bbox[1] - y0), int(body_bbox[2] - x0), int(body_bbox[3] - y0)],
        "body_length_px": int(body_len),
        "front_leg_attach": {"left": sp(FRONT_ATTACH_L), "right": sp(mirror(FRONT_ATTACH_L))},
        "front_leg_rest": {
            "left": {"knee": sp(FRONT_KNEE_L), "ankle": sp(FRONT_ANKLE_L), "tip": sp(FRONT_TIP_L)},
            "right": {"knee": sp(mirror(FRONT_KNEE_L)), "ankle": sp(mirror(FRONT_ANKLE_L)), "tip": sp(mirror(FRONT_TIP_L))},
        },
        "front_leg_width_px": list(FRONT_LEG_WIDTH),
        "leg_color_bgr": {"dark": [int(v) for v in col_dark], "mid": [int(v) for v in col_mid], "light": [int(v) for v in col_light]},
        "removed_front_leg_polyline_left": [[int(x - x0), int(y - y0), int(r)] for x, y, r in FRONT_LEG_L],
        "background_level_bgr": [round(float(v), 1) for v in bg_level],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(anchors, f, indent=1)

    # ---- checks -------------------------------------------------------------------------
    h, w = a8.shape
    yy, xx = np.mgrid[0:h, 0:w]
    check = np.where(((xx // 40 + yy // 40) % 2 == 0)[..., None], np.array([150, 120, 110], np.uint8), np.array([120, 90, 80], np.uint8))
    a = a8.astype(np.float32)[..., None] / 255.0
    comp = np.clip(c8.astype(np.float32) + (1 - a) * check.astype(np.float32), 0, 255).astype(np.uint8)
    for key in ("head_center", "eye_left", "eye_right", "body_center"):
        p = anchors[key]
        cv2.circle(comp, (int(p[0]), int(p[1])), 6, (255, 0, 255), 1, cv2.LINE_AA)  # type: ignore[index]
    for sd in ("left", "right"):
        att = anchors["front_leg_attach"][sd]  # type: ignore[index]
        rest = anchors["front_leg_rest"][sd]  # type: ignore[index]
        pts = [att, rest["knee"], rest["ankle"], rest["tip"]]
        for p, q in zip(pts[:-1], pts[1:]):
            cv2.line(comp, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(comp, (int(att[0]), int(att[1])), 5, (0, 255, 255), 1, cv2.LINE_AA)
    bb = anchors["body_bbox"]
    cv2.rectangle(comp, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255, 255, 0), 1)  # type: ignore[index]
    cv2.imwrite(os.path.join(OUT_DIR, "fly_sprite_check.png"), comp)
    zoom = np.concatenate([comp[220:520, 0:260], comp[220:520, 435:695], comp[300:600, 0:260]], axis=1)
    cv2.imwrite(os.path.join(OUT_DIR, "fly_sprite_check_zoom.png"), cv2.resize(zoom, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC))

    dbg = np.clip(img.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
    dbg[band_bin > 0] = (0.5 * dbg[band_bin > 0] + (0, 0, 127)).astype(np.uint8)
    cnts, _ = cv2.findContours((alpha > 0.5).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dbg, cnts, -1, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(OUT_DIR, "fly_sprite_debug.png"), dbg[y0:y1, x0:x1])
    print(f"wrote {OUT_PNG} ({w}x{h}) and {OUT_JSON}")
    print(json.dumps({k: anchors[k] for k in ("crop_origin", "sprite_size", "eye_left", "eye_right", "body_center",
                                              "body_bbox", "body_length_px", "front_leg_attach", "leg_color_bgr")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

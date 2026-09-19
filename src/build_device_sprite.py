#!/usr/bin/env python3
"""Build the foldable-device ("iPhone Duo") body sprite with both screens cut out.

Input:  assets/duo_reference.jpg   photoreal render of the open foldable on a black background.
                                   Its screens show third-party app UIs which must NOT be
                                   reproduced: the whole screen interior is removed.
Output: assets/duo_sprite.png      RGBA, **premultiplied over black** (same convention as
                                   assets/fly_sprite.png):  out = rgb + (1 - alpha) * background.
                                   alpha 1 on the body (frame, hinge, bezels, camera dot, side
                                   buttons), 0 on the background and 0 inside both screens (a
                                   rounded rectangle; the black bezel around each screen stays
                                   opaque).  ~1.5 px feather.
        assets/duo_sprite.json     screen quads "L"/"R" (TL, TR, BR, BL), hinge line, bounding
                                   box, corner radius, physical panel size, ... (sprite pixels,
                                   see ``COORDINATE CONVENTION`` below)
        out/duo_sprite_check.png   sprite over a blue checker with the quads drawn (+ zoomed
                                   corners in out/duo_sprite_check_zoom.png)
        out/duo_warp_check.png     two grid test panels warped onto the screens (``warp_panels``)

Screen-edge measurement (all automatic; the constants below only pick the scan windows):
  * Scanning inward from the device silhouette, every outer edge has the same structure:
    bright titanium frame -> a uniform grey chamfer band (8-11 px, lum 20-45) -> the black
    bezel (lum < 10; 4 px at the sides, 7 px top/bottom) -> screen content.  The bezel-screen
    transition is sampled per row/column where the content is lit, the chamfer-bezel
    transition everywhere.  Along the hinge (no chamfer) the scan starts on the hinge centre
    line.  Straight segments are fitted with RANSAC + least squares; corners are the
    intersections of adjacent lines.
  * The bottom of both screens shows black UI over the black bezel, so the bottom edge is
    not directly observable: it is the fitted bottom chamfer line offset inward by the top
    bezel width (scaled by the bottom/top edge length ratio).  This is the one modelled
    (not measured) edge.
  * Corner radius: least-squares circular-arc fit of the edge samples around each corner
    where the content is lit.

COORDINATE CONVENTION: quad corners and lines are continuous *pixel-corner* coordinates:
the boundary between pixel columns 251 and 252 is x = 252.0.  To use them with OpenCV
(pixel centres at integers) subtract 0.5, as ``warp_panels`` does.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "assets", "duo_reference.jpg")
OUT_PNG = os.path.join(ROOT, "assets", "duo_sprite.png")
OUT_JSON = os.path.join(ROOT, "assets", "duo_sprite.json")
OUT_DIR = os.path.join(ROOT, "out")

PANEL_MM = (79.0, 111.0)            # physical size of each screen panel in our model (w, h)

BLACK = 10                          # lum below this is bezel / hinge gap / background
GREY_LO, GREY_HI = 10, 80           # the chamfer band lives in this range (uniform, +-8)
SIDE_ROWS = (320, 1060)             # rows used for the vertical edges (clear of the corners)
TOP_MARGIN = 45                     # columns this far from the side lines are skipped
MARGIN = 6                          # crop margin (px)

Line = Tuple[float, float, float]   # nx*x + ny*y + c = 0 with (nx, ny) a unit normal


# ----------------------------------------------------------------------------- geometry ---
def line_from_points(p: Sequence[float], q: Sequence[float]) -> Line:
    dx, dy = q[0] - p[0], q[1] - p[1]
    n = float(np.hypot(dx, dy)) + 1e-12
    nx, ny = -dy / n, dx / n
    return (nx, ny, -(nx * p[0] + ny * p[1]))


def line_dist(line: Line, pts: np.ndarray) -> np.ndarray:
    return pts[:, 0] * line[0] + pts[:, 1] * line[1] + line[2]


def intersect(a: Line, b: Line) -> Tuple[float, float]:
    d = a[0] * b[1] - a[1] * b[0]
    x = (a[1] * b[2] - a[2] * b[1]) / d
    y = (a[2] * b[0] - a[0] * b[2]) / d
    return (float(x), float(y))


def offset_line(line: Line, d: float) -> Line:
    """Shift a line by ``d`` along its normal."""
    return (line[0], line[1], line[2] - d)


def fit_line_ransac(pts: np.ndarray, thr: float = 1.0, iters: int = 400, seed: int = 0) -> Tuple[Line, np.ndarray]:
    """RANSAC line through Nx2 points (inlier |dist| <= thr), refined by least squares.
    Returns the line and the inlier mask."""
    rng = np.random.default_rng(seed)
    n = len(pts)
    if n < 2:
        raise ValueError("need >= 2 points")
    best_line: Optional[Line] = None
    best_in: np.ndarray = np.zeros(n, bool)
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        if np.allclose(pts[i], pts[j]):
            continue
        ln = line_from_points(pts[i], pts[j])
        inl = np.abs(line_dist(ln, pts)) <= thr
        if inl.sum() > best_in.sum():
            best_in, best_line = inl, ln
    assert best_line is not None
    for _ in range(2):                       # LSQ refinement on the inliers, re-select
        sel = pts[best_in].astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        refined = line_from_points((float(x0), float(y0)), (float(x0 + vx), float(y0 + vy)))
        best_in = np.abs(line_dist(refined, pts)) <= thr
        best_line = refined
    return best_line, best_in


def runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """[start, end) of the True runs of a 1-D bool array."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


# ----------------------------------------------------------------------------- scanning ---
def scan_outer(p: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    """``p``: luminance from the device's outer silhouette (index 0) inward.

    Returns (chamfer_end, screen_start): index of the first bezel pixel after the uniform
    grey chamfer band and index of the first lit content pixel after the bezel (None when the
    content there is black, i.e. the bezel-screen boundary is not observable)."""
    grey = (p >= GREY_LO) & (p <= GREY_HI)
    for s, e in runs(grey):
        if e - s < 4 or s < 6:
            continue
        seg = p[s:e].astype(int)
        if np.count_nonzero(np.abs(seg - np.median(seg)) <= 8) < 4:
            continue                                     # high-contrast frame band, not the chamfer
        j = e
        while j < len(p) and p[j] < BLACK:
            j += 1
        blk = j - e
        if blk < 3:
            continue
        if blk <= 12 and j + 3 <= len(p) and bool((p[j:j + 3] >= BLACK).all()):
            return e, j
        return e, None
    return None, None


def scan_hinge(p: np.ndarray) -> Optional[int]:
    """``p``: luminance from the hinge centre line (index 0) outward.  Skips the (optional)
    hinge highlight, then needs a black bezel run followed by lit content."""
    i = 0
    while i < 5 and i < len(p) and p[i] >= BLACK:
        i += 1
    j = i
    while j < len(p) and p[j] < BLACK:
        j += 1
    blk = j - i
    if 2 <= blk <= 12 and j + 3 <= len(p) and bool((p[j:j + 3] >= BLACK).all()):
        return j
    return None


def silhouette(lum: np.ndarray) -> np.ndarray:
    """Device mask: everything enclosed by the bright body (screens and hinge gap included)."""
    m = (lum > 24).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    big = 1 + int(np.argmax(st[1:, 4]))
    m = (lab == big).astype(np.uint8)
    inv = (m == 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = m.shape
    for i in range(1, n):
        x, y, ww, hh, _ = st[i]
        if not (x == 0 or y == 0 or x + ww >= w or y + hh >= h):
            m[lab == i] = 1                              # enclosed hole -> body
    return m


# ----------------------------------------------------------------------------- measuring ---
def measure_screens(lum: np.ndarray, sil: np.ndarray) -> Dict[str, Any]:
    H, W = lum.shape
    rows = np.arange(SIDE_ROWS[0], SIDE_ROWS[1] + 1)

    # hinge centre: the black gap between the screens (column means over the middle rows)
    band = lum[400:1000, 650:760].mean(0)
    gap = runs(band < 6)
    gs, ge = max(gap, key=lambda r: r[1] - r[0])
    hinge_c = 650 + (gs + ge) // 2                     # integer column on the hinge line
    hinge_x = 650 + 0.5 * (gs + ge)                    # continuous (pixel-corner coords)

    def outer_side(side: str) -> Tuple[np.ndarray, np.ndarray]:
        scr, cham = [], []
        for y in rows:
            xs = np.where(sil[y] > 0)[0]
            if side == "L":
                x0 = int(xs.min())
                p = lum[y, x0:x0 + 70]
                e, j = scan_outer(p)
                if e is not None:
                    cham.append((x0 + e, y + 0.5))
                if j is not None:
                    scr.append((x0 + j, y + 0.5))
            else:
                x0 = int(xs.max())
                p = lum[y, x0:x0 - 70:-1]
                e, j = scan_outer(p)
                if e is not None:
                    cham.append((x0 - e + 1, y + 0.5))
                if j is not None:
                    scr.append((x0 - j + 1, y + 0.5))
        return np.array(scr, np.float64), np.array(cham, np.float64)

    def hinge_side(side: str) -> np.ndarray:
        pts = []
        for y in rows:
            if side == "L":                             # left screen's right edge
                j = scan_hinge(lum[y, hinge_c::-1])
                if j is not None:
                    pts.append((hinge_c - j + 1, y + 0.5))
            else:
                j = scan_hinge(lum[y, hinge_c:])
                if j is not None:
                    pts.append((hinge_c + j, y + 0.5))
        return np.array(pts, np.float64)

    def horizontal(x_lo: int, x_hi: int, top: bool) -> Tuple[np.ndarray, np.ndarray]:
        scr, cham = [], []
        for x in range(x_lo, x_hi + 1):
            ys = np.where(sil[:, x] > 0)[0]
            if top:
                y0 = int(ys.min())
                e, j = scan_outer(lum[y0:y0 + 70, x])
                if e is not None:
                    cham.append((x + 0.5, y0 + e))
                if j is not None:
                    scr.append((x + 0.5, y0 + j))
            else:
                y0 = int(ys.max())
                e, j = scan_outer(lum[y0:y0 - 70:-1, x])
                if e is not None:
                    cham.append((x + 0.5, y0 - e + 1))
                if j is not None:
                    scr.append((x + 0.5, y0 - j + 1))
        return np.array(scr, np.float64), np.array(cham, np.float64)

    out: Dict[str, Any] = {"hinge_x": hinge_x, "screens": {}}
    for side in ("L", "R"):
        outer_pts, outer_cham = outer_side(side)
        hinge_pts = hinge_side(side)
        l_outer, in_o = fit_line_ransac(outer_pts)
        l_hinge, in_h = fit_line_ransac(hinge_pts)
        l_ocham, _ = fit_line_ransac(outer_cham)
        # column window for the horizontal edges: between the two vertical lines at y = 400
        y_mid = 400.0
        xo = -(l_outer[1] * y_mid + l_outer[2]) / l_outer[0]
        xh = -(l_hinge[1] * y_mid + l_hinge[2]) / l_hinge[0]
        x_lo, x_hi = sorted((xo, xh))
        x_lo_i, x_hi_i = int(round(x_lo)) + TOP_MARGIN, int(round(x_hi)) - TOP_MARGIN
        top_pts, top_cham = horizontal(x_lo_i, x_hi_i, top=True)
        bot_pts, bot_cham = horizontal(x_lo_i, x_hi_i, top=False)
        l_top, in_t = fit_line_ransac(top_pts)
        l_tcham, _ = fit_line_ransac(top_cham)
        l_bcham, _ = fit_line_ransac(bot_cham)

        # bezel widths: perpendicular distance chamfer line -> screen line at the edge middle
        def bezel(l_ch: Line, pts: np.ndarray) -> float:
            mid = pts.mean(0)[None, :]                   # a point on the screen line
            return float(abs(line_dist(l_ch, mid)[0]))

        bez_top = bezel(l_tcham, top_pts)
        bez_outer = bezel(l_ocham, outer_pts)
        # bottom chamfer corners -> edge length ratio bottom/top (perspective scale)
        ct_a, ct_b = intersect(l_tcham, l_outer), intersect(l_tcham, l_hinge)
        cb_a, cb_b = intersect(l_bcham, l_outer), intersect(l_bcham, l_hinge)
        scale = float(np.hypot(cb_a[0] - cb_b[0], cb_a[1] - cb_b[1]) / np.hypot(ct_a[0] - ct_b[0], ct_a[1] - ct_b[1]))
        bez_bot = bez_top * scale
        # shift the bottom chamfer line inward (towards the screen centre) by bez_bot
        centre = np.array([[0.5 * (x_lo + x_hi), 0.5 * (ct_a[1] + cb_a[1])]])
        sgn = 1.0 if float(line_dist(l_bcham, centre)[0]) > 0 else -1.0
        l_bot = offset_line(l_bcham, sgn * bez_bot)
        # bottom-edge samples where the content happened to be lit (report only)
        bot_direct = bot_pts

        tl = intersect(l_top, l_outer if side == "L" else l_hinge)
        tr = intersect(l_top, l_hinge if side == "L" else l_outer)
        br = intersect(l_bot, l_hinge if side == "L" else l_outer)
        bl = intersect(l_bot, l_outer if side == "L" else l_hinge)
        quad = np.array([tl, tr, br, bl], np.float64)

        # corner radius: arc fit around the two top corners (lit content) using the top and
        # side samples within 90 px of the corner
        radii = []
        for corner, ln_a, pa, ln_b, pb in ((tl, l_top, top_pts, l_outer if side == "L" else l_hinge,
                                            outer_pts if side == "L" else hinge_pts),
                                           (tr, l_top, top_pts, l_hinge if side == "L" else l_outer,
                                            hinge_pts if side == "L" else outer_pts)):
            r = fit_corner_radius(np.array(corner), (ln_a, pa), (ln_b, pb))
            if r is not None:
                radii.append(r)
        resid = {
            "outer_rms": float(np.sqrt(np.mean(line_dist(l_outer, outer_pts[in_o]) ** 2))),
            "hinge_rms": float(np.sqrt(np.mean(line_dist(l_hinge, hinge_pts[in_h]) ** 2))),
            "top_rms": float(np.sqrt(np.mean(line_dist(l_top, top_pts[in_t]) ** 2))),
            "n_outer": int(in_o.sum()), "n_hinge": int(in_h.sum()), "n_top": int(in_t.sum()),
            "n_bottom_direct": int(len(bot_direct)),
        }
        if len(bot_direct) >= 5:
            resid["bottom_direct_vs_model_px"] = float(np.median(line_dist(l_bot, bot_direct)))
        out["screens"][side] = {
            "quad": quad, "lines": {"top": l_top, "bottom": l_bot, "outer": l_outer, "hinge": l_hinge,
                                    "top_chamfer": l_tcham, "bottom_chamfer": l_bcham, "outer_chamfer": l_ocham},
            "bezel_px": {"top": bez_top, "outer": bez_outer, "bottom_model": bez_bot, "scale_bottom_over_top": scale},
            "corner_radius_px": float(np.median(radii)) if radii else float("nan"),
            "corner_radii_px": radii, "fit": resid,
            "samples": {"outer": outer_pts, "hinge": hinge_pts, "top": top_pts, "bottom_direct": bot_direct},
        }
    return out


def fit_corner_radius(corner: np.ndarray, a: Tuple[Line, np.ndarray], b: Tuple[Line, np.ndarray],
                      reach: float = 90.0) -> Optional[float]:
    """Least-squares radius of a circular arc tangent to lines a and b, from the edge samples
    within ``reach`` px of the corner (their inward deviation from the straight line)."""
    items: List[Tuple[np.ndarray, np.ndarray]] = []     # (t along edge from corner, inward deviation)
    for (ln, pts), (other, _) in ((a, b), (b, a)):
        if len(pts) == 0:
            continue
        d = np.abs(line_dist(ln, pts))                   # deviation from the straight line
        t = np.abs(line_dist(other, pts))                # distance from the corner along the edge
        sel = t < reach
        if sel.sum() < 5:
            continue
        items.append((t[sel], d[sel]))
    if not items:
        return None
    t_all = np.concatenate([i[0] for i in items])
    d_all = np.concatenate([i[1] for i in items])
    best_r, best_err = None, float("inf")
    for r in np.arange(4.0, 80.0, 0.5):
        inset = np.where(t_all < r, r - np.sqrt(np.maximum(r * r - (r - t_all) ** 2, 0.0)), 0.0)
        err = float(np.mean((inset - d_all) ** 2))
        if err < best_err:
            best_err, best_r = err, float(r)
    return best_r


# ----------------------------------------------------------------------------- rendering ---
def rounded_rect_poly(w: float, h: float, r: float, n_arc: int = 12) -> np.ndarray:
    """Closed polygon (Nx2) of a w x h rounded rectangle with corner radius r, origin (0, 0)."""
    r = max(0.0, min(r, 0.5 * min(w, h)))
    pts: List[Tuple[float, float]] = []
    centres = [(r, r, np.pi, 1.5 * np.pi), (w - r, r, 1.5 * np.pi, 2 * np.pi),
               (w - r, h - r, 0.0, 0.5 * np.pi), (r, h - r, 0.5 * np.pi, np.pi)]
    for cx, cy, a0, a1 in centres:
        for a in np.linspace(a0, a1, n_arc + 1):
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(pts, np.float64)


def panel_homography(w: float, h: float, quad: np.ndarray) -> np.ndarray:
    """Homography from panel pixel-corner coords ([0,w]x[0,h]) to sprite pixel-corner coords."""
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    return cv2.getPerspectiveTransform(src, np.asarray(quad, np.float32))


def warp_points(Hm: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(-1, 1, 2), Hm).reshape(-1, 2)


def fill_poly_aa(shape: Tuple[int, int], pts_corner: np.ndarray) -> np.ndarray:
    """Anti-aliased float mask of a polygon given in pixel-corner coordinates."""
    m = np.zeros(shape, np.uint8)
    p = np.round((np.asarray(pts_corner, np.float64) - 0.5) * 16).astype(np.int32)   # centre coords, 1/16 px
    cv2.fillPoly(m, [p.reshape(-1, 1, 2)], 255, cv2.LINE_AA, shift=4)
    return m.astype(np.float32) / 255.0


def screen_hole_mask(shape: Tuple[int, int], quad: np.ndarray, radius_px: float) -> np.ndarray:
    """Rounded-rectangle screen mask (1 inside) warped onto ``quad`` (pixel-corner coords)."""
    q = np.asarray(quad, np.float64)
    w = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
    h = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
    Hm = panel_homography(w, h, q)
    poly = warp_points(Hm, rounded_rect_poly(w, h, radius_px))
    return fill_poly_aa(shape, poly)


def warp_panels(frame_bgr_L: np.ndarray, frame_bgr_R: np.ndarray, sprite_rgba: np.ndarray,
                meta: Dict[str, Any]) -> np.ndarray:
    """Perspective-warp two portrait panel images onto the screen quads and composite the
    premultiplied device sprite on top.  Returns a BGR uint8 image of the sprite's size."""
    h, w = sprite_rgba.shape[:2]
    canvas = np.zeros((h, w, 3), np.float32)
    radius = float(meta.get("screen_corner_radius_px", 0.0))
    for key, frame in (("L", frame_bgr_L), ("R", frame_bgr_R)):
        quad = np.asarray(meta["screens"][key]["quad"], np.float64)
        qw = 0.5 * (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3]))
        qh = 0.5 * (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1]))
        # INTER_AREA-quality: shrink to roughly the on-screen size first, then warp bilinearly
        tw, th = max(2, int(round(qw))), max(2, int(round(qh)))
        fh, fw = frame.shape[:2]
        if fw > tw * 1.15 or fh > th * 1.15:
            small = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        else:
            small = frame
        sh, sw = small.shape[:2]
        mask = fill_poly_aa((sh, sw), rounded_rect_poly(sw, sh, radius * sw / max(qw, 1e-6)))
        bgra = np.dstack([small.astype(np.float32) * mask[..., None], mask * 255.0])
        # pixel-corner -> pixel-centre coordinates for OpenCV
        Hm = panel_homography(sw, sh, quad)
        shift = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]], np.float64)
        Hc = shift @ Hm @ np.linalg.inv(shift)
        warped = cv2.warpPerspective(bgra, Hc, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        a = warped[..., 3:4] / 255.0
        canvas = warped[..., :3] + (1.0 - a) * canvas
    pm = sprite_rgba[..., :3].astype(np.float32)
    alpha = sprite_rgba[..., 3:4].astype(np.float32) / 255.0
    return np.clip(pm + (1.0 - alpha) * canvas, 0, 255).astype(np.uint8)


def test_panel(w: int, h: int, colour: Tuple[int, int, int], label: str) -> np.ndarray:
    """Solid colour panel with a white grid every 1/10 of the width, a border and a label."""
    p = np.zeros((h, w, 3), np.uint8)
    p[:] = colour
    step = w // 10
    for x in range(0, w, step):
        cv2.line(p, (x, 0), (x, h - 1), (255, 255, 255), max(2, w // 300))
    for y in range(0, h, step):
        cv2.line(p, (0, y), (w - 1, y), (255, 255, 255), max(2, w // 300))
    cv2.rectangle(p, (0, 0), (w - 1, h - 1), (255, 255, 255), max(4, w // 120))
    cv2.circle(p, (w // 2, h // 2), w // 4, (0, 0, 0), max(4, w // 120), cv2.LINE_AA)
    cv2.putText(p, label, (w // 2 - w // 8, h // 2 + w // 24), cv2.FONT_HERSHEY_SIMPLEX, w / 300.0, (0, 0, 0), max(4, w // 120), cv2.LINE_AA)
    return p


def draw_quad(img: np.ndarray, quad: Sequence[Sequence[float]], colour: Tuple[int, int, int]) -> None:
    q = [(int(round((p[0] - 0.5) * 16)), int(round((p[1] - 0.5) * 16))) for p in quad]
    for i in range(4):
        cv2.line(img, q[i], q[(i + 1) % 4], colour, 1, cv2.LINE_AA, shift=4)
    for i, p in enumerate(q):
        cv2.circle(img, p, 6 * 16, colour, 1, cv2.LINE_AA, shift=4)
        cv2.putText(img, str(i), (p[0] // 16 + 8, p[1] // 16 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)


# ----------------------------------------------------------------------------- main ---
def main() -> int:
    img = cv2.imread(SRC, cv2.IMREAD_COLOR)
    if img is None:
        print(f"missing {SRC}", file=sys.stderr)
        return 2
    H, W = img.shape[:2]
    os.makedirs(OUT_DIR, exist_ok=True)
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = np.concatenate([img[:60, :60].reshape(-1, 3), img[:60, -60:].reshape(-1, 3),
                              img[-60:, :60].reshape(-1, 3), img[-60:, -60:].reshape(-1, 3)])
    bg_level = corners.mean(0).astype(np.float32)

    sil = silhouette(lum)
    meas = measure_screens(lum, sil)

    # ---- alpha matte: body = silhouette; soft ring outside it keeps the thin edge highlights
    alpha = sil.astype(np.float32)
    ring = cv2.dilate(sil, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) & (1 - sil)
    lum_f = np.clip(lum.astype(np.float32) - float(bg_level.mean()), 0, 255)
    alpha[ring > 0] = np.clip(lum_f[ring > 0] / 40.0, 0.0, 1.0)

    radius = float(np.nanmedian([meas["screens"][s]["corner_radius_px"] for s in ("L", "R")]))
    holes = np.zeros((H, W), np.float32)
    for s in ("L", "R"):
        holes = np.maximum(holes, screen_hole_mask((H, W), meas["screens"][s]["quad"], radius))
    alpha *= 1.0 - holes

    pm = img.astype(np.float32) - (1.0 - alpha)[..., None] * bg_level[None, None, :]
    pm = np.clip(pm, 0, 255) * (1.0 - holes)[..., None]        # no screen colour survives
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0.7)                 # ~1.5 px feather
    pm = cv2.GaussianBlur(pm, (5, 5), 0.7)
    pm = np.minimum(pm, 255.0 * alpha[..., None] + 1e-3)
    # hard zero strictly inside the holes (the blur may leak a fraction of a level inward)
    inner = cv2.erode((holes > 0.999).astype(np.uint8), np.ones((5, 5), np.uint8))
    alpha[inner > 0] = 0.0
    pm[inner > 0] = 0.0

    # ---- crop
    a_ys, a_xs = np.where(alpha > 0.02)
    x0, x1 = max(0, int(a_xs.min()) - MARGIN), min(W, int(a_xs.max()) + MARGIN + 1)
    y0, y1 = max(0, int(a_ys.min()) - MARGIN), min(H, int(a_ys.max()) + MARGIN + 1)
    a8 = np.clip(alpha[y0:y1, x0:x1] * 255 + 0.5, 0, 255).astype(np.uint8)
    c8 = np.clip(pm[y0:y1, x0:x1] + 0.5, 0, 255).astype(np.uint8)
    rgba = np.dstack([c8, a8])
    cv2.imwrite(OUT_PNG, rgba)
    h, w = a8.shape

    def sp(p: Sequence[float]) -> List[float]:
        return [round(float(p[0]) - x0, 2), round(float(p[1]) - y0, 2)]

    sys_, sxs = np.where(sil > 0)
    dev_bbox = [int(sxs.min()) - x0, int(sys_.min()) - y0, int(sxs.max()) + 1 - x0, int(sys_.max()) + 1 - y0]
    hinge_x = float(meas["hinge_x"])
    hinge_top = min(meas["screens"]["L"]["quad"][1][1], meas["screens"]["R"]["quad"][0][1])
    hinge_bot = max(meas["screens"]["L"]["quad"][2][1], meas["screens"]["R"]["quad"][3][1])

    screens_json: Dict[str, Any] = {}
    for s in ("L", "R"):
        m = meas["screens"][s]
        q = m["quad"]
        qw = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
        qh = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
        screens_json[s] = {
            "quad": [sp(p) for p in q],
            "quad_order": ["top-left", "top-right", "bottom-right", "bottom-left"],
            "size_px_mean": [round(float(qw), 1), round(float(qh), 1)],
            "apparent_aspect": round(float(qw / qh), 4),
            "bezel_px": {k: round(float(v), 2) for k, v in m["bezel_px"].items()},
            "corner_radius_px": round(float(m["corner_radius_px"]), 1),
            "fit": {k: (round(float(v), 3) if isinstance(v, float) else v) for k, v in m["fit"].items()},
        }
    meta: Dict[str, Any] = {
        "source": os.path.relpath(SRC, ROOT),
        "source_size": [W, H],
        "alpha_convention": "premultiplied-over-black: out = rgb + (1 - alpha) * background",
        "coordinates": "continuous pixel-corner coordinates of the sprite (subtract 0.5 for OpenCV pixel centres)",
        "crop_origin": [int(x0), int(y0)],
        "sprite_size": [int(w), int(h)],
        "device_bbox": dev_bbox,
        "hinge_line": [sp((hinge_x, hinge_top)), sp((hinge_x, hinge_bot))],
        "panel_mm": list(PANEL_MM),
        "panel_aspect": round(PANEL_MM[0] / PANEL_MM[1], 4),
        "screen_corner_radius_px": round(radius, 1),
        "screens": screens_json,
        "bottom_edge_note": "bottom edges are the bottom chamfer line offset by the top bezel width (content is black there)",
        "background_level_bgr": [round(float(v), 1) for v in bg_level],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(meta, f, indent=1)

    # ---- checks: sprite over a blue checker with the quads
    yy, xx = np.mgrid[0:h, 0:w]
    check = np.where(((xx // 40 + yy // 40) % 2 == 0)[..., None], np.array([170, 110, 60], np.uint8), np.array([140, 85, 40], np.uint8))
    a = a8.astype(np.float32)[..., None] / 255.0
    comp = np.clip(c8.astype(np.float32) + (1 - a) * check.astype(np.float32), 0, 255).astype(np.uint8)
    draw_quad(comp, meta["screens"]["L"]["quad"], (255, 255, 0))
    draw_quad(comp, meta["screens"]["R"]["quad"], (0, 140, 255))
    hl = meta["hinge_line"]
    cv2.line(comp, (int(round(hl[0][0])), int(round(hl[0][1]))), (int(round(hl[1][0])), int(round(hl[1][1]))), (255, 0, 255), 1, cv2.LINE_AA)
    bb = meta["device_bbox"]
    cv2.rectangle(comp, (bb[0], bb[1]), (bb[2] - 1, bb[3] - 1), (0, 255, 0), 1)
    cv2.imwrite(os.path.join(OUT_DIR, "duo_sprite_check.png"), comp)
    # zoomed edge midpoints and corners (4x) on the boosted source with the quads drawn, for a
    # pixel-accurate check: row 1 = L (TL, top-mid, TR, hinge-mid, BR, bottom-mid, BL, outer-mid), row 2 = R
    src_zoom = np.clip(img.astype(np.float32) * 3, 0, 255).astype(np.uint8)
    tiles = []
    for s, col in (("L", (255, 255, 0)), ("R", (0, 140, 255))):
        q = meas["screens"][s]["quad"]
        spots = [q[0], 0.5 * (q[0] + q[1]), q[1], 0.5 * (q[1] + q[2]), q[2], 0.5 * (q[2] + q[3]), q[3], 0.5 * (q[3] + q[0])]
        for p in spots:
            cx, cy = int(round(p[0])), int(round(p[1]))
            ox, oy = max(0, cx - 30), max(0, cy - 30)
            tile = cv2.resize(src_zoom[oy:cy + 30, ox:cx + 30].copy(), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            qz = [((v[0] - ox) * 4, (v[1] - oy) * 4) for v in q]
            for i in range(4):
                cv2.line(tile, (int(round((qz[i][0] - 0.5) * 16)), int(round((qz[i][1] - 0.5) * 16))),
                         (int(round((qz[(i + 1) % 4][0] - 0.5) * 16)), int(round((qz[(i + 1) % 4][1] - 0.5) * 16))), col, 1, cv2.LINE_AA, shift=4)
            tiles.append(tile)
    cv2.imwrite(os.path.join(OUT_DIR, "duo_sprite_check_zoom.png"), np.concatenate([np.concatenate(tiles[:8], 1), np.concatenate(tiles[8:], 1)], 0))

    # every pixel strictly inside the (rounded) screens must be fully transparent and black
    worst_a, worst_c = 0, 0
    for s in ("L", "R"):
        q_sprite = np.array(meta["screens"][s]["quad"], np.float64)
        inner_m = screen_hole_mask((h, w), q_sprite, radius) > 0.999
        inner_m = cv2.erode(inner_m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        worst_a = max(worst_a, int(a8[inner_m].max()))
        worst_c = max(worst_c, int(c8[inner_m].max()))
    print(f"screen interior (rounded screen eroded 2 px): max alpha {worst_a}, max colour {worst_c}  (both must be 0)")
    if worst_a or worst_c:
        print("ERROR: screen content survives", file=sys.stderr)
        return 1

    # ---- warp test
    frame_l = test_panel(1335, 1878, (140, 120, 20), "L")
    frame_r = test_panel(1335, 1878, (30, 90, 220), "R")
    warped = warp_panels(frame_l, frame_r, rgba, meta)
    cv2.imwrite(os.path.join(OUT_DIR, "duo_warp_check.png"), warped)
    zoom_tiles = []
    for s in ("L", "R"):
        for p in meta["screens"][s]["quad"]:
            cx, cy = int(round(p[0])), int(round(p[1]))
            t = warped[max(0, cy - 30):cy + 30, max(0, cx - 30):cx + 30]
            zoom_tiles.append(cv2.resize(t, (240, 240), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(os.path.join(OUT_DIR, "duo_warp_check_zoom.png"), np.concatenate([np.concatenate(zoom_tiles[:4], 1), np.concatenate(zoom_tiles[4:], 1)], 0))

    print(f"wrote {OUT_PNG} ({w}x{h}) and {OUT_JSON}")
    print(json.dumps({k: meta[k] for k in ("crop_origin", "sprite_size", "device_bbox", "hinge_line", "screen_corner_radius_px")}))
    for s in ("L", "R"):
        print(s, json.dumps(meta["screens"][s]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

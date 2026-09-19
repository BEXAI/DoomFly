#!/usr/bin/env python3
"""Arena world for the free-roaming fly (Phase 2, docs/ARENA_DESIGN.md).

Three pieces, all 2-D top-down in millimetres, origin at the room centre, x right, y up,
heading th in radians (0 = +x, counter-clockwise positive; the fly's LEFT is th + 90 deg):

  Phone         geometry of the open Duo lying flat at the centre (two portrait panels
                79 x 111 mm side by side, crease at x = 0) and the "which panel is under
                this point" query.
  Body          correlated-random-walk baseline (identical in every condition) modulated by
                the brain: speed from the descending-neuron population rate, turning from the
                right-minus-left DNa imbalance; walls reflect; front-leg tips for contact.
  PanoramicEye  a 2 mm luminance map of the room with the two feed panels pasted in, sampled
                from the fly's pose into two (24, 18) = (distance, azimuth) grids that are fed
                to the existing ``EyeEncoder`` (see class docstring for the reinterpretation).

Scale: the real fly is ~3 mm long; the arena fly is 10x (body 30 mm, front-leg reach 22 mm)
so that "a leg over a panel" is a meaningful event and the top-down video is legible.

Self-test: ``python3 src/arena.py`` walks 60 s with zero brain input, plots the trajectory to
``out/arena_selftest.png`` and prints eye-grid statistics on vs off the phone.
"""
from __future__ import annotations

import math
import os
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- world constants (mm)
ROOM_MM: float = 400.0                 # square room, centred at the origin
HALF_MM: float = ROOM_MM / 2.0
WALL_BAND_MM: float = 4.0              # luminance band of the walls in the eye map
PHONE_W_MM: float = 158.0              # two panels side by side
PHONE_H_MM: float = 111.0
PANEL_W_MM: float = PHONE_W_MM / 2.0   # 79
PANEL_H_MM: float = PHONE_H_MM         # 111
FLY_SCALE: float = 10.0                # real fly ~3 mm; arena fly 30 mm (documented in README)
BODY_LEN_MM: float = 30.0
REACH_MM: float = 22.0                 # front-leg tip distance from body centre
LEG_ANGLE_RAD: float = math.radians(35.0)   # front legs at +-35 deg from heading

# eye map luminances
LUM_FLOOR: float = 0.03
LUM_WALL: float = 0.06
LUM_BEZEL: float = 0.15
PANEL_GRID: Tuple[int, int] = (55, 39)   # feed.luminance_grid(rows, cols) per panel at 2 mm

Point = Tuple[float, float]
Pose = Union[Sequence[float], "Body"]


def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


# ====================================================================== Phone
class Phone:
    """Open Duo lying flat at the room centre: left panel x in [-79, 0], right panel x in
    [0, 79], both y in [-55.5, 55.5]. ``rects`` are (x0, y0, x1, y1) in mm."""

    def __init__(self, cx: float = 0.0, cy: float = 0.0) -> None:
        self.cx, self.cy = float(cx), float(cy)
        hw, hh = PHONE_W_MM / 2.0, PHONE_H_MM / 2.0
        self.rect: Tuple[float, float, float, float] = (cx - hw, cy - hh, cx + hw, cy + hh)
        self.rects: Dict[str, Tuple[float, float, float, float]] = {
            "L": (cx - hw, cy - hh, cx, cy + hh),
            "R": (cx, cy - hh, cx + hw, cy + hh),
        }

    @staticmethod
    def _inside(p: Point, r: Tuple[float, float, float, float]) -> bool:
        return r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3]

    def contains(self, p: Point) -> bool:
        """Point within the whole phone rectangle (used for 'on_phone')."""
        return self._inside(p, self.rect)

    def panel_under(self, p: Point) -> Optional[str]:
        """'L' | 'R' | None. The crease x = cx belongs to the right panel."""
        if not self.contains(p):
            return None
        return "L" if p[0] < self.cx else "R"

    def distance(self, p: Point) -> float:
        """Euclidean distance from ``p`` to the phone centre (mm)."""
        return math.hypot(p[0] - self.cx, p[1] - self.cy)

    def to_json(self) -> Dict[str, object]:
        return dict(centre=[self.cx, self.cy], rect=list(self.rect),
                    panels={k: list(v) for k, v in self.rects.items()},
                    panel_mm=[PANEL_W_MM, PANEL_H_MM])


# ====================================================================== Body
class Body:
    """Walking fly: correlated random walk baseline + brain modulation.

    Baseline (same in every condition): speed ``v0`` = 12 mm/s, heading noise from an
    Ornstein-Uhlenbeck process (stationary sigma 1.2 rad/s, tau 0.5 s).

    Brain modulation, per ``step(dt, r_dn, r_dna_L, r_dna_R)`` where every rate is an 80 ms
    trace in Hz per cell:

        v     = clip(v0_eff + g_v * r_dn, 0, v_max)                    g_v = 6 mm/s per Hz
        steer = g_omega * (r_dna_R - r_dna_L) / (r_dna_R + r_dna_L + eps)   g_omega = 3 rad/s
        dth   = (omega_noise - steer) * dt

    Sign convention: ``steer`` is a *right-turn* rate (positive when the right DNa set is more
    active, DNa02 ipsilateral-turn convention, docs/RESEARCH_REFS.md section 7); since th is
    counter-clockwise positive, a right turn is a negative dth, hence the minus sign.

    "Standing on the screen": when ``r_dn < stop_hz`` (0.05 Hz) and the body centre is on the
    phone, v0_eff = ``v_stand`` (3 mm/s) instead of v0.

    Walls: the body centre is kept ``wall_margin`` (half the body length, 15 mm) inside the
    wall's inner face; on contact the heading is reflected off that wall and given a random
    kick of +-30 deg, and the OU noise is reset to zero.

    Condition "random": ``g_v = g_omega = 0`` (pure baseline walker).
    """

    def __init__(self, phone: Phone, rng: np.random.Generator,
                 x: float = 0.0, y: float = 0.0, th: float = 0.0,
                 v0: float = 12.0, g_v: float = 6.0, g_omega: float = 3.0,
                 sigma_omega: float = 1.2, tau_omega: float = 0.5,
                 v_max: float = 60.0, v_stand: float = 3.0, stop_hz: float = 0.05,
                 eps_hz: float = 0.1, wall_kick_deg: float = 30.0,
                 wall_margin: float = BODY_LEN_MM / 2.0) -> None:
        self.phone = phone
        self.rng = rng
        self.x, self.y, self.th = float(x), float(y), wrap_angle(float(th))
        self.v = 0.0
        self.omega_noise = 0.0
        self.omega = 0.0            # last total dth/dt (rad/s, CCW positive)
        self.steer = 0.0            # last brain right-turn rate (rad/s)
        self.v0, self.g_v, self.g_omega = float(v0), float(g_v), float(g_omega)
        self.sigma_omega, self.tau_omega = float(sigma_omega), float(tau_omega)
        self.v_max, self.v_stand, self.stop_hz = float(v_max), float(v_stand), float(stop_hz)
        self.eps_hz = float(eps_hz)
        self.wall_kick = math.radians(float(wall_kick_deg))
        self.wall_margin = float(wall_margin)
        self.limit = HALF_MM - WALL_BAND_MM - self.wall_margin   # |x|,|y| <= limit
        self.wall_hits = 0
        self.standing = False

    # ------------------------------------------------------------ queries
    @property
    def pos(self) -> Point:
        return (self.x, self.y)

    @property
    def on_phone(self) -> bool:
        return self.phone.contains(self.pos)

    @property
    def dist_mm(self) -> float:
        """Distance from the body centre to the phone centre."""
        return self.phone.distance(self.pos)

    def front_leg_tips(self) -> Tuple[Point, Point]:
        """(left tip, right tip): ``REACH_MM`` from the centre at +-35 deg from heading."""
        aL, aR = self.th + LEG_ANGLE_RAD, self.th - LEG_ANGLE_RAD
        return ((self.x + REACH_MM * math.cos(aL), self.y + REACH_MM * math.sin(aL)),
                (self.x + REACH_MM * math.cos(aR), self.y + REACH_MM * math.sin(aR)))

    def panel_under(self, p: Point) -> Optional[str]:
        return self.phone.panel_under(p)

    def reach(self) -> Dict[str, Optional[str]]:
        """Panel under each front-leg tip: {'L': 'L'|'R'|None, 'R': ...}."""
        tl, tr = self.front_leg_tips()
        return {"L": self.panel_under(tl), "R": self.panel_under(tr)}

    def pose(self) -> Dict[str, float]:
        return dict(x=self.x, y=self.y, th=self.th, v=self.v)

    # ------------------------------------------------------------ dynamics
    def step(self, dt: float, r_dn: float, r_dna_L: float, r_dna_R: float) -> Dict[str, float]:
        """Advance the pose by ``dt`` seconds given the three 80 ms rate traces (Hz per cell)."""
        dt = float(dt)
        # OU heading noise (stationary std sigma_omega)
        tau = self.tau_omega
        self.omega_noise += (-self.omega_noise / tau) * dt + self.sigma_omega * math.sqrt(2.0 * dt / tau) * float(self.rng.standard_normal())
        # brain steering (right-turn positive)
        rl, rr = max(0.0, float(r_dna_L)), max(0.0, float(r_dna_R))
        self.steer = self.g_omega * (rr - rl) / (rr + rl + self.eps_hz) if self.g_omega else 0.0
        self.omega = self.omega_noise - self.steer
        # speed
        self.standing = bool(r_dn < self.stop_hz and self.on_phone)
        v0_eff = self.v_stand if self.standing else self.v0
        self.v = float(np.clip(v0_eff + self.g_v * max(0.0, float(r_dn)), 0.0, self.v_max))
        # integrate (heading first, then translate along the new heading)
        self.th = wrap_angle(self.th + self.omega * dt)
        self.x += self.v * math.cos(self.th) * dt
        self.y += self.v * math.sin(self.th) * dt
        self._walls()
        return self.pose()

    def _walls(self) -> None:
        lim = self.limit
        hit = False
        c, s = math.cos(self.th), math.sin(self.th)
        if self.x > lim:
            self.x = lim
            if c > 0:
                self.th = math.pi - self.th
            hit = True
        elif self.x < -lim:
            self.x = -lim
            if c < 0:
                self.th = math.pi - self.th
            hit = True
        if self.y > lim:
            self.y = lim
            if s > 0:
                self.th = -self.th
            hit = True
        elif self.y < -lim:
            self.y = -lim
            if s < 0:
                self.th = -self.th
            hit = True
        if hit:
            self.th = wrap_angle(self.th + float(self.rng.uniform(-self.wall_kick, self.wall_kick)))
            # make sure the kick did not point the fly back into the wall it just hit
            if self.x >= lim and math.cos(self.th) > 0:
                self.th = wrap_angle(math.pi - self.th)
            elif self.x <= -lim and math.cos(self.th) < 0:
                self.th = wrap_angle(math.pi - self.th)
            if self.y >= lim and math.sin(self.th) > 0:
                self.th = wrap_angle(-self.th)
            elif self.y <= -lim and math.sin(self.th) < 0:
                self.th = wrap_angle(-self.th)
            self.omega_noise = 0.0
            self.wall_hits += 1

    def params(self) -> Dict[str, float]:
        return dict(v0=self.v0, g_v=self.g_v, g_omega=self.g_omega, sigma_omega=self.sigma_omega,
                    tau_omega=self.tau_omega, v_max=self.v_max, v_stand=self.v_stand, stop_hz=self.stop_hz,
                    eps_hz=self.eps_hz, wall_kick_deg=math.degrees(self.wall_kick), wall_margin=self.wall_margin,
                    body_len_mm=BODY_LEN_MM, reach_mm=REACH_MM, leg_angle_deg=math.degrees(LEG_ANGLE_RAD),
                    scale=FLY_SCALE)


def random_start_pose(rng: np.random.Generator, phone: Phone, min_dist: float = 120.0,
                      limit: float = HALF_MM - WALL_BAND_MM - BODY_LEN_MM / 2.0) -> Tuple[float, float, float]:
    """Random (x, y, th) inside the walkable area with ``dist to phone centre >= min_dist``."""
    for _ in range(10000):
        x, y = rng.uniform(-limit, limit, size=2)
        if phone.distance((float(x), float(y))) >= min_dist:
            return float(x), float(y), float(rng.uniform(-math.pi, math.pi))
    raise RuntimeError("could not place the fly")


# ====================================================================== PanoramicEye
class PanoramicEye:
    """Luminance map of the room + ray-sampled (distance x azimuth) grids for both eyes.

    Map: ``cell_mm`` = 2 mm cells (200 x 200), indexed ``map[iy, ix]`` with ix along +x and iy
    along +y. Floor 0.03, a 4 mm wall band 0.06, phone bezel 0.15, and each panel = that
    feed's ``luminance_grid(55, 39)`` pasted on its rectangle (feed row 0 = the top of the
    screen = the +y edge; feed col 0 = the -x edge; both panels upright). Call
    ``update_panels(grid_L, grid_R)`` every control step.

    Grids: ``grids(pose) -> (grid_L, grid_R)``, float32 (24, 18) in [0, 1]. This shape is what
    the existing ``EyeEncoder`` expects for a 24 x 18 *screen* grid; here we REINTERPRET its
    two axes: rows = distance from the fly (row 0 = nearest = lowest in the visual field,
    24 log-spaced bins from 5 to 300 mm) and columns = azimuth (18 bins of 10 deg, col 0 =
    frontal 0-10 deg, col 17 = rear 170-180 deg) on that eye's side. The left eye covers the
    left hemifield (azimuth measured CCW from heading), the right eye the right hemifield
    (CW), so col 0 is frontal for both. Each cell is the mean of ``samples_per_bin`` (6)
    nearest-cell lookups along the ray at the column's central azimuth; samples outside the
    room read 0. Fully vectorised: one numpy pass for both eyes.
    """

    def __init__(self, phone: Optional[Phone] = None, cell_mm: float = 2.0, room_mm: float = ROOM_MM,
                 n_dist: int = 24, n_az: int = 18, d_min: float = 5.0, d_max: float = 300.0,
                 samples_per_bin: int = 6) -> None:
        self.phone = phone or Phone()
        self.cell = float(cell_mm)
        self.n_cells = int(round(room_mm / cell_mm))
        self.half = room_mm / 2.0
        self.n_dist, self.n_az, self.samples = int(n_dist), int(n_az), int(samples_per_bin)
        self.d_min, self.d_max = float(d_min), float(d_max)
        self.map = np.full((self.n_cells, self.n_cells), LUM_FLOOR, np.float32)
        band = max(1, int(round(WALL_BAND_MM / self.cell)))
        self.map[:band, :] = LUM_WALL
        self.map[-band:, :] = LUM_WALL
        self.map[:, :band] = LUM_WALL
        self.map[:, -band:] = LUM_WALL
        # phone bezel then panel windows
        r = self.phone.rect
        self._bezel = (self._ix(r[0]), self._ix(r[2]), self._iy(r[1]), self._iy(r[3]))
        bx0, bx1, by0, by1 = self._bezel
        self.map[by0:by1, bx0:bx1] = LUM_BEZEL
        rows, cols = PANEL_GRID
        self.panel_grid = PANEL_GRID
        iy0 = int(round((self.phone.cy - PANEL_H_MM / 2.0 + self.half) / self.cell))
        ixL = int(round((self.phone.cx - PANEL_W_MM + self.half) / self.cell)) + 1   # 1-cell bezel margin
        ixR = int(round((self.phone.cx + self.half) / self.cell))
        self._panel_slices: Dict[str, Tuple[slice, slice]] = {
            "L": (slice(iy0, iy0 + rows), slice(ixL, ixL + cols)),
            "R": (slice(iy0, iy0 + rows), slice(ixR, ixR + cols)),
        }
        # ray sample offsets in the body frame: distance d and azimuth a, flattened as
        # [eye, row, col, sample]; left eye a > 0 (CCW), right eye a < 0 (CW)
        edges = np.geomspace(self.d_min, self.d_max, self.n_dist + 1)
        frac = (np.arange(self.samples) + 0.5) / self.samples
        d = edges[:-1, None] + frac[None, :] * (edges[1:] - edges[:-1])[:, None]      # (n_dist, S)
        az = (np.arange(self.n_az) + 0.5) * (math.pi / self.n_az)                       # bin centres
        d_full = np.broadcast_to(d[:, None, :], (self.n_dist, self.n_az, self.samples))
        a_full = np.broadcast_to(az[None, :, None], (self.n_dist, self.n_az, self.samples))
        self._d = np.concatenate([d_full.ravel(), d_full.ravel()]).astype(np.float64)
        self._a = np.concatenate([a_full.ravel(), -a_full.ravel()]).astype(np.float64)
        self.dist_edges_mm = edges
        self.az_edges_deg = np.arange(self.n_az + 1) * (180.0 / self.n_az)

    # ------------------------------------------------------------ map helpers
    def _ix(self, x: float) -> int:
        return int(np.clip(int(round((x + self.half) / self.cell)), 0, self.n_cells))

    def _iy(self, y: float) -> int:
        return self._ix(y)

    def update_panels(self, grid_L: np.ndarray, grid_R: np.ndarray) -> None:
        """Paste the two feed luminance grids (55, 39) onto their panels (row 0 -> +y edge)."""
        for key, g in (("L", grid_L), ("R", grid_R)):
            g = np.asarray(g, np.float32)
            if g.shape != self.panel_grid:
                raise ValueError(f"panel grid {g.shape} != {self.panel_grid}")
            sy, sx = self._panel_slices[key]
            self.map[sy, sx] = g[::-1]

    @property
    def luminance_map(self) -> np.ndarray:
        return self.map

    # ------------------------------------------------------------ sampling
    def grids(self, pose: Pose) -> Tuple[np.ndarray, np.ndarray]:
        """(grid_L, grid_R): float32 (n_dist, n_az) mean luminance along each ray bin."""
        if isinstance(pose, Body):
            x, y, th = pose.x, pose.y, pose.th
        else:
            x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        ang = self._a + th
        px = x + self._d * np.cos(ang)
        py = y + self._d * np.sin(ang)
        ix = np.floor((px + self.half) / self.cell).astype(np.int64)
        iy = np.floor((py + self.half) / self.cell).astype(np.int64)
        n = self.n_cells
        valid = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        vals = self.map[np.clip(iy, 0, n - 1), np.clip(ix, 0, n - 1)]
        vals = np.where(valid, vals, np.float32(0.0))
        g = vals.reshape(2, self.n_dist, self.n_az, self.samples).mean(axis=3)
        g = np.clip(g, 0.0, 1.0).astype(np.float32)
        return g[0], g[1]

    def params(self) -> Dict[str, object]:
        return dict(cell_mm=self.cell, map_cells=self.n_cells, n_dist=self.n_dist, n_az=self.n_az,
                    d_min_mm=self.d_min, d_max_mm=self.d_max, samples_per_bin=self.samples,
                    lum=dict(floor=LUM_FLOOR, wall=LUM_WALL, bezel=LUM_BEZEL, wall_band_mm=WALL_BAND_MM),
                    panel_grid=list(self.panel_grid), dist_edges_mm=[round(float(e), 2) for e in self.dist_edges_mm])


# ====================================================================== self-test
def _self_test() -> None:
    import sys
    import time

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from feeds import make_pair

    phone = Phone()
    dt = 0.016
    n = int(round(60.0 / dt))
    n_seeds = 8
    trails = []
    on_frac, entries, hits_all = [], [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        x0, y0, th0 = random_start_pose(rng, phone)
        body = Body(phone, rng, x0, y0, th0)
        xs, ys = np.zeros(n), np.zeros(n)
        on = np.zeros(n, bool)
        reach_any = np.zeros(n, bool)
        for k in range(n):
            body.step(dt, 0.0, 0.0, 0.0)
            xs[k], ys[k] = body.x, body.y
            on[k] = body.on_phone
            r = body.reach()
            reach_any[k] = r["L"] is not None or r["R"] is not None
        lim = body.limit
        assert np.all(np.abs(xs) <= lim + 1e-6) and np.all(np.abs(ys) <= lim + 1e-6), "left the room"
        ent = int(np.sum(np.diff(on.astype(int)) == 1) + int(on[0]))
        trails.append((xs, ys))
        on_frac.append(float(on.mean()))
        entries.append(ent)
        hits_all.append(body.wall_hits)
        print(f"seed {seed}: start ({x0:6.0f}, {y0:6.0f}) th {math.degrees(th0):5.0f} deg | wall hits {body.wall_hits:2d} | "
              f"on_phone {on.mean()*100:5.1f}% of steps, entries {ent}, any-leg-over-panel {reach_any.mean()*100:5.1f}% | speed {body.v:.1f} mm/s")
    print(f"random walk 60 s x {n_seeds} seeds: {sum(e > 0 for e in entries)}/{n_seeds} seeds visit the phone, "
          f"mean on-phone {np.mean(on_frac)*100:.1f}%, total wall hits {sum(hits_all)}")
    assert sum(hits_all) > 0, "expected wall bounces"
    assert sum(e > 0 for e in entries) > 0, "expected at least one seed to visit the phone in 60 s"

    # eye grids: on a panel vs far away
    feedL, feedR = make_pair(1, 2, scale=0.25)
    eye = PanoramicEye(phone)
    eye.update_panels(feedL.luminance_grid(*PANEL_GRID), feedR.luminance_grid(*PANEL_GRID))
    t0 = time.perf_counter()
    for _ in range(200):
        gL, gR = eye.grids((-40.0, 0.0, 0.0))
    ms = (time.perf_counter() - t0) / 200 * 1000
    print(f"eye.grids: {ms:.3f} ms per call (both eyes), shapes {gL.shape} {gR.shape} dtype {gL.dtype}")
    assert gL.shape == (24, 18) and gR.shape == (24, 18)
    for label, pose in (("on left panel, facing +x", (-40.0, 0.0, 0.0)),
                        ("on right panel, facing +y", (40.0, 0.0, math.pi / 2)),
                        ("far corner, facing centre", (-170.0, -170.0, math.pi / 4)),
                        ("far corner, facing wall", (-170.0, -170.0, -3 * math.pi / 4))):
        gL, gR = eye.grids(pose)
        print(f"  {label:28s} L min {gL.min():.3f} max {gL.max():.3f} mean {gL.mean():.3f} | "
              f"R min {gR.min():.3f} max {gR.max():.3f} mean {gR.mean():.3f} | "
              f"near rows (<20 mm) mean L {gL[:8].mean():.3f} R {gR[:8].mean():.3f}")
    gL_on, _ = eye.grids((-40.0, 0.0, 0.0))
    gL_far, _ = eye.grids((-170.0, -170.0, -3 * math.pi / 4))
    assert gL_on[:8].mean() > gL_far[:8].mean(), "panel under the fly should be brighter than the far floor"
    assert gL_on.max() <= 1.0 and gL_far.min() >= 0.0

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw=dict(width_ratios=[1, 1]))
    ax = axes[0]
    ax.set_facecolor("#111318")
    ax.add_patch(Rectangle((-HALF_MM, -HALF_MM), ROOM_MM, ROOM_MM, fill=False, ec="#888", lw=1.5))
    for prect in phone.rects.values():
        ax.add_patch(Rectangle((prect[0], prect[1]), prect[2] - prect[0], prect[3] - prect[1], fc="#2a3140", ec="#9aa", lw=1))
    cmap = plt.get_cmap("tab10")
    for i, (xs, ys) in enumerate(trails):
        ax.plot(xs, ys, color=cmap(i % 10), lw=0.7, alpha=0.85)
        ax.plot(xs[0], ys[0], "o", color=cmap(i % 10), ms=5)
        ax.plot(xs[-1], ys[-1], "s", color=cmap(i % 10), ms=5)
    ax.set_aspect("equal")
    ax.set_xlim(-HALF_MM - 10, HALF_MM + 10)
    ax.set_ylim(-HALF_MM - 10, HALF_MM + 10)
    ax.set_title(f"Baseline random walk, 60 s x {n_seeds} seeds (o start, square end; "
                 f"{sum(e > 0 for e in entries)}/{n_seeds} visit the phone)", loc="left", fontsize=10)
    ax = axes[1]
    im = ax.imshow(eye.map, origin="lower", cmap="gray", vmin=0, vmax=1,
                   extent=(-HALF_MM, HALF_MM, -HALF_MM, HALF_MM))
    pose = (-40.0, 0.0, 0.0)
    ax.plot(pose[0], pose[1], "o", color="#fb7185", ms=5)
    for a in (LEG_ANGLE_RAD, -LEG_ANGLE_RAD):
        ax.plot([pose[0], pose[0] + REACH_MM * math.cos(pose[2] + a)], [pose[1], pose[1] + REACH_MM * math.sin(pose[2] + a)], color="#fb7185", lw=1)
    ax.set_title("Eye luminance map (2 mm cells) with feed panels", loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = os.path.join(ROOT, "out", "arena_selftest.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    print("self-test passed")


if __name__ == "__main__":
    _self_test()

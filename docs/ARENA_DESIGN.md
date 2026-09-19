# Arena experiment — design contract (Phase 2)

Goal: let the simulated fly **choose** whether to approach and scroll the phone. A walking fly in
a dark box with one open iPhone Duo lying flat in the middle. The brain (MaleCNS LIF, unchanged)
sees the room through a moving panoramic eye, steers the body through its descending neurons, and
swipes only when a front leg is physically over a panel. A mushroom-body dopamine rule lets
"novel post seen while on the phone" act as a reward, so the question *attracted to motion*
vs *learned to want it* becomes testable against a no-dopamine control.

All modules below must run with the existing `src/sim.py`, `src/encoder.py`, `src/decoder.py`,
`src/feeds.py` unchanged in interface (additions allowed, no breaking changes).

## Units and frames
- World: 2-D top-down, millimetres. Room 400 × 400 mm, origin at room centre, x right, y up.
- Phone: open Duo lying flat at the centre, 158 × 111 mm (two portrait panels 79 × 111 mm side
  by side, crease at x = 0). Left panel x ∈ [−79, 0], right panel x ∈ [0, 79], y ∈ [−55.5, 55.5].
- Fly: body length 3 mm is real, but for the video and for a meaningful "reach" we use a scaled
  fly: body length **30 mm**, front-leg reach **22 mm** from body centre at ±35° from heading.
  Document this scale factor (10×) in the README.
- Heading θ in radians, 0 = +x, counter-clockwise positive. Fly "left" is +90° from heading.

## Body model (`src/arena.py`, class `Body`)
Correlated random walk **baseline** identical in every condition, modulated by the brain:
- speed v = v0 + g_v · r_DN, with v0 = 12 mm/s, g_v = 6 mm/s per Hz of descending-neuron
  population rate (Hz per cell, 80 ms trace), clipped to [0, 60] mm/s.
- turning ω = ω_noise − g_ω · (r_DNa_R − r_DNa_L)/(r_DNa_R + r_DNa_L + ε), with ω_noise an
  Ornstein–Uhlenbeck process (σ = 1.2 rad/s, τ = 0.5 s), g_ω = 3 rad/s. Sign convention: θ is
  counter-clockwise positive, so the minus sign makes more right-side DNa activity turn the fly
  **right** (ipsilateral turn, DNa02 convention, see docs/RESEARCH_REFS.md §7). The OU process
  is integrated with the exact discretisation ω ← a·ω + σ·sqrt(1 − a²)·N(0,1), a = exp(−dt/τ),
  so its stationary std is exactly σ (Euler–Maruyama would give σ/sqrt(1 − dt/2τ) ≈ 1.21 rad/s).
- DNa sets: all descending neurons whose type starts with "DNa" on each side; fall back to all
  DNs per side if a set is empty. Expose both as pops so the log records them.
- Walls: on contact, reflect heading away from the wall plus a random ±30° kick.
- Stops: when r_DN < 0.05 Hz and the fly is on the phone, v0 is reduced to 3 mm/s ("standing on
  the screen"); document. Note that the DN population is bursty: its 80 ms trace is below
  0.05 Hz/cell in ~80–90 % of control steps in every condition, so in practice a connected fly
  slows down for most of its time on the phone.
- r_DN, r_DNa_L/R are Hz per cell of the cells actually counted (dn_L ∪ dn_R = 1,304; `dn_all`
  also contains 10 unsided DNs that no population count includes). The same n_dn is used by the
  burst decoder.
- `Body.step(dt, r_dn, r_dna_L, r_dna_R)` updates pose; `Body.front_leg_tips()` returns the two
  tip positions; `Body.panel_under(point)` → "L" | "R" | None.
- Condition "random": `Body(connected=False)` (and g_v = g_ω = 0): the body ignores all three
  rates — no speed modulation, no steering **and no standing rule** — so it is the pure baseline
  walker; the brain is still simulated and logged but not connected to the body or to swipes.
  (Before this flag the standing rule alone made random flies slow to 3 mm/s on the phone
  whenever the DN trace was quiet, i.e. most of the time, inflating the control's dwell time.)
- Random streams: `SeedSequence(seed).spawn(2)` → child 0 seeds the brain (forced Poisson
  spikes), child 1 the body (start pose, OU noise, wall kicks); the feeds keep
  `RandomState(2·seed+1)` / `RandomState(2·seed+2)`.

## Eye (`src/arena.py`, class `PanoramicEye`)
- Luminance map of the room: 2 mm cells (200 × 200). Floor 0.03, walls 0.06 (a 4 mm band),
  phone bezel 0.15, each panel = that feed's `luminance_grid(rows=55, cols=39)` placed on its
  rectangle (panel 111 × 79 mm at 2 mm ≈ 55 × 39 cells), updated every control step.
- Each eye samples a grid of shape **(24, 18)** = (distance rows, azimuth columns) that is fed to
  the existing `EyeEncoder.encode()` (whose hex→grid map was built for a 24 × 18 panel grid; we
  reinterpret the vertical axis as distance-from-fly and the horizontal axis as azimuth; state
  this in the docstring). Azimuth columns: 18 bins of 10° from frontal (col 0 = 0–10°) to rear
  (col 17 = 170–180°) on that eye's side (left eye = left hemifield, right eye mirrored so col 0
  is frontal for both). Distance rows: 24 log-spaced bins from 5 mm to 300 mm (row 0 = nearest,
  i.e. lowest in the visual field). Value = mean luminance of map cells along that ray segment
  (sample ~6 points per bin, nearest-cell lookup; outside the room = 0). `EyeEncoder`'s default
  maps grid row 0 to the *top* of the screen (dorsal, hex2-up), so the arena encoders are built
  with `flip=True` to put the nearest row on the **ventral** retina, as the physical reading
  requires.
- `PanoramicEye.grids(pose)` → (grid_L, grid_R) float32 (24, 18) in [0, 1].

## Swipes and contact
- Same decoder (`burst` mode): a DN burst above 1.5 Hz/cell picks a side by medulla evidence.
- The swipe is **applied only if that side's front-leg tip is over a panel**; the panel under the
  tip is the one swiped (usually the same side, but geometry decides). Otherwise log
  `"swipe_blocked": side` and apply nothing.
- Log per step which panel is under each tip (`reach`).

## Dopamine / mushroom-body plasticity (`src/sim.py` additions)
- Sets: `kc` (4,064 Kenyon cells), `mbon` (97), `pam` (316 appetitive DANs), `ppl1` (16).
- `Brain.enable_plasticity(kc_idx, mbon_idx, dan_idx, eta=..., tau_trace_s=1.0)`: caches the
  positions in `W.data` of every existing KC→MBON edge (postsynaptic MBON row, presynaptic KC
  column), keeps `w0` (initial signed weights), a per-KC eligibility trace e_kc (decays with
  tau_trace, += 1 on each KC spike), and per-MBON dopamine gain
  g_m = Σ_{dan spiked this step} |W0[m, dan]| (dopamine the MBON compartment received this step,
  taken from the wiring; normalise by the mean over MBONs so the rule is scale-free).
- Rule (dopamine-gated depression, flybrain conditioning4 / Hige et al.): every step,
  for each KC→MBON edge (kc, m): w ← w · (1 − eta · g_m · e_kc · dt), bounded below at 0.2·w0
  (sign preserved). No recovery in v1 (document). eta default 0.05 per (unit gain·trace·s).
  The rule fires on **any** PAM spike, not only on the reward drive: spontaneous and visually
  evoked PAM activity (PAM→KC and KC/MBON→PAM wiring) depresses too, so in the `dopamine`
  condition w/w0 drifts below 1 from the first seconds even in runs with no reward event
  (observed: ≈0.998 after 120 s without rewards). Treat the reward-free drift as the control
  against which reward-driven depression is judged; a reward-gated variant would need an
  explicit gate in `_plasticity_step`.
- `Brain.dopamine_drive(dan_idx, rate_hz)`: adds the DAN set to the forced-Poisson drive for
  the current step (same mechanism as sensory drive).
- Diagnostics: `Brain.plasticity_stats()` → dict(mean_w_over_w0, frac_edges_changed, n_edges).
- Unit test in `src/test_plasticity.py`: drive 200 KCs at 50 Hz + PAM at 100 Hz for 1 s →
  mean w/w0 on their edges < 0.9; same without PAM → == 1.0; KCs not driven → unchanged.

## Reward definition (`src/run_arena.py`)
- A **reward event** fires when a "novel" card becomes visible (rising edge of `novel_visible`)
  on a panel while the fly is on the phone (body centre within the phone rectangle). It drives
  `pam` at 100 Hz for 300 ms. Conditions:
  - `real`     : real wiring, brain steers, swipes gated by contact, **no** dopamine.
  - `dopamine` : as real, plus reward → PAM drive and KC→MBON plasticity enabled.
  - `shuffled` : wiring-shuffled graph (`Connectome.shuffled(0)`), brain steers, no dopamine.
  - `random`   : real wiring simulated for logging, but body uses baseline walk only and swipes
    are never applied (g_v = g_ω = 0).
- Start pose: random position ≥ 120 mm from the phone centre, random heading (seeded).
- Duration default 120 s per run; seeds 0–4 per condition.

## Log format (`out/arena_<cond>_s<seed>.jsonl`)
First line `{"meta": {...}}` with: condition, seed, room, phone rects, body params (v0, g_v, g_ω,
scale), eye params, lif config, plasticity params, reward params, n_neurons, sets sizes.
Then one line per 16 ms control step:
```
{"step", "t", "pose": {"x","y","th","v"}, "on_phone": bool, "dist_mm": float,
 "reach": {"L": "L"|"R"|null, "R": ...}, "swipe": null|"L"|"R", "swipe_blocked": null|"L"|"R",
 "burst_hz": float, "side_ev": float, "spikes": int,
 "pops": {"eye_L","eye_R","dn_L","dn_R","dna_L","dna_R","leg_L","leg_R","kc","mbon","pam"},
 "reward": bool, "w_ratio": float (mean w/w0 of KC->MBON, 1.0 if plasticity off),
 "novel": {"L": bool, "R": bool}, "posts": {"L": int, "R": int}}
```
`pose`, `on_phone`, `dist_mm` and `reach` are the values **after** the step's `Body.step`
(the swipe gate itself uses the pre-step reach, i.e. the pose at time t). `spikes` is the total
number of spikes in the 8 substeps **including the forced ones** (lamina L1/L2 Poisson drive,
≈ two thirds of the total, and the PAM drive inside a reward window); `pops.eye_L/eye_R` are
exactly those forced lamina spikes if brain-generated activity is wanted.
Also `out/arena_<cond>_s<seed>_spikes.npz` (SpikeRecorder frames) **only** for the run that
will be rendered (flag `--spikes`).

## Analysis (`src/analyze_arena.py`)
Per run: time-on-phone fraction, time-within-reach fraction, number of visits (entries onto the
phone), mean visit duration, swipes, blocked swipes, approach rate after novel cuts (probability
of reducing distance by > 20 mm within 2 s after a novel card appears while off the phone,
vs matched random times), first-half vs second-half time-on-phone (learning), MBON/KC mean
rates, final w_ratio. Across seeds: mean ± 95 % bootstrap CI per condition; paired comparison
dopamine vs real per seed. Figures: trajectories per condition (5 seeds overlaid, phone drawn),
time-on-phone bars with CIs, distance-to-phone over time, w_ratio over time. Output
`out/arena_summary.json` + `out/arena_*.png`; be honest if effects are null.

## Video (`src/render_arena.py`)
Top-down arena view 1080 × 1080 at the top (room, phone with live replayed feeds, the fly sprite
rotated to heading, trail, reward flashes), below it the brain PiP and a HUD: condition, time on
phone, visits, swipes, dopamine events, KC→MBON weight ratio, distance. Storyboard: title,
split-screen comparison card (real vs dopamine vs shuffled vs random time-on-phone bars) near the
end, attribution. Replays feeds deterministically from seeds + swipe times as `render.py` does.

## Training protocol (README 9.2)
- `--pin-on-phone`: start pose (0, 0, +90°); after every `Body.step` the pose is reset to it and
  v = 0, so the body never moves (the OU/wall random draws still advance). Left tip over L,
  right tip over R (reach 22 mm at ±35°).
- `--forced-swipe-s P`: at t = P, 2P, ... a swipe is applied to the panel under the alternating
  L/R tip and logged with `forced: true` (`swipes_forced` in the summary); a brain-triggered
  swipe is skipped in a step that already swiped by force.
- `--save-weights` (needs `dopamine`): `Brain.export_plastic_weights()` -> npz with rows (MBON),
  cols (KC), w, w0, source/condition/seed/duration/rewards.
- `--load-weights` (any condition; required by `trained`): `Brain.import_weights(rows, cols, w)`
  overwrites exactly those edges before the run; `meta.loaded_weights` records path, n_edges,
  mean/min w/w0. Condition `trained` = `real` + loaded weights, plasticity off.
- Only the weights persist between training and test; all dynamic state starts fresh.

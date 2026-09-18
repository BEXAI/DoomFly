# DoomFly — a fly brain double-doomscrolls an iPhone Duo

A 50 s vertical (1080 × 1920, 60 fps) video in which a simulated fruit-fly brain — the complete
**MaleCNS v1.0 connectome** (166,700 neurons, 6.24 M connections) running as a
leaky-integrate-and-fire spiking network — sits on an open foldable phone mock-up and
scrolls two feeds at once: **left eye → left panel, right eye → right panel; front-left leg
swipes the left feed, front-right leg swipes the right feed**. A picture-in-picture shows
the real neurons firing.

Nothing in the video is scripted. Every swipe is emitted by a fixed, unfitted rule over the
spikes of real MaleCNS neurons: *a swipe happens when the descending-neuron population bursts,
and it goes to the side whose medulla saw the change.* What is chosen by hand, and what is
not, is stated in [§6](#6-what-is-and-is-not-learned-or-fitted).

Deliverables: `out/final.mp4` (rendered from `out/events_final_s4.jsonl` + `out/spikes_final_s4.npz`),
this README, `docs/` (verified data notes, all test results as JSON, figures).

---

## 1. Pipeline

```
data/raw/*.feather ──► src/build_graph.py ──► data/graph/{W.npz, ids.npy, sets.json, eye_map.npz, positions.npy, groups.npy}
                                                        │
 src/feeds.py  (two synthetic feeds, 1335×1878 each,    │
                autoplaying cards)                       ▼
      │ luminance grid per panel (24×18)        src/sim.py  (LIF, dt = 2 ms, scipy CSC, STD)
      ▼                                                 │
 src/encoder.py  (retinotopic L1/L2 Poisson drive) ──►  brain.step() ×8 = one 16 ms control step
                                                        │ spike counts: DNs, medulla L/R, leg MNs
                                                        ▼
                                                src/decoder.py  (burst rule → swipe L / R)
                                                        │ feed.swipe()
                                                        ▼
 src/run_episode.py  logs out/events_<tag>.jsonl + out/spikes_<tag>.npz  ──►  src/render.py  ──►  out/final.mp4
 src/validate.py     Tests A–C (Shiu benchmark, runaway, wiring shuffle)
 src/analyze_episode.py  swipe statistics + figure for any episode
```

Everything ran offline on 4 CPU cores / 15 GB RAM, no GPU. A 50 s episode simulates in ~70 s
(18 ms per 16 ms control step) and renders in ~100 s (30 fps).

## 2. Data

MaleCNS v1.0 flat connectome (CC-BY 4.0), three files, ~1.1 GB
(`src/download.py`, bucket `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`):

| file | used for |
|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | bodyId, type, superclass, class, subclass, somaSide / rootSide, somaNeuromere, `assignedOlHex1/2` (optic-lobe column coordinates), `somaLocation` |
| `body-neurotransmitters-male-cns-v1.0.feather` | `consensus_nt` per body (synister predictions, Eckstein, Bates et al. 2024) |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 151.9 M `body_pre, body_post, weight` rows |

Column-level notes with printed value counts: `docs/RESEARCH_DATA.md`.
Reference-implementation notes (Shiu constants, sign conventions, licences): `docs/RESEARCH_REFS.md`.

### Graph build (`src/build_graph.py`)

| choice | value |
|---|---|
| neurons kept | every body with a non-null `superclass` → **166,700** |
| edges kept | synapse count ≥ 5, both ends kept, 33 self-edges dropped → **6,242,085 edges, 89,859,936 synapses** |
| sign | presynaptic `consensus_nt`: ACh, DA, OA, 5-HT → +1; GABA, glutamate (GluCl, Liu & Wilson 2013), histamine → −1; unclear/missing (3,177 cells) → +1 |
| inhibitory edge fraction | **37.2 %** |
| NT counts | ACh 103,720 · Glu 29,302 · GABA 22,069 · His 7,891 · unclear 2,999 · DA 392 · OA 101 · 5-HT 48 · missing 178 |
| side | `somaSide` → `rootSide` (sensory cells) → `instance` suffix `_L/_R` |
| soma positions | `somaLocation` × 8 nm for 139,662 cells; 26,236 more filled by the mean of synaptic partners (4 rounds); 802 unknown |

These match the community builds (hotocoo/malecns: 166,700 / 6,242,118 / 89.86 M).

### Neuron sets (`data/graph/sets.json`)

| role | selection | L | R |
|---|---|---|---|
| eye input, ON | type `L1` (lamina) with hex column coords | 875 | 892 |
| eye input, OFF | type `L2` | 874 | 893 |
| medulla (side evidence) | types `Mi1`, `Tm1`, `Tm2`, `Tm9`, `L5` — direct targets of L1/L2 | 4,429 | 4,445 |
| optic lobe | superclass `ol_intrinsic` | 44,607 | 44,796 |
| visual projection neurons | superclass `visual_projection` | 4,589 | 4,612 |
| **descending neurons (swipe trigger)** | superclass `descending_neuron` | 656 | 648 (+10 midline) = **1,314** |
| front-leg motor neurons | superclass `vnc_motor` & subclass `fl` (T1) | 68 | 67 |
| all motor neurons | `vnc_motor` + `cb_motor` | 815 | |
| sugar GRNs (proxy) | labellar bristle types `LB3a-d` | 78 | |
| bitter GRNs (proxy) | labellar bristle types `LB1a-e` | 56 | |
| MN9 | type `MN9` (bodyIds 10331 L, 16949 R) | 2 | |
| PAM / PPL1 / KC / MBON | for Phase 2 (unused) | 316 / 16 / 4,064 / 97 | |

Caveats found in the data: the left eye is under-reconstructed (R1–R6 photoreceptors 1,112 L vs
2,265 R; `L3`/`C2`/`Tm4` have column coordinates only on the right), so drive is injected at
L1/L2 (complete on both sides) and never at photoreceptors (histamine sign trap). MaleCNS has
**no sugar/bitter GRN label**; the LB3/LB1 split is a connectivity proxy (strongest two-hop path to
MN9), so Test A is reported with that caveat.

## 3. Neuron model (`src/sim.py`)

Shiu et al. (Nature 2024) LIF constants, verified line-by-line against
`philshiu/Drosophila_brain_model` and `hotocoo/malecns` (`docs/RESEARCH_REFS.md` §1):

| constant | value |
|---|---|
| V_rest = V_reset | −52 mV |
| V_threshold | −45 mV |
| refractory | 2.2 ms (→ 1 step at dt = 2 ms) |
| membrane τ | 20 ms |
| synaptic τ (exponential) | 5 ms |
| PSP per synapse | 0.275 mV peak (τ_m/τ_s pre-compensation as hotocoo) |
| synaptic delay | 1.8 ms (→ 1 step) |
| dt / control step | 2 ms / 16 ms (8 substeps) |
| **`weight_scale`** | **0.5** (Test B) |
| spike-frequency adaptation | 0.6 mV, τ 120 ms (hotocoo's addition; 0 = pure Shiu) |
| **short-term depression** (Tsodyks–Markram, one resource per presynaptic neuron) | **U = 0.15 per spike, recovery 480 ms** (TheMrRaGe/flybrain measured U = 0.08, τ 480 ms; it must be global) |

Sensory input follows Shiu: driven cells are forced to spike as Poisson processes at the
requested rate; everything downstream is the wiring. The step is a sparse column gather on the
CSC matrix (only columns of neurons that spiked are touched): 1–20 ms per 2 ms step on 4 cores.

## 4. Eyes and legs

**Feeds (`src/feeds.py`).** Two deterministic synthetic feeds (seeded card generators, no text
glyphs, no logos, no real posts). Image cards and "novel" high-contrast cards **autoplay**: like
muted clips they cut to a new brightness level about once a second (period and phase fixed per
card, ±80 % for image cards, ±60 % for novel cards). Between cuts the frame is static. This is the
ambient stimulus — a property of the content, not of the brain. A swipe flicks the feed by 0.55
screen heights with an ease-out over 350 ms.

**Encoder (`src/encoder.py`).** Each panel → 24 × 18 luminance grid. The ~875 L1 and ~875 L2
columns of each eye carry MaleCNS hex coordinates (`assignedOlHex1/2`); axial → cartesian
(x = h1 − h2/2, y = h2·√3/2 gives an upright 1.15:1 ellipse), PCA-upright, normalised, binned
(≈ 2.5 columns per populated grid cell). Temporal contrast c = (I − I_bg)/(I_bg + 0.05) against a
250 ms adapting background; c > 0 drives that column's **L1** (ON), c < 0 its **L2** (OFF),
rate = clip(6·|c|, 0, 1) × 150 Hz. Static frames go silent within ~1 s; a swipe gives a ~110 Hz burst.

**Decoder (`src/decoder.py`, mode `burst`, no fit).** Two population signals, both real neurons:
1. *Trigger*: an 80 ms trace of the spike rate of all **1,314 descending neurons**. When it exceeds
   **1.5 Hz per cell** (baseline is 0), a swipe is emitted (≥ 400 ms per panel, ≥ 300 ms globally).
2. *Side*: an 800 ms trace of the per-cell rate of the **left vs right medulla** (Mi1/Tm1/Tm2/Tm9/L5,
   one synapse downstream of the driven lamina cells). The panel of the more active medulla is swiped.

Two other readouts were built and rejected, and are kept in the code for the record:
`direct` (front-leg motor-neuron pool z-score — the legs fire on every burst but bilaterally, so it
cannot pick a panel) and `ridge` (hotocoo-style linear readout fitted to a teacher — see §5.4).

## 5. Results

All numbers below are printed from `docs/validate_results.json`, `docs/episode_*.json`,
`docs/side_eval_open_loop.txt` and `docs/ridge_fit_diag.json`.

### 5.1 Test A — Shiu benchmark (sugar → MN9), `src/validate.py`

78 sugar-proxy GRNs at 150 Hz for 1 s; MN9 rate (Hz per cell) during the drive; baseline 0 Hz.

| weight_scale | sugar | sugar + bitter | bitter | population rate | state |
|---|---|---|---|---|---|
| 1.0 (pure Shiu) | 106.5 | 101.5 | 67.0 | 22 Hz, 24 % of neurons | runaway |
| 0.5 | 186.5 | 76.0 | 114.5 | 13.7 Hz | runaway |
| 0.3 | 147.0 | 89.5 | 23.0 | 8 Hz plateau | latched |
| 0.15 | 107.5 | 0.0 | 0.0 | 2.7 Hz | latched at 3 Hz after drive |
| 0.15 + adapt 0.6 | 13.5 | 0.0 | 0.0 | 0.95 Hz | ~quiescent after drive |

Bitter suppresses MN9 at every scale, most cleanly at 0.15 (108 → 0 Hz). Absolute MN9 rates
are physiological only with adaptation. The sugar/bitter sets are proxies (§2), so this is a
consistency check, not a reproduction of Shiu's exact experiment. Figure: `docs/figures/validate_A_mn9.png`.

### 5.2 Test B — runaway, silence, and the operating point

* No input, 1 s: **0 spikes** at every configuration (no noise term).
* Left-eye drive (L1+L2 at 60 Hz, 1.5 s), no depression: population rate 26 Hz at scale 1.0,
  15.6 Hz at 0.5 (both latched, persisting after the drive stops); ≤ 1.2 Hz and dying instantly
  at ≤ 0.4. The cliff is between 0.4 and 0.5. But at ≤ 0.4 **nothing leaves the optic lobe**
  (VPN 0.02 Hz, DN 0.00 Hz, motor 0).
* Closed-loop probes (8 s, swipes as stimulus): 0.2 → activity stays in the optic lobe; 0.25–0.3 →
  the network latches into a 4 Hz self-sustained state after the first swipe, in which descending
  activity is constant and input-blind.
* **With global short-term depression** (U = 0.15) at scale 0.5 (+ adapt 0.6): no latch;
  descending neurons are silent between events and burst on each visual onset (62 DN spikes/step
  in the 300 ms after a swipe vs 8 before; 0 in quiet periods). U = 0.08 at 0.5 still latched. This
  is the operating point used everywhere below. Mean population rate in the episodes: 0.4–0.6 Hz;
  DN rate at swipe 7.7–9.2 Hz per cell vs 1.5–2.5 Hz one second earlier.

Figure: `docs/figures/validate_B_lateral.png`.

### 5.3 Test C — it is the wiring

*Lateralisation under left-eye drive* (`validate.py`, scale 0.4): optic-lobe rate L 2.16 vs R 0.01 Hz,
index +0.99 on the real graph; on the wiring-shuffled graph (same weights and signs, postsynaptic
targets permuted) L 41.6 vs R 46.6 Hz, index −0.06, and the shuffled graph runs away (53 Hz) where the
real one is quiet.

*One-sided probes at the operating point* (`out/events_lat_L/R.jsonl`): left-only swipes give
55,000 left-eye spikes and **0** right-eye spikes per event; descending neurons are silent before
and fire ~1,800 spikes in the 600 ms after; the first visual-projection spikes are ipsilateral for
~50 ms, then the response is bilateral with a right bias (the right optic lobe is better
reconstructed and carries more synapses).

*Open-loop side test* (scripted swipes, 15 s left-only then 15 s right-only, decoder only logs
what it would do; `docs/side_eval_open_loop.txt`):

| graph | left block | right block | overall |
|---|---|---|---|
| **real wiring** | 10 / 12 | 13 / 13 | **23 / 25 = 92 %** |
| shuffled wiring | 0 / 19 | 13 / 13 | 13 / 32 = 41 % |

*Closed loop on the shuffled graph* (`docs/episode_shuffled.json`): it still bursts (35 swipes),
but every one of them goes right (0 L / 35 R, posts 0 / 55): the shuffled brain can be startled
but does not know which eye saw the change.

### 5.4 The fitted readout is a negative result

A ridge readout from 2 × 2,129 DN + MN traces (fast 80 ms, slow 800 ms) to a teacher that swiped
each panel at random 1–1.8 s intervals (100 s, 138 teacher swipes, λ chosen on held-out blocks)
reaches held-out correlation **0.02 (L) / 0.06 (R)** (`docs/ridge_fit_diag.json`). The
descending/motor state does not encode "how long ago was this panel still", so this readout was
dropped rather than tuned.

### 5.5 The episode (`docs/episode_final_s4.json`, video)

Six 50 s runs at the operating point, differing only in the Poisson seed:

| seed | swipes | L / R | longest quiet gap | side = eye that saw the onset¹ | DN at swipe vs 1 s before |
|---|---|---|---|---|---|
| 0 | 13 | 9 / 4 | 19.5 s | 85 % | 9.1 / 1.7 Hz |
| 1 | 17 | 7 / 10 | 17.5 s | 82 % | 7.7 / 2.0 Hz |
| 2 | 18 | 14 / 4 | 14.4 s | 83 % | 8.2 / 2.5 Hz |
| 3 | 8 | 0 / 8 | 34.2 s | 75 % | 8.5 / 1.5 Hz |
| **4 (video)** | **29** | **9 / 20** | **4.3 s** | 72 % | 9.2 / 2.1 Hz |
| 5 | 13 | 9 / 4 | 12.4 s | 100 % | 8.0 / 1.8 Hz |

¹ fraction of swipes whose panel had the larger eye drive in the preceding 250 ms; confounded in
closed loop because the swipe itself is the strongest onset, hence the open-loop test in §5.3.

Seed 4 was chosen for the video because it covers both panels with no long pause; the dynamics
are the same in all six (bursts locked to onsets, swipe chains of 2–3 at 0.4 s spacing, then
habituation until the next autoplay cut ignites a burst). The right bias is real wiring asymmetry.
Video episode: 29 swipes (9 L / 20 R), 11 / 32 posts consumed, 4.6 M spikes, 0.61 Hz mean rate,
front-leg MNs fire 55 (L) / 36 (R) spikes per swipe. Figure: `docs/figures/episode_final_s4.png`.

## 6. What is and is not learned or fitted

- **Not fitted / not scripted:** the wiring (every synapse count and sign is MaleCNS), the neuron
  model, the retinotopic input map (real hex column coordinates), which cells are driven (L1/L2),
  which cells are read (all descending neurons; medulla per side), the swipe rule.
- **Chosen by hand (and reported):** `weight_scale` 0.5, adaptation 0.6 mV, STD U 0.15 / 480 ms,
  encoder gain 6 and 150 Hz cap, DN threshold 1.5 Hz, side-trace 0.8 s, refractory 0.4 s / 0.3 s,
  autoplay amplitude and rate, the seed shown.
- **Fitted:** nothing in the video pipeline. (The ridge readout in §5.4 was fitted and rejected.)
- **Stimulus design:** the feeds autoplay. Without any change on the screens the brain is silent
  (Test B), so the loop needs content that moves, exactly like real feeds.
- **Never claimed:** awareness, intent, or that a fly would do this. It is a static wiring diagram
  driven by LIF dynamics with a fixed population readout.

## 7. Video (`src/render.py`)

Pure OpenCV/numpy compositing at 1080 × 1920, 60 fps (≈ 30 fps render): dark desk, open two-panel
device mock-up (two 1335 × 1878 portrait panels, crease, titanium bevel, **no logo, no real UI**), a
stylised top-down Drosophila perched over the hinge whose front-left / front-right leg flicks on each
swipe, eye → panel glow cues in the intro, a 400 × 400 PiP of the 165,898 soma positions (L eye cyan,
R eye orange, DNs green, front-leg MNs magenta) lit frame-accurately by `spikes_final_s4.npz`, a HUD
with the descending-neuron burst signal, posts consumed L / R, swipes, spikes/s with a sparkline,
title card (0–4 s), stats card (38–46 s) and attribution card (46–50 s). Feed frames are replayed
deterministically from the swipe times in `events_final_s4.jsonl` (same seeds and autoplay schedule).

## 8. Reproduce

```bash
cd ~/Projects && git clone https://github.com/BEXAI/DoomFly.git && cd DoomFly
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas pyarrow pillow opencv-python-headless matplotlib tqdm imageio imageio-ffmpeg
python3 src/download.py                                  # ~1.1 GB
python3 src/build_graph.py                               # ~3 min, peaks ~8 GB RAM
python3 src/validate.py --quick                          # Tests A-C -> out/validate_results.json + figures
python3 src/run_episode.py --duration 50 --seed 4 --tag final_s4          # the episode (burst mode is the default)
python3 src/run_episode.py --duration 50 --tag shuffled --shuffle 0 --no-spikes   # wiring-shuffled control
python3 src/run_episode.py --duration 30 --tag side_real --no-spikes --scripted L:15,R:15   # open-loop side test
python3 src/analyze_episode.py final_s4 shuffled          # figures + JSON
python3 src/render.py --events out/events_final_s4.jsonl --spikes out/spikes_final_s4.npz \
        --positions data/graph/positions.npy --groups data/graph/groups.npy --out out/final.mp4
```

## 9. Attribution (CC-BY)

> Connectome: MaleCNS v1.0 — HHMI Janelia FlyEM Project Team, the Cambridge Drosophila
> Connectomics Group (University of Cambridge / MRC Laboratory of Molecular Biology), and Google
> Research Connectomics. CC-BY 4.0. https://male-cns.janelia.org/ — Berg et al., "Sexual dimorphism
> in the complete connectome of the Drosophila male central nervous system," *Cell* (2026),
> doi:10.1016/j.cell.2026.08.015. Neurotransmitter predictions: Eckstein, Bates et al. (2024) /
> funkelab. Neuron model after Shiu et al., *Nature* (2024). Simulation design informed by
> hotocoo/malecns and TheMrRaGe/flybrain (re-implemented; those repositories carry no licence file,
> so no code was copied). iPhone Duo is a trademark of Apple Inc.; the device shown is a mock-up;
> feeds are synthetic and contain no real posts, apps or logos.

Suggested caption: *A real fruit-fly connectome (MaleCNS v1.0, 166,700 neurons) simulated as a
spiking network, doomscrolling two feeds at once — one per eye, one per leg. Every swipe is a
burst of the fly's own descending neurons, sent to the side whose eye saw the change. Nothing
scripted, nothing fitted. Data CC-BY Janelia FlyEM · Cambridge · Google.*

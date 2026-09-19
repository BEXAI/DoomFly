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

Deliverables: the video `out/final.mp4` and its sources `out/events_final_v2.jsonl` +
`out/spikes_final_v2.npz` are **generated locally** by the commands in §8 (media and logs are
git-ignored; the video was shared directly). The repository carries the code, this README and
`docs/`: verified data notes, every test result as JSON, the episode statistics, and figures.

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
 src/analyze_episode.py  swipe statistics + figure for any episode;  src/side_eval.py  open-loop side test
```

Everything ran offline on 4 CPU cores / 15 GB RAM, no GPU. A 50 s episode simulates in
60–80 s (19–26 ms per 16 ms control step, depending on activity) and renders in ~100 s (30 fps).

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
| edges kept | synapse count ≥ 5, both ends kept, 33 self-edges dropped → **6,242,085 edges, 89,859,938 synapses** |
| sign | presynaptic `consensus_nt`: ACh, DA, OA, 5-HT → +1; GABA, glutamate (GluCl, Liu & Wilson 2013), histamine → −1; unclear/missing (3,177 cells) → +1 |
| inhibitory edge fraction | **37.2 %** |
| NT counts | ACh 103,720 · Glu 29,302 · GABA 22,069 · His 7,891 · unclear 2,999 · DA 392 · OA 101 · 5-HT 48 · missing 178 |
| side | `somaSide` → `rootSide` (sensory cells) → `instance` suffix `_L/_R` |
| soma positions | `somaLocation` × 8 nm for 139,662 cells; 26,236 more filled by the mean of synaptic partners (4 rounds); 802 unknown |

These match the community builds: hotocoo/malecns reports 166,700 neurons and 6,242,118 edges
(89.86 M synapses); the 33-edge difference is exactly the self-edges dropped here.

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
**L1/L2 only** (complete on both sides: 1,749 cells left, 1,785 right) and never at photoreceptors
(histamine sign trap) or L3. MaleCNS has **no sugar/bitter GRN label**; the LB3/LB1 split is a
connectivity proxy (strongest two-hop path to MN9), so Test A is reported with that caveat.

## 3. Neuron model (`src/sim.py`)

Shiu et al. (Nature 2024) LIF constants, verified line-by-line against
`philshiu/Drosophila_brain_model` and `hotocoo/malecns` (`docs/RESEARCH_REFS.md` §1):

| constant | value |
|---|---|
| V_rest = V_reset | −52 mV |
| V_threshold | −45 mV |
| refractory | 2.2 ms (→ 1 step at dt = 2 ms; a cell can fire again two steps after a spike, i.e. a 250 Hz ceiling) |
| membrane τ | 20 ms |
| synaptic τ (exponential) | 5 ms |
| synaptic weight | 0.275 mV per synapse, integrated (τ_m/τ_s pre-compensation as hotocoo; the peak PSP of one synapse at dt = 2 ms is ≈ 0.15 mV) |
| synaptic delay | 1.8 ms (→ 1 step) |
| dt / control step | 2 ms / 16 ms (8 substeps) |
| **`weight_scale`** | **0.5**, set by the closed-loop probes with depression (§5.2); Test B without depression recommends 0.4 |
| spike-frequency adaptation | 0.6 mV, τ 120 ms (hotocoo's addition; 0 = pure Shiu) |
| **short-term depression** (Tsodyks–Markram, one resource per presynaptic neuron) | **U = 0.15 per spike, recovery 480 ms** (TheMrRaGe/flybrain measured U = 0.08, τ 480 ms; it must be global). The resource is depleted at spike time and weights delivery one step later, so a rested cell transmits 0.85 of its weight; the effective gain of a first spike is therefore 0.5 × 0.85 ≈ 0.43 |

Sensory input follows Shiu: driven cells are forced to spike as Poisson processes at the
requested rate; everything downstream is the wiring. The step is a sparse column gather on the
CSC matrix (only columns of neurons that spiked are touched): 1–20 ms per 2 ms step on 4 cores.

## 4. Eyes and legs

**Feeds (`src/feeds.py`).** Two deterministic synthetic feeds (seeded card generators, no text
glyphs, no logos, no real posts). Image cards and "novel" high-contrast cards **autoplay**: like
muted clips they cut to a new brightness level about once a second (period and phase fixed per
card, ±80 % for image cards, ±60 % for novel cards). Between cuts the frame is static. This is the
ambient stimulus — a property of the content, not of the brain. A swipe flicks the feed by 0.55
screen heights with an ease-out over 350 ms. The renderer replays the feeds from the seeds and
autoplay parameters recorded in the episode's meta line, so no frames are stored.

**Encoder (`src/encoder.py`).** Each panel → 24 × 18 luminance grid. The ~875 L1 and ~875 L2
columns of each eye carry MaleCNS hex coordinates (`assignedOlHex1/2`); axial → cartesian
(x = h1 − h2/2, y = h2·√3/2 gives an upright 1.15:1 ellipse), PCA-upright, normalised, binned
(≈ 2.5 columns per populated grid cell). Temporal contrast c = (I − I_bg)/(I_bg + 0.05) against a
250 ms adapting background; c > 0 drives that column's **L1** (ON), c < 0 its **L2** (OFF),
rate = clip(6·|c|, 0, 1) × 150 Hz. Static frames go silent within ~1 s; a swipe gives a ~110 Hz burst.

**Decoder (`src/decoder.py`, mode `burst`, no fit).** Two population signals, both real neurons:
1. *Trigger*: an 80 ms trace of the spike rate of all **1,314 descending neurons**. When it exceeds
   **1.5 Hz per cell** (baseline is 0), a swipe is emitted (≥ 400 ms per panel, ≥ 300 ms globally,
   none in the first second).
2. *Side*: an 800 ms trace of the per-cell rate of the **left vs right medulla** (Mi1/Tm1/Tm2/Tm9/L5,
   one synapse downstream of the driven lamina cells). The panel of the more active medulla is swiped.

Two other readouts were built and rejected, and are kept in the code for the record:
`direct` (front-leg motor-neuron pool z-score — the legs fire on every burst but bilaterally, so it
cannot pick a panel) and `ridge` (hotocoo-style linear readout fitted to a teacher — see §5.4).

## 5. Results

All numbers below are printed from `docs/validate_results.json`, `docs/episodes/*.json`,
`docs/probes/*.json` and `docs/ridge_fit_diag.json`. An earlier set of runs drove the right eye's
L3 cells as well (a code default; the left eye has no L3 columns); that asymmetry was found in
audit and removed, and every episode number below is from the corrected L1/L2-only runs.

### 5.1 Test A — Shiu benchmark (sugar → MN9), `src/validate.py`

Tests A–C run **without** short-term depression (it was added after them, for the closed loop);
the depression regime is characterised by the closed-loop probes in §5.2.

78 sugar-proxy GRNs at 150 Hz for 1 s; MN9 rate (Hz per cell) during the drive; baseline 0 Hz.

| weight_scale | sugar | sugar + bitter | bitter | population rate | state |
|---|---|---|---|---|---|
| 1.0 (pure Shiu) | 106.5 | 101.5 | 67.0 | 22 Hz, 24 % of neurons | runaway |
| 0.5 | 186.5 | 76.0 | 114.5 | 13.7 Hz | runaway |
| 0.3 | 147.0 | 89.5 | 23.0 | 8 Hz plateau | latched |
| 0.15 | 107.5 | 0.0 | 0.0 | 2.7 Hz | latched at 3 Hz after drive |
| 0.15 + adapt 0.6 | 13.5 | 0.0 | 0.0 | 0.95 Hz | 0.9 Hz after drive (below the 1 Hz latch criterion) |

Bitter suppresses MN9 at every scale, most cleanly at 0.15 (108 → 0 Hz). Absolute MN9 rates
are physiological only with adaptation. The sugar/bitter sets are proxies (§2), so this is a
consistency check, not a reproduction of Shiu's exact experiment. Figure: `docs/figures/validate_A_mn9.png`.

### 5.2 Test B — runaway, silence, and the operating point

* No input, 1 s: **0 spikes** at every configuration (no noise term).
* Left-eye drive (all 1,770 left L1+L2 cells at 60 Hz, 1.5 s), no depression: population rate
  26 Hz at scale 1.0, 15.6 Hz at 0.5 (both latched, persisting after the drive stops); ≤ 1.2 Hz and
  dying instantly at ≤ 0.4. The cliff is between 0.4 and 0.5. But at ≤ 0.4 **nothing leaves the
  optic lobe** (VPN 0.02 Hz, DN < 0.01 Hz, motor 0).
* Closed-loop probes (8 s, swipes as stimulus; `docs/probes/closed_loop_probes.json`): at 0.2 the
  activity stays in the optic lobe; at 0.25–0.3 the network latches after the first swipe into a
  3–4 Hz self-sustained state in which descending activity is constant (≈ 120 DN spikes per step
  before and after a swipe alike) and input-blind.
* **With global short-term depression** (U = 0.15) at scale 0.5 (+ adapt 0.6): no latch;
  descending neurons are silent between events and burst on each visual onset (62 DN spikes per
  step in the 300 ms after a swipe vs 8 before; 0 in quiet periods). U = 0.08 at 0.5 still latched
  (162 vs 131). This is the operating point used everywhere below. Mean population rate in the
  50 s episodes: 0.33–0.67 Hz; DN rate at swipe 7.9–10.7 Hz per cell vs 1.3–2.4 Hz one second earlier.

Figure: `docs/figures/validate_B_lateral.png`.

### 5.3 Test C — it is the wiring

*Lateralisation under left-eye drive* (`validate.py`, scale 0.4): optic-lobe rate L 2.16 vs R 0.01 Hz,
index +0.99 on the real graph; on the wiring-shuffled graph (same weights and signs, postsynaptic
targets permuted) L 41.6 vs R 46.6 Hz, index −0.06, and the shuffled graph runs away (53 Hz) where the
real one is quiet.

*Open-loop side test* (`src/side_eval.py`; scripted swipes, 15 s left-only then 15 s right-only, the
decoder only logs what it would do; `docs/episodes/side_eval_open_loop.txt`):

| graph | left block | right block | overall |
|---|---|---|---|
| **real wiring** | 10 / 11 | 13 / 13 | **23 / 24 = 96 %** |
| shuffled wiring | 1 / 18 | 13 / 13 | 14 / 31 = 45 % |

*Closed loop on the shuffled graph* (`docs/episodes/episode_v2_shuffled.json`): it still bursts
(35 swipes), but every one of them goes right (0 L / 35 R, posts 0 / 55): the shuffled brain can be
startled but does not know which eye saw the change.

### 5.4 The fitted readout is a negative result

A ridge readout from 2 × 2,129 DN + MN traces (fast 80 ms, slow 800 ms) to a teacher that swiped
each panel at random 1–1.8 s intervals (100 s, 138 teacher swipes, λ chosen on held-out blocks)
reaches held-out correlation **0.02 (L) / 0.06 (R)** (`docs/ridge_fit_diag.json`; run before the L3
fix, not repeated). The descending/motor state does not encode "how long ago was this panel still",
so this readout was dropped rather than tuned.

### 5.5 The episode (`docs/episodes/episode_v2_s1.json`, video)

Six 50 s runs at the operating point, differing only in the Poisson seed:

| seed | swipes | L / R | longest quiet gap | side = eye that saw the onset¹ | DN at swipe vs 1 s before | mean rate |
|---|---|---|---|---|---|---|
| 0 | 27 | 9 / 18 | 5.4 s | 67 % | 8.6 / 1.5 Hz | 0.67 Hz |
| **1 (video)** | **28** | **14 / 14** | **5.2 s** | **96 %** | 7.9 / 2.4 Hz | 0.62 Hz |
| 2 | 27 | 2 / 25 | 6.0 s | 81 % | 9.3 / 1.3 Hz | 0.57 Hz |
| 3 | 17 | 1 / 16 | 11.7 s | 76 % | 10.1 / 2.1 Hz | 0.49 Hz |
| 4 | 9 | 0 / 9 | 28.1 s | 89 % | 8.1 / 1.6 Hz | 0.35 Hz |
| 5 | 9 | 0 / 9 | 33.4 s | 100 % | 10.7 / 1.4 Hz | 0.33 Hz |

¹ fraction of swipes whose panel had the larger eye drive in the preceding 256 ms (ties do not
count); confounded in closed loop because the swipe itself is the strongest onset, hence the
open-loop test in §5.3.

Seed 1 was chosen for the video because it covers both panels evenly with no long pause; the
dynamics are the same in all six (bursts locked to onsets, swipe chains of 2–3 at 0.4 s spacing,
then habituation until the next autoplay cut ignites a burst). Most seeds still favour the right
panel: whichever panel ignites first gets swiped, swipes are the strongest onsets, and the right
optic lobe of this reconstruction ignites bursts more readily. Video episode: 28 swipes (14 L / 14 R),
20 / 23 posts consumed, 5.1 M spikes, 0.62 Hz mean rate, front-leg MNs fire 60 (L) / 40 (R)
spikes in the 0.5 s after a swipe. Figure: `docs/figures/episode_v2_s1.png`.

## 6. What is and is not learned or fitted

- **Not fitted / not scripted:** the wiring (every synapse count and sign is MaleCNS), the neuron
  model, the retinotopic input map (real hex column coordinates), which cells are driven (L1/L2),
  which cells are read (all descending neurons; medulla per side), the swipe rule.
- **Chosen by hand (and reported):** `weight_scale` 0.5, adaptation 0.6 mV, STD U 0.15 / 480 ms,
  encoder gain 6 and 150 Hz cap, DN threshold 1.5 Hz, side-trace 0.8 s, refractory 0.4 s / 0.3 s
  and 1 s warm-up, autoplay amplitude and rate, the seed shown.
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
R eye orange, DNs green, front-leg MNs magenta; the volume's x axis is mirrored so the fly's left
lobe is on the viewer's left, like the head-up fly) lit frame-accurately by the spike file, a HUD
with the descending-neuron rate per side against its 1.5 Hz threshold, posts consumed L / R, swipes,
spikes/s with a sparkline, title card (0–4 s), stats card (38–46 s) and attribution card (46–50 s).
Feed frames are replayed deterministically from the swipe times and meta line in the events log.

## 8. Reproduce

```bash
cd ~/Projects && git clone https://github.com/BEXAI/DoomFly.git && cd DoomFly
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas pyarrow pillow opencv-python-headless matplotlib tqdm imageio imageio-ffmpeg
python3 src/download.py                                  # ~1.1 GB
python3 src/build_graph.py                               # ~3 min, peaks ~8 GB RAM
python3 src/validate.py --quick                          # Tests A-C -> out/validate_results.json + figures
python3 src/run_episode.py --duration 50 --seed 1 --tag final_v2                      # the episode (burst mode is the default)
python3 src/run_episode.py --duration 50 --tag v2_shuffled --shuffle 0 --no-spikes    # wiring-shuffled control
python3 src/run_episode.py --duration 30 --tag v2_side_real --no-spikes --scripted L:15,R:15 --teacher-gap 1.2 2.0             # open-loop side test
python3 src/run_episode.py --duration 30 --tag v2_side_shuf --no-spikes --scripted L:15,R:15 --teacher-gap 1.2 2.0 --shuffle 0
python3 src/side_eval.py v2_side_real v2_side_shuf
python3 src/analyze_episode.py final_v2 v2_shuffled      # figures + JSON
python3 src/render.py --events out/events_final_v2.jsonl --spikes out/spikes_final_v2.npz \
        --positions data/graph/positions.npy --groups data/graph/groups.npy --out out/final.mp4
# optional, the rejected ridge readout of §5.4:
python3 src/run_episode.py --mode teacher --duration 100 --teacher-gap 1.0 1.8 --no-spikes --out-dir out/calib
python3 src/decoder.py fit out/calib/calib.npz out/calib/readout.npz
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

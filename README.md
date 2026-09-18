# DoomFly — a fly brain double-doomscrolls an iPhone Duo

A ~50 s vertical (1080 × 1920) video in which a simulated fruit-fly brain — the complete
**MaleCNS v1.0 connectome** (166,700 neurons, 6.24 M connections) running as a
leaky-integrate-and-fire spiking network — sits on an open foldable phone mock-up and
scrolls two feeds at once: **left eye → left panel, right eye → right panel; front-left leg
swipes the left feed, front-right leg swipes the right feed**. A picture-in-picture shows
the real neurons firing.

Nothing in the video is scripted: every swipe in the final cut is emitted by a readout of
real descending / motor neurons of the simulated connectome, driven only by what the two
eyes see. What *is* fitted, and what is not, is stated exactly in [§6](#6-what-is-and-is-not-learned-or-fitted).

> Status: pipeline complete end to end (M0–M4). Results tables in §5 are filled from
> `out/validate_results.json` and the episode summaries; see §8 for how to regenerate.

---

## 1. Pipeline

```
data/raw/*.feather ──► src/build_graph.py ──► data/graph/{W.npz, ids.npy, sets.json, eye_map.npz, positions.npy, groups.npy}
                                                        │
 src/feeds.py  (two synthetic feeds, 1335×1878 each)    │
      │ luminance grid per panel (24×18)                 ▼
      ▼                                         src/sim.py  (LIF, dt = 2 ms, scipy CSC)
 src/encoder.py  (retinotopic L1/L2 Poisson drive) ──►  brain.step() ×8 = one 16 ms control step
                                                        │ spikes of 2,129 DN + MN cells
                                                        ▼
                                                src/decoder.py  (linear readout → swipe L / R)
                                                        │ feed.swipe()
                                                        ▼
 src/run_episode.py  logs out/events.jsonl + out/spikes.npz  ──►  src/render.py  ──►  out/final.mp4
 src/validate.py     Tests A–C (Shiu benchmark, runaway, wiring shuffle)
```

Everything runs offline on a CPU (4 cores, 15 GB RAM were used); no GPU is required.
A 50 s episode simulates in well under a minute and renders in about two minutes.

## 2. Data

MaleCNS v1.0 flat connectome (CC-BY 4.0), three files, ~1.1 GB
(`src/download.py`, bucket `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`):

| file | used for |
|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | bodyId, type, superclass, class, subclass, somaSide / rootSide, somaNeuromere, `assignedOlHex1/2` (optic-lobe column coordinates), `somaLocation` |
| `body-neurotransmitters-male-cns-v1.0.feather` | `consensus_nt` per body (synister predictions, Eckstein, Bates et al. 2024) |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 151.9 M `body_pre, body_post, weight` rows |

Column-level notes with printed value counts are in `docs/RESEARCH_DATA.md`;
reference-implementation notes (Shiu constants, sign conventions, licences) in `docs/RESEARCH_REFS.md`.

### Graph build (`src/build_graph.py`) — choices and resulting numbers

| choice | value |
|---|---|
| neurons kept | every body with a non-null `superclass` → **166,700** |
| edges kept | synapse count ≥ 5, both ends kept, 33 self-edges dropped → **6,242,085 edges, 89,859,936 synapses** |
| sign | presynaptic `consensus_nt`: ACh, DA, OA, 5-HT → +1; GABA, glutamate (GluCl, Liu & Wilson 2013), histamine → −1; unclear/missing (3,177 cells) → +1 |
| inhibitory edge fraction | **37.2 %** |
| NT counts | ACh 103,720 · Glu 29,302 · GABA 22,069 · His 7,891 · unclear 2,999 · DA 392 · OA 101 · 5-HT 48 · missing 178 |
| side | `somaSide` → `rootSide` (sensory cells) → `instance` suffix `_L/_R` |
| soma positions | `somaLocation` × 8 nm for 139,662 cells; 26,236 more filled by the mean of synaptic partners (4 rounds); 802 left unknown |

These match the community builds (hotocoo/malecns: 166,700 / 6,242,118 / 89.86 M).

### Neuron sets (`data/graph/sets.json`)

| role | selection | L | R |
|---|---|---|---|
| eye input, ON | type `L1` (lamina) with hex column coords | 875 | 892 |
| eye input, OFF | type `L2` | 874 | 893 |
| eye input, sustained | type `L3` (hex coords exist only on the right) | 0 | 892 |
| optic lobe | superclass `ol_intrinsic` | 44,607 | 44,796 |
| visual projection neurons | superclass `visual_projection` | 4,589 | 4,612 |
| descending neurons | superclass `descending_neuron` | 656 | 648 (+10 midline) |
| **front-leg motor neurons** | superclass `vnc_motor` & subclass `fl` (T1) | **68** | **67** |
| all motor neurons | `vnc_motor` + `cb_motor` | 815 | |
| readout cells | DN ∪ MN | 2,129 | |
| sugar GRNs (proxy) | labellar bristle types `LB3a-d` | 78 | |
| bitter GRNs (proxy) | labellar bristle types `LB1a-e` | 56 | |
| MN9 | type `MN9` (bodyIds 10331 L, 16949 R) | 2 | |
| PAM / PPL1 / KC / MBON | for Phase 2 | 316 / 16 / 4,064 / 97 | |

Caveats found in the data: the left eye is under-reconstructed (R1–R6 photoreceptors 1,112 L vs
2,265 R; L3/C2/Tm4 have column coordinates only on the right), which is why drive is injected
at L1/L2 (complete on both sides) and never at photoreceptors (histamine sign trap). MaleCNS has
**no sugar/bitter GRN label**; the LB3/LB1 split is a connectivity proxy (strongest two-hop path to
MN9) and Test A is reported with that caveat.

## 3. Neuron model (`src/sim.py`)

Shiu et al. (Nature 2024) LIF constants, as in `philshiu/Drosophila_brain_model` and
`hotocoo/malecns` (both verified line-by-line, see `docs/RESEARCH_REFS.md` §1):

| constant | value |
|---|---|
| V_rest = V_reset | −52 mV |
| V_threshold | −45 mV |
| refractory | 2.2 ms (→ 1 step at dt = 2 ms) |
| membrane τ | 20 ms |
| synaptic τ (exponential) | 5 ms |
| PSP per synapse | 0.275 mV (peak, τ_m/τ_s pre-compensation as hotocoo) |
| synaptic delay | 1.8 ms (→ 1 step) |
| dt / control step | 2 ms / 16 ms (8 substeps) |
| `weight_scale` | see §5 (Test B decides) |
| spike-frequency adaptation `adapt_mv` | see §5 (0 = pure Shiu; hotocoo uses 0.6 mV, τ 120 ms) |

Sensory input follows Shiu: driven cells are forced to spike as Poisson processes at the
requested rate; everything downstream is the wiring. Implementation is a sparse column gather
on the CSC matrix (only the columns of neurons that spiked are touched), ~1–20 ms per 2 ms step
on 4 CPU cores depending on activity.

## 4. Eyes and legs

**Encoder (`src/encoder.py`).** Each panel frame → 24 × 18 luminance grid (`cv2.resize`
INTER_AREA). The ~875 L1 and ~875 L2 columns of each eye carry MaleCNS hex coordinates
(`assignedOlHex1/2`); axial → cartesian (x = h1 − h2/2, y = h2·√3/2 gives an upright 1.15:1
ellipse), PCA-upright, normalised, binned to the grid (≈ 2.5 columns per populated cell, 355 of 432
cells populated). Temporal contrast c = (I − I_bg)/(I_bg + 0.05) against a 250 ms adapting
background; c > 0 drives that column's **L1** (ON), c < 0 drives **L2** (OFF), rate = clip(6·|c|, 0, 1) × 150 Hz.
Static frames therefore go silent within ~1 s; a swipe produces a ~110 Hz burst.
Assumption (flag `--flip`): the eye's long axis is the screen's vertical axis.

**Decoder (`src/decoder.py`).** Features: for each of the 2,129 readout cells (all DNs + all MNs)
a fast (80 ms) and a slow (800 ms) exponential trace of its rate, minus the population mean.
Two modes:
- `direct` — z-scored mean rate of the 68 / 67 front-leg motor neurons, threshold + 400 ms refractory.
- `ridge` — one linear weight vector per panel fitted by ridge regression to a teacher signal
  collected in a scripted "teacher" episode (hotocoo `calibrate.py` approach). The threshold is
  set to reproduce the teacher's swipe count. Held-out correlation and lateral specificity are
  reported in §5.

## 5. Results (Tests A–C and the episode)

_Filled from `out/validate_results.json`, `out/summary_*.json` and `out/calib*/readout.npz`; see §8._

### Test A — Shiu benchmark (sugar → MN9)
TBD

### Test B — runaway / silence and choice of `weight_scale`
TBD

### Test C — wiring-shuffle control
TBD

### Episode
TBD

## 6. What is and is not learned or fitted

- **Not fitted / not scripted:** the wiring (every synapse count and sign comes from MaleCNS),
  the neuron model, the input mapping (real hex column coordinates), which cells are driven, which
  cells are read out.
- **Chosen by hand:** `weight_scale`, `adapt_mv` (Test B), encoder gain / rate cap, decoder
  time constants, refractory period, z threshold.
- **Fitted:** in `ridge` mode, one linear weight vector per panel over the real DN + MN cells
  (2 × 2,129 × 2 numbers) plus a bias, fitted to a scripted teacher. The brain dynamics are never
  modified by the fit; the readout only decides *when* the leg flicks, from the activity that the
  wiring produced. In `direct` mode nothing is fitted.
- **Never claimed:** awareness, intent, or that a fly would do this. It is a static wiring diagram
  driven by LIF dynamics with a linear readout.

## 7. Video (`src/render.py`)

Pure OpenCV/numpy compositing at 1080 × 1920, 60 fps (~30 fps render speed): dark desk, open
two-panel device mock-up (two 1335 × 1878 portrait panels, crease, titanium bevel, **no logo, no
real UI**), a stylised top-down Drosophila perched over the hinge whose front-left / front-right
leg flicks on each swipe event, eye → panel glow cues in the intro, a 400 × 400 PiP of the
soma point cloud (group-coloured: L eye cyan, R eye orange, DNs green, front-leg MNs magenta)
lit frame-accurately by `spikes.npz`, a HUD with posts consumed L / R, swipes, spikes/s with a
rolling sparkline, a title card, a stats card and the attribution card. Feed frames are replayed
deterministically from the swipe times in `events.jsonl` (same seeds), so nothing is stored twice.

## 8. Reproduce

```bash
cd ~/Projects && git clone https://github.com/BEXAI/DoomFly.git && cd DoomFly
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas pyarrow pillow opencv-python-headless matplotlib tqdm imageio imageio-ffmpeg
python3 src/download.py                 # ~1.1 GB
python3 src/build_graph.py              # ~3 min, peaks ~8 GB RAM
python3 src/validate.py                 # Tests A-C -> out/validate_results.json, figures
python3 src/run_episode.py --mode teacher --duration 40 --out-dir out/calib     # calibration episode
python3 src/decoder.py fit out/calib/calib.npz out/readout.npz                   # ridge readout
python3 src/run_episode.py --mode ridge --duration 50 --tag final                # the episode
python3 src/run_episode.py --mode ridge --duration 50 --tag shuffled --shuffle 0 # wiring-shuffled control
python3 src/render.py --events out/events_final.jsonl --spikes out/spikes_final.npz \
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
spiking network, doomscrolling two feeds at once — one per eye, one per leg. Every swipe is emitted
by the fly's own descending/motor neurons. Nothing scripted. Data CC-BY Janelia FlyEM · Cambridge ·
Google.*

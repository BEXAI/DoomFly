# Reference-repo research notes (MaleCNS v1.0 LIF simulation)

All paths are relative to `/home/user/DoomFly/ref/`. Line numbers are from the checked-out
copies (read 2026-09-18). Four repos: `Drosophila_brain_model` (philshiu, Brian2, FlyWire
630), `malecns` (hotocoo, torch/Metal, MaleCNS), `flybrain` (TheMrRaGe, numpy, MaleCNS),
`connectome-lab` (avisharm, graph-ML, MaleCNS).

---

## 1. LIF constants: philshiu vs hotocoo (vs flybrain)

| constant | philshiu (Brian2) | hotocoo `malecns` | flybrain `flysim.py` |
|---|---|---|---|
| v_rest | -52 mV `model.py:22` | -52 `src/brain.py:40` | 0 (relative; "~-52") `scripts/flysim.py:55` |
| v_threshold | -45 mV `model.py:24` | -45 `brain.py:42` | +7 mV rel. `flysim.py:55` |
| v_reset | -52 mV `model.py:23` | -52 `brain.py:41` | 0 rel. `flysim.py:56` |
| refractory | 2.2 ms `model.py:31` | 2.2 ms `brain.py:43` -> `ref_steps = max(1, round(2.2/dt))` = **1 step at dt=2** `brain.py:173` | 2.2 ms `flysim.py:58` |
| membrane tau | 20 ms `model.py:25` | 20 ms `brain.py:44` | 20 ms `flysim.py:57` |
| synaptic tau (alpha syn) | 5 ms `model.py:28`, `dg/dt=-g/tau` `model.py:46` | 5 ms `brain.py:45` | 5 ms `flysim.py:61` |
| mV per synapse (w_syn) | 0.275 mV `model.py:37`, `syn.w = signed_count * w_syn` `model.py:183` | 0.275 `brain.py:46` | 0.275 `flysim.py:59` |
| synaptic delay | 1.8 ms `model.py:34`, `Synapses(..., delay=t_dly)` `model.py:175` | 1.8 ms `brain.py:47` -> `delay_steps = max(1, round(1.8/dt))` = **1 step (2 ms)** `brain.py:174` | 1.8 ms `flysim.py:62`, `n_dly = max(1, round(1.8/dt))` `flysim.py:713` |
| dt | Brian2 default (not set in code; `method='linear'` `model.py:165`); flybrain notes 0.1 ms in paper `FINDINGS.md:50` | `LIFConfig.dt_ms=1.0` `brain.py:48` but every entry point uses `defaults.DT_MS = 2.0` `src/defaults.py:9` (e.g. `src/calibrate.py:219`) | 1.0 ms `flysim.py:72` |
| substeps / control step | n/a (1000 ms trials x 30 runs `model.py:17-18`) | `SUBSTEPS = 8` -> 16 ms, 62.5 Hz `defaults.py:10-11` | body tick 150 ms `fly3d.py:370`; flybox `TICK_MS` |
| weight scale | none (1.0) | `WEIGHT_SCALE = 0.15` `defaults.py:17`, multiplied into W `brain.py:146`; "below ~0.25 the network stays out of runaway" `README.md:274` | `gain = 1.0` `flysim.py:63` (was 0.15 before alpha synapse, `flysim.py:64-70`) |
| adaptation | none | `ADAPT_MV = 0.6` `defaults.py:18`, `adapt_mv` `brain.py:52`, `tau_adapt_ms = 120` `brain.py:53`; "0 reproduces the paper model" `README.md:275` | none (Tsodyks-Markram STD optional: `std_u 0.08`, `std_tau_rec_ms 480` `flysim.py:190-191`) |
| noise | none | none | 0.15 mV/step Gaussian `flysim.py:71` |
| Poisson input | `r_poi = 150 Hz` `model.py:39`, `r_poi2 = 0` `:40`, `f_poi = 250` -> weight `w_syn*f_poi` = 68.75 mV per event `model.py:41,90` (each event forces a spike); target refractory set to 0 `model.py:92`; notebook sweeps 10-200 Hz `figures.ipynb` cell 4 | kick `kick_mv = 8.0` mV `src/agent.py:61` (> 7 mV threshold); rate soft-saturates at `max_input_hz = 300` `agent.py:62,659`; prob = rate*dt/1000 `agent.py:690`; common random numbers across bodies `agent.py:63` | receptors are pure Poisson spike sources at `drive_hz`, ceiling `max_rate_hz = 200` `flysim.py:159,899-914` |

Hotocoo's documented parameter block: `brain.py:10-12`, `docs/RESEARCH.md:76-82` ("basal firing = 0,
gap junctions excluded").

### Synapse model / integration differences (important)

- **philshiu**: `dv/dt = (v_0 - v + g)/t_mbr`, `dg/dt = -g/tau`, both `(unless refractory)`
  `model.py:45-46`; `on_pre='g += w'` `model.py:175`; reset `v = v_rst; g = 0*mV` `model.py:52`
  (g is zeroed on spike, and v and g are frozen during refractory).
- **hotocoo**: exact exponential decays `decay_v = exp(-dt/tau_m)`, `decay_s = exp(-dt/tau_s)`
  `brain.py:157-158`; `syn_kick = (dt/tau_m) * w_syn * (tau_m/tau_s)` `brain.py:164-169` — the
  `tau_m/tau_s = 4x` pre-compensation makes **one synapse peak at ~0.275 mV of PSP** (RESEARCH.md:94-98),
  whereas in Shiu's alpha synapse a single 0.275 mV jump in g yields only ~0.25 x 0.275 mV of peak PSP
  (flybrain `flysim.py:65-68`). With `WEIGHT_SCALE 0.15` the net per-synapse effect is ~0.6x Shiu's.
  Step order `brain.py:538-573`: j += syn_kick*W@delayed_spikes, j *= decay_s; inflow = j -
  adapt_kick*adapt (+ external); inflow masked to 0 if refractory (**membrane still leaks; j not
  frozen or reset**); u = decay_v*u + inflow; fire at u >= 7; u -> 0; adapt = decay_a*adapt + spike.
  Refractory with one step: "previous spike bit is the flag" `brain.py:225-227`, `src/metal_lif.py:18-20`.
- **flybrain**: `g += W` on arrival, `g -= g*dt/tau_syn`, `syn = g*dt/tau_m`, `dv = -v/tau_m*dt + syn +
  ext*dt/tau_m + noise` `flysim.py:854-890`; v updated only if not refractory `:892-894`; v floored at
  `-v_thresh` `:895`; g neither frozen nor reset. Verdict text: "alpha-synapse dynamics and the 1.8 ms
  delay are not optional ... instantaneous jump is ~4x too much" `FINDINGS.md:54-57`.
- Runaway: hotocoo measured that at literature weights the full graph latches into 30-100 Hz
  self-sustained activity, insensitive to input; fix = `weight_scale 0.15 + adapt_mv 0.6` giving
  0.6-4.6 Hz global, 2-10 Hz DN rates `docs/RESEARCH.md:106-112`. flybrain runs at gain 1.0 with 0 Hz
  rest and reports 590 of 162,517 neurons spiking per ms (0.36%) `FINDINGS.md:109`.

---

## 2. Neurotransmitter -> sign convention

| NT | philshiu | hotocoo | flybrain |
|---|---|---|---|
| acetylcholine | + | +1 | +1 |
| GABA | - | -1 | -1 |
| glutamate | - | -1 | -1 |
| dopamine / 5-HT / octopamine | + (Shiu convention, quoted at `malecns/src/build_graph.py:8-11`) | +1 | **0** (dropped from fast matrix; DAN edges used separately for plasticity `flysim.py:469-472`) |
| histamine | (FlyWire predictions had no histamine class) | -1 (`build_graph.py:24-27`, Hardie 1989) | -1 |
| unknown / unclear / NaN | n/a (sign pre-baked in parquet) | **+1** ("unknown -> excitatory prior", flagged `nt_known=False`) | **0** (silent) |
| magnitude | `syn.w = df_con['Excitatory x Connectivity'] * w_syn` (signed synapse count x 0.275 mV) `model.py:183` | `signed = weight * sign[pre]` (synapse count, float32) `build_graph.py:141` | `data = sign[pre] * w * mv_per_synapse * gain`, zero-weight edges dropped `flysim.py:294,305-308` |

Quoted code:

```python
# malecns/src/build_graph.py:26-27, 63-69
EXCITATORY = {"acetylcholine", "dopamine", "serotonin", "octopamine"}
INHIBITORY = {"gaba", "glutamate", "histamine"}
consensus = neurons["consensus_nt"].fillna("unknown").str.lower()
sign = np.where(consensus.isin(INHIBITORY), -1.0, 1.0)  # unknown -> excitatory prior
neurons["nt_known"] = consensus.isin(EXCITATORY | INHIBITORY)
```
```python
# flybrain/scripts/build_graph.py:21-22  (identical in build_creature.py:28-29)
SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1,
        "dopamine": 0, "octopamine": 0, "serotonin": 0, "unclear": 0, "unknown": 0}
```

NT feather columns read:
- hotocoo: `["body", "consensus_nt", "predicted_nt_confidence"]`, `body` renamed to `bodyId`
  `build_graph.py:57-60`. **`predicted_nt_confidence` is read but never thresholded** (grep: only
  line 59).
- flybrain: `["body", "consensus_nt"]`, `drop_duplicates("body")`, reindex to node table, NaN ->
  "unknown" `build_graph.py:56-59`. No confidence threshold.
- flybrain per-type override: `sign_override = (("lLN1", -1), ("lLN2", -1))` `flysim.py:97-113`
  (antennal-lobe LNs re-signed inhibitory regardless of table; decision 13).
- Only the `minconf-0.5` feather variants are used (file names `flybrain/README.md:17-19`,
  `malecns/src/download.py:20-30`); the 0.5 refers to synapse-prediction confidence in the release,
  not a per-neuron filter in either repo.
- Shiu et al. on the glutamate question (paraphrased by flybrain): a brainwide flip to excitatory
  raised optogenetic false positives from 1% to 16% `FINDINGS.md:1079-1084`.

---

## 3. Graph build

### hotocoo `malecns/src/build_graph.py`
- Inputs (renamed by `src/download.py:20-30`): `body-annotations.feather` <-
  `body-annotations-male-cns-v1.0-minconf-0.5.feather`; `body-neurotransmitters.feather`;
  `connectome-weights.feather` <- `connectome-weights-male-cns-v1.0-minconf-0.5.feather`.
- Annotation columns: `bodyId, superclass, class, type, instance, somaSide, assignedOlHex1,
  assignedOlHex2` `:44-53`. Filter: **only `superclass.notna()`** `:55` (no `status`/`type` filter).
- Weights columns: `weight`, `body_pre`, `body_post` `:95,99,100`. Filter: `weight >= --min-weight`
  (default 1 `:130`; README/RESEARCH builds use `--min-weight 5 --out data/graph_w5` `README.md:27`),
  both endpoints must be in the neuron table `:101`.
- Outputs `:144-149`: `neurons.parquet` (with `nt`, `sign`, `nt_known`), `edges.npz` (`pre`, `post`
  int32; `weight` = signed float32), `roles.json`.
- Roles by `superclass` `:31-38`: photoreceptor=`ol_sensory`, visual_projection=`visual_projection`,
  descending=`descending_neuron`, motor=`vnc_motor`+`cb_motor`, ascending=`ascending_neuron`+
  `sensory_ascending`, mechanosensory=`vnc_sensory`.
- Numbers (`docs/RESEARCH.md:20-35`, `README.md:9`):

| | full (>=1) | pruned (>=5) |
|---|---|---|
| neurons | 166,700 | 166,700 |
| directed connections | 25,582,938 | **6,242,118** |
| synapses | 124,177,616 | 89,860,280 |

Roles: photoreceptor 6,098; visual_projection 9,201; descending 1,314; motor 815; ascending 2,383;
mechanosensory 6,370 `RESEARCH.md:30-35`. NT mix: ACh 62.2%, Glu 17.6%, GABA 13.2%, His 4.7%,
DA 0.2%, OA 0.1% `RESEARCH.md:37-39`.

### flybrain `scripts/build_graph.py` / `build_creature.py`
- Annotation columns `bodyId, type, superclass, status` (+ `class, somaSide, rootSide` in
  build_creature `:86-87`). Filter `status == "Traced" & type.notna()` `build_graph.py:41`,
  sorted by bodyId `:51`. Optional `--drop-optic-lobe` drops `superclass.startswith("ol_")` `:45-49`.
- Weights: `weight >= 5` default `:31`, columns `weight, body_pre, body_post` `:74,79,80`.
- Output npz: `bodyId, type, superclass, nt, sign(int8), pre, post, w(int32)` `:92-94`;
  build_creature adds `cls`, `side` (L/R/M) `build_creature.py:147-155`.
- Result: raw 151,856,684 rows -> **162,517 neurons / 6,138,378 connections** `FINDINGS.md:17,21-22`;
  58% of neurons are optic lobe `:22`.
- Creature subset selectors `build_creature.py:32-49` (KEEP_CLASS: olfactory, gustatory, thermo-,
  hygro-, mechanosensory, ALPN, ALLN, ALIN, ALON, SEZPN, Kenyon_Cell, MBON, DAN, CX; KEEP_SUPERCLASS:
  descending_neuron, cb_motor, ascending_neuron, cb_endocrine, cb_intrinsic).

### connectome-lab `src/connectome_lab/flybrain_fm.py`
Same three files; columns `body_pre, body_post, weight` `:113`; `status == traced` `:61-63`;
`min_synapses` default 5 `:88`; edge weights `log1p(weight)` `:139`.

---

## 4. hotocoo `calibrate.py`: ridge-regression readout + DAgger

Purpose (`src/calibrate.py:1-26`): ES over a random 144-parameter readout failed for 795
generations; instead fit the two readout directions directly by imitation.

**Readout population.** `readout_roles = "descending,motor"` `src/agent.py:105` -> the 1,314 DNs
(capped by `max_dn`, `agent.py:218`) plus 815 `vnc_motor`+`cb_motor` cells, deduplicated
`agent.py:225-239` = **2,129 cells** (`calibrate.py:7`, `README.md:159`).

**Features per control step.** `brain.dn_acc` counts spikes of readout cells over the 8 substeps
(16 ms); `rate = dn_acc / substeps`; `motor_state = (1-motor_tau)*motor_state + motor_tau*rate`
with `motor_tau = 0.2` (~80 ms low-pass) `agent.py:68-70,708-709`. Calibration records
`signal = motor_state - motor_state.mean(dim=1)` (common mode removed) per alive car, stored fp16
`calibrate.py:110-114`.

**Targets.** Teacher = `LinearTeacher` (`src/teacher.py`) on lidar proximity + speed
`calibrate.py:85-87`; labels = teacher's (steer, pedal) in [-1,1], optionally EMA-smoothed
(`--target-smooth`) `:106`; regression target `y = atanh(clamp(label, +-0.9))` `:47,266`
(pre-tanh space).

**Ridge** `calibrate.py:50-57`: centre x and y; `W = solve(Xc^T Xc + lam*I, Xc^T Yc)`;
`b = mean_y - mean_x @ W` (intercept unpenalised). Lambda picked by held-out R^2 on odd-indexed cars
(train even / test odd) from `--lams 3,10,30,100,300,1000` `:64-75,197`, then refit on all data.

**Install** `calibrate.py:148-185`: random projection `(n_readout, 64)/sqrt(n_readout)` seeded by
`projection_seed`; columns 0/1 replaced by `W/||W||`; channel mean/std computed on training data
and set via `agent.set_readout(projection, mean, std)` (`readout_norm="channel"` z-scores
channels, `agent.py:575-578`); `g_out = ||W_k|| * std_k * boost`, `b_out = ||W_k||*mean_k + b_k`,
`w_out` = identity block on channels 0/1. Agent output = `tanh(g_out * <channels, w_hat> + b_out)`
`agent.py:715-717`; steering positive = left, pedal positive = throttle `README.md:171-173`.

**DAgger loop** `calibrate.py:258-272`, defaults `--cars 12 --steps 4000 --rounds 3 --beta 0.5`
`:192-195`:
```
for round in 0..rounds:
    beta = 1.0 if round == 0 else (0.5 if round < rounds else 0.0)
    run `steps` control steps; action = beta*teacher + (1-beta)*student   (:108)
    record (signal, teacher label, car id) each step; pool with previous rounds
    fit ridge on the pooled data; install; continue with the new theta
final: closed-loop eval `--eval-steps 6000` with beta 0 (:274), write ES checkpoint (:281-316)
```
Sensory params fixed during calibration: `ray_gain 1.0, bias_hz 0.05, speed_gain 0.5,
loom_gain 0.5` `:46`. Brain built with `LIFConfig(dt_ms=2.0, adapt_mv=0.6)`, `weight_scale 0.15`
`:216-223`. Result claimed: laps full-scale Monaco from every start, then ES refines `README.md:163-164`.

---

## 5. Soma positions and the spiking point cloud

**`src/build_positions.py`** — no neuprint. Reads annotation columns `["bodyId", "somaLocation"]`
`:36-38` (voxel xyz; `VOXEL_NM = 8.0` `:32`), keeps rows with `somaLocation.notna()` `:39`. ~16%
of cells lack it (optic-lobe cells annotated only by hex, fragments) `:12-13`; those are placed at
the mean position of their located synaptic partners, iterated `--rounds 4` (`infer_missing`,
`:51-75, :82`), remainder at global mean `:73-74`. Then `pos *= 8 nm`, centred on measured mean,
divided by max |coord| -> unit frame `:99-103`. Writes `positions.npy (n,3) float32` and
`positions_known.npy (n,) bool` `:105-106`.

**Server side** (`src/viewer.py`): loads both `:352-361`; `role_ids` uint8 per neuron, 255 = no
role `:363-367`; endpoints `/api/positions.bin` (float32 xyz) and roles.bin = `role_ids.tobytes() +
position_known.astype(uint8).tobytes()` `:1587-1597`. Each control step's spike vector is sent as
`base64(np.packbits(spiked))` `:573`. `BrainViewConfig`: `activation_decay 0.82`, `inferred_dim 0.45`,
`point_scale 1.6`, yaw 0.6 / pitch 0.15 / distance 1.75 `src/viewer_config.py:154-167`.

**Client** (`web/brain.js`, raw WebGL2 `gl.POINTS`, no library `:3-11`): `load(positions, roleIds,
measured)` uploads static buffers `:186-213`; `applySpikes(mask)` walks the packbits mask
(MSB-first per byte) and sets `activation[i] = 1.0` on a spike else `*= decay` `:215-224`; per frame
only the activation buffer is re-uploaded `:249-250`. Vertex shader: colour = role colour mixed
toward white by activation `:34`, alpha 0.16 -> 1.0 with activation and dimmed by `inferredDim` for
unmeasured cells `:35`, point size `pointScale*depth*(1 + 2.2*activation)` `:37`. `web/app.js`
wires it: fetch positions/roles `:1342-1346`, `applySpikes(decodeMask(frame.mask))` `:1406`.

---

## 6. flybrain `FINDINGS.md` — every trap / lesson (one line each)

| line | lesson |
|---|---|
| 8-9 | MaleCNS v1.0 is CC-BY 4.0; commercial game use permitted with attribution. |
| 26-27 | **Glutamate is inhibitory** (GluCl-alpha); inverting silently gives a dead or seizing network. |
| 28-32 | **Sensory neurons have `somaSide == "M"`** (all 2,635 olfactory, 1,416 gustatory, 4,078/4,107 visual); use `rootSide` for laterality or the creature is spatially blind. |
| 33-35 | Right hemisphere is more completely traced (9.7% more weight on 1.7% more neurons); every L/R readout inherits the bias. |
| 36-38 | Odours must be defined over the 53 receptor *types* (glomerular), not random receptors. |
| 39-40 | `~python_bool` is -1, not False: silently turns a numpy mask into fancy indexing. |
| 41-42, 97-100 | Whole-DN-population steering readout is **sign-inverted**; use the DNa family (d'=4.21); DNa02 alone fires ~1 spike/trial. |
| 54-57 | Alpha synapse + 1.8 ms delay are not optional; instantaneous jumps are ~4x too strong and force a bogus gain fudge. |
| 61-64 | Basal firing is 0 Hz by design; absolute rates are meaningless, always baseline-subtract. |
| 215-231 | **Vision delivered exactly zero**: all 4,107 photoreceptors histaminergic, 29,469 out-edges sign -1; against 0 Hz rest there is nothing to inhibit. |
| 338-353 | Fix: photoreceptors fire a tonic 90 Hz in the dark, `see(brightness)` lowers the rate; each of 14,311 targets gets tonic current so it rests at 0.72 of threshold (`lamina_hold_frac`); hold is exact `ext = v_hold - W*r*tau_syn`. |
| 233-252, 368-389 | Escape sensitised +52% over 20 pulses (no adaptation in model); Tsodyks-Markram STD `U=0.08, tau 480 ms` gives habituation; **STD must be global** (afferent-only made it worse). |
| 401-421 | `stimulate()` caps tonic drive at 12.6 mV, which odour-evoked inhibition silences; use explicit mV (70 mV) for DAN punishment. Identical results from opposite conditions = condition never applied. |
| 453-458 | Motor pool reads saturate at +-1 per tick (781 spikes over 699 MNs in 400 ms); integrate longer or drive harder before claiming the connectome walks. |
| 517-534 | Embodied fly walked in circles: (1) splitting receptors by array index is 68-71% impure; (2) structural bias (-1.30) is 16x the side signal (0.078). |
| 536-553 | Calibrate the zero point from a *symmetric* stimulus at several drive levels; even then continuous turn integration gives circles, so treat asymmetry as evidence for an (imposed) saccade generator. |
| 589-597 | Mirroring the connectome by (pre type, post type, crossing) brought R/L weight from 1.0968 to 1.0001; KCs excluded (random wiring). |
| 653-659 | APL surrogate compared per-ms KC fraction to a per-odour 5% target and never engaged: every `apl` argument had been a no-op. |
| 661-665 | KC->KC recurrence (55% of PN input) is *not* the cause of dense KC codes (ablation changed nothing). |
| 667-684 | **lLN1/lLN2 mislabelled** (ACh / 50-50 split): one glomerulus drove 45/53; re-signing 151 cells inhibitory gives 1/53 and zero shared KCs. |
| 686-705 | Real APL is in the graph and 7x too strong as a spiking LIF; `apl_scale 0.1` + `kc_thresh 1.5` gives ~5% KC sparsity, below-chance overlap. |
| 894-912 | MBONs need tonic drive (`mbon_hold_frac 0.85` -> 3.4 Hz rest); MBON18/21 stay silent regardless. |
| 1001-1012 | Every layer past the first synapse is dark under 0 Hz rest; fixing one layer at a time is whack-a-mole. |
| 1022-1036 | Global background hold (`bg_hold_frac`) refuted: at 0.7 spontaneous activity swamps the learned signal. |
| 1038-1043 | Global STD is incompatible with a sustained odour (afferents settle to 11%); habituation and olfactory results were measured on different engines. |
| 1051-1058 | Convergence layer is net-inhibited 1.5-5x during odour: the E/I balance under uniform 0.275 mV/synapse + label signs is the regime Shiu et al. warned about. |
| 1079-1089 | Global glutamate flip ruled out by Shiu et al.'s own experiment. |
| 1091-1112 | Synapse-count saturation (`syn_sat_k`) monotonically worse at every k. |
| 1183-1184, 1222-1225 | Do not saturate the network: 200 Hz on every receptor pins PNs at 204 Hz (refractory ceiling); at strength 0.35 (PN ~130 Hz) learning reaches the DNs. |
| 1253-1260, 1350-1361 | Cells with `consensus_nt == "unclear"` (DNd02; IPC, DH44, LK, NPFL1, Hugin peptidergic cells) have sign 0 and no output; model hunger through its *targets* (PPL101 gating, sugar-GRN gain). |
| 1330-1348 | Gustatory types are named by body part (LB, LgLG/LgAG, WG, PhG), not modality; SEZ is not modality-selective (aversive LB3d drives MN9 harder than sugar LB3c). |
| 1409-1418 | Water (ppk28, LB3a) -> proboscis is dark: second-order GNG229/GNG175 are labelled GABAergic. |
| flysim.py:372-397 | Per-KC threshold normalisation by PN input only (decision 11). |
| flysim.py:412-427 | Two antennae must be *matched* per receptor type (1,343 R vs 883 L receptors). |

---

## 7. Descending-neuron -> movement map, laterality, leg MN names

**DN mapping actually used in code**

| DN | movement | where |
|---|---|---|
| DNa family (prefix `DNa`), split by side | steering: `T = (R-L)/(R+L)`; DNa02 drives an ipsilateral turn so T>0 = right turn | `learned_steering.py:12-16,50-54`; `fly3d.py:70-72`; `desktop_fly.py:133-138`; `flybox.py:166`; `verge_swarm.py:230` |
| DNa02 alone | "best-characterised turning neuron", 1/side, ~1 spike/trial -> too sparse | `FINDINGS.md:79-80,94-100`; `flyworld.py:176`; `desktop_fly.py:128-132` |
| DNp01 (Giant Fiber) | escape takeoff / burst away from threat | `fly3d.py:22,75,206-208`; `desktop_fly.py:139`; `flyworld.py:187-188`; `verge_swarm.py:32,233` |
| `DNp*` family | escape habituation readout | `habituation.py:162` |
| total DN rate | walking drive / ground speed | `fly3d.py:21`, `desktop_fly.py:201` (`speed = min(1, tot/40)`) |
| `CB*` + `DNg*` (unnamed) | "grab", novel-effector channel | `flyworld.py:190-195` |
| DNd02, DNge069, DNbe007, DNg33, DNp56 | carry *learned* valence; DNge069 (Glu) suppresses T2 trochanter flexor & primes TTMn jump muscle; DNbe007 hind-leg + wing steering; DNg33 abdominal + wing power (DLMn/DVMn) | `FINDINGS.md:1236,1247-1263` |
| DNa02 direct targets | coxa rotators in all three legs (+92/+87/+64) | `FINDINGS.md:1262-1263` |
| DNg13 (steer), DNp10 (jump) | **not used in flybrain code**; cited only as DesktopFly-Linux / Fly64 mappings that agree with DNa02/DNp01 | `README.md:83-84`, `FINDINGS.md:569-571` |

**Left/right identification**: `side = somaSide if in {L,R} else rootSide, else "M"`
`build_creature.py:51-65`; readouts index `b.side[dn] == "L"/"R"`; fallback to whole-DN halves if a
family has <2 cells per side `fly3d.py:73-74`, `desktop_fly.py:129-132`; mechanosensory L/R also by
side with an array-halves fallback `desktop_fly.py:141-146`. Bias calibration: measure R-share
under symmetric drive at levels 0/0.25/0.55/1.0 x 180 Hz, interpolate by total drive, evidence =
`(R/(R+L) - ref - slow_baseline) * 12`, clipped +-1 `desktop_fly.py:163-200`, `fly3d.py:96-131`.
Hemisphere weight rebalance at the matrix: `w *= inL/inR` for right-side posts `flysim.py:273-291`.

**Leg motor neurons in MaleCNS (`scripts/motormap.py`)**: `superclass == "vnc_motor"` `:122`
(699 in the sim, 708 in annotations `:13`); segment from annotation column `somaNeuromere`
(T1 = prothoracic = front legs, exitNerve `ProLN`; T2 mid legs + wings; T3 hind) `:13-20,100-102`;
side from `side`. Antagonist `type` names `:65-82`:

| joint | flexor-side types | extensor-side types | T1L pool (flex/ext) | T1R |
|---|---|---|---|---|
| coxa | `Sternal anterior rotator MN`, `Tergopleural/Pleural promotor MN` | `Sternal posterior rotator MN`, `Pleural remotor/abductor MN` | 6/6 | 6/4 |
| trochanter | `Tr flexor MN`, `Acc. tr flexor MN` | `Tr extensor MN`, `Sternotrochanter MN`, `Tergotr. MN` | 11/8 | 10/8 |
| tibia | `Ti flexor MN`, `Acc. ti flexor MN` | `Ti extensor MN` | 15/2 | 14/2 |
| tarsus | `ltm MN`, `ltm1-tibia MN`, `ltm2-femur MN` | `Ta depressor MN` | 8/5 | 7/4 |

(pool sizes from `results/motormap.json`). Wings: `DLMn*` downstroke (5/side), `DVMn*` upstroke
(7/side), steering prefixes `MNwm, b1-b3, hg, iii, ps1, tp1, tp2, i1, i2, tt` (16/side) `:84-87`;
halteres `MNhm*` `flybox.py:174-175`; other MN names seen: `TTMn` (jump), `MNhl62`, `MNad03/22/25/42`
(abdominal) `FINDINGS.md:1249-1251,1278-1282`. Joint read = `(flexor - extensor)/(flexor + extensor)`
`:39-46,151-156`. Proboscis MNs (`MN9, MN10, MN11D, MN11V, MN12D, MNx01`) are in `cb_motor`, not
`vnc_motor` `flysim.py:806`, `flybox.py:178-179`. hotocoo's "motor" role = `vnc_motor`+`cb_motor`
= 815 cells `malecns/docs/RESEARCH.md:33`.

---

## 8. Fly body drawing (fly3d / desktop_fly / flyroom) and DesktopFly licence

- **Where geometry lives**: `fly3d.py` only simulates the body state (position, yaw/pitch/roll,
  mode, `gait` phase 0-1, `wing`, `groom`) and writes JSONL frames `fly3d.py:430-436,443-451`; the
  3D mesh is in `web/flyroom.html`. `desktop_fly.py` draws a 2D top-down Tk canvas.
- **Primitives (flyroom.html, three.js)**: scaled `SphereGeometry` ellipsoids for thorax
  (r 4.6), abdomen (r 5.0, banded canvas texture), head (r 3.0), two eyes (r 2.0) `:194-202`;
  `CylinderGeometry` antennae (0.16 x 2.2) and proboscis (0.22->0.6 x 2.4) `:204-208`; legs = three
  `CylinderGeometry` bones (femur/tibia/tarsus, radii 0.48/0.38/0.24) in nested root->knee->ankle
  groups `:222-236`; wings `CircleGeometry` `:238-240`. Frame: +Y forward, +Z up, rotated -90 deg
  about X `:177-179`. Body ~23 units nose to tail, viewer scale 0.115 `fly3d.py:47-48`.
- **Leg table** (side, attach xyz, yaw offset, gait phase, isFront, femur, tibia, tarsus)
  `flyroom.html:210-217` = `desktop_fly.py:81-88`: front (+-3.1, 5.3) yaw 0.95 phases 0.0/0.5,
  segments 4.2/4.8/3.2; mid (+-3.7, 2.0) yaw -0.10 phases 0.5/0.0, 4.8/5.6/3.8; hind (+-3.3, -1.2)
  yaw -0.95 phases 0.0/0.5, 5.8/7.0/4.6. Alternating 0/0.5 phases per side = tripod.
- **Gait**: `p = (gait + phase) % 1`; stance for `p < 0.6`: `ang = amp*(1 - 2p/0.6)`, lift 0; swing:
  `ang = amp*(2s-1)`, `lift = sin(pi*s)*0.42` `flyroom.html:416-419`; `amp = 0.20 + 0.30*min(1,
  drive/6)` `:403`; `STANCE_FRACTION 0.6`, `GAIT_FREQ (3, 11) Hz`, `GAIT_AMP (0.20, 0.50) rad`
  `desktop_fly.py:94-96`; gait frequency `f = 3 + 8*min(1, speed/14)` `fly3d.py:321`; front legs
  groom over the head in "groom" mode `flyroom.html:405-410`, `desktop_fly.py:383-398`; legs tucked in
  flight `flyroom.html:412-415`. Modes `stop/walk/groom/takeoff/flight/land` `fly3d.py:50`, random
  takeoff 0.26 `:54`; saccade generator `:231-256`.
- **Licence of the ported geometry**: "ported from DesktopFly-Linux (MIT, (c) 2026 Denis Shiryaev
  and contributors)" `desktop_fly.py:15-17,79`, `flyroom.html:176-177`, `README.md:81-82`,
  `FINDINGS.md:567-568`. No copy of the DesktopFly LICENSE text is in the repo.

---

## 9. Licences of the four repos

| repo | LICENSE file | terms | attribution needed to reuse code |
|---|---|---|---|
| `Drosophila_brain_model` | `LICENSE:1-3` **MIT**, "Copyright (c) 2023 Philip Shiu and Nico Spiller" | MIT | keep copyright + permission notice (`LICENSE:12-13`); cite Shiu et al. 2024 for parameters |
| `connectome-lab` | `LICENSE:1-3` **MIT**, "Copyright (c) 2026 Avin Sharma"; `README.md:200-201`; `CITATION.cff:125` | MIT | keep notice; CITATION.cff asks to cite the repo |
| `malecns` (hotocoo) | **none** (`ls` shows no LICENSE/COPYING) | `README.md:319-324` covers *data* only (CC-BY 4.0); code is all-rights-reserved by default (sole author `hotocoo`, git log) | ask the author before copying code; re-implement from the description otherwise |
| `flybrain` (TheMrRaGe) | **none** | `README.md:86-88` authorship "Mr. Moon (TheMrRaGe)" + Claude; `FINDINGS.md:8-9` data CC-BY 4.0; ported geometry MIT (Shiryaev) | same as above: no code licence granted; credit DesktopFly-Linux (MIT) for the leg table if reused |

Data: MaleCNS v1.0 CC-BY 4.0 (HHMI Janelia FlyEM + Cambridge + MRC LMB + Google Research)
`malecns/README.md:321-322`, `flybrain/README.md:49-51`, `connectome-lab/docs/MALECNS.md:84-86`.
Viewer assets in hotocoo: three.js r170 MIT, Ferrari model CC-BY, Poly Haven CC0, OSM ODbL,
EU-DEM `malecns/README.md:289-317`.

---

## 10. "Sugar -> MN9" benchmark

**philshiu (FlyWire 630)**: 21 right labellar sugar GRNs by FlyWire id `example.ipynb` cell 3
(= `figures.ipynb` cell 4, "labellar sugar-sensing gustatory receptor neurons on right hemisphere");
MN9 id `720575940660219265` (`example.ipynb` cell 17; `figures.ipynb` cell 8; annotated "left" in
cell 23). Stimulation: `r_poi` default **150 Hz** `model.py:39` (notebook text says "By default,
the neurons are excited at 200 Hz", cell 6 — the code disagrees); example run at 100 Hz (cell 13);
Fig 1D sweep 10-200 Hz in 10 Hz steps (cell 4-5); Fig 3A sugar x bitter 0-200 Hz grid on MN9
(cells 23-25), sugar x Ir94e likewise (cell 27). Trials 1000 ms x 30 `model.py:17-18`. Expected:
">400 000 spikes" total and "only about 400 neurons show activity" (cells 8, 10); MN9 rate at
100 Hz sugar reported (cell 17) and reduced by silencing top-3 upstream neurons (cells 18-19). No
numeric MN9 Hz is stored in the repo (raw results are external, `Readme.md:12-13`).

**flybrain (MaleCNS)**: feeding readout `FEEDING_MN = ("MN9", "MN10", "MN11D", "MN11V", "MN12D",
"MNx01")` `flysim.py:806` ("the pathway Shiu et al. validated" `:790`), selected from
`pop["motor"]` = `cb_motor` `flybox.py:178-179`. Sweet GRN types `TASTE_SWEET = ("LB3c", "LB3b",
"dorsal_tpGRN", "claw_tpGRN")` (LB3b-c = Gr64f sugar; taste pegs = Gr5a/Gr64e) `flysim.py:794-803`;
bitter `LB1a-d` (Gr33a), water `LB3a` (ppk28) `:804-805`. Drive: `drive_hz = max_rate_hz(200) *
|quality|` `:816`. Measured: LB3c alone 564 feeding-MN spikes/600 ms (aversive LB3d 1,633, bitter 0,
water 41) `FINDINGS.md:1341-1343`; full sugar set with no mechano confound **~260 (later 454)
proboscis-MN spikes per tick, rest ~4-6, bitter 0, saturating by 100 Hz** `FINDINGS.md:1377-1378,
1410-1411`; feeding threshold `FEED_MN = 60` spikes/tick `flybox.py:286-287`. Caveat: SEZ not
modality-selective and walking (mechano 60 Hz) alone gives 1,590 `FINDINGS.md:1344-1348`.

**hotocoo / connectome-lab**: no sugar or MN9 benchmark (grep of `MN9|sugar|Gr64|Gr5a` returns
nothing under `malecns/` or `connectome-lab/`).

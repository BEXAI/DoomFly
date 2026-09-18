# MaleCNS v1.0 flat-file research notes

Verified against the three files in `data/raw/` (all numbers below were printed by scripts run on 2026-09-18;
scratch outputs in the session scratchpad `ann1.txt, ann2.txt, ann3.txt, nt.txt, w.txt, w2.txt, w3.txt, grn*.txt`).
Companion machine-readable outputs: `data/graph/sets_draft.json` (bodyId sets) and `data/graph/columns.json` (column names used).

---

## 1. Annotations — `body-annotations-male-cns-v1.0-minconf-0.5.feather`

Shape **211,577 rows x 36 columns**. `bodyId` is unique (range 10001 – 1,571,825,087).

### Columns and dtypes (pandas dtype after `read_feather`; arrow type where different)

| column | dtype | non-null frac | notes |
|---|---|---|---|
| assignedOlHex1 | float64 | 0.112 | optic-lobe hex column coordinate 1 (1–36), **only for 15 `ol_intrinsic` columnar types** |
| assignedOlHex2 | float64 | 0.112 | hex coordinate 2 (1–39) |
| bodyId | int64 | 1.000 | primary key |
| flywireType | str | 0.677 | |
| group | float64 | 0.697 | numeric group id |
| instance | str | 0.763 | `type_L` / `type_R` / `type_M` — 75.5% of rows end in `_L/_R/_M` |
| somaSide | str | 0.712 | L / R / M |
| statusLabel | category (arrow dictionary, ordered) | 0.996 | fine-grained tracing status |
| superclass | str | 0.788 | |
| type | str | 0.778 | cell type (same vocabulary as NT file `cell_type`) |
| vfbId | str | 0.788 | |
| hemibrainType | str | 0.156 | |
| itoleeHl | str | 0.178 | hemilineage (Ito/Lee) |
| supertype | str | 0.161 | numeric-string ids |
| birthtime | str | 0.037 | early/late |
| mancBodyid, mancGroup | float64 | 0.089 / 0.069 | MANC cross-refs |
| mancType | str | 0.108 | |
| subclass | str | 0.104 | motor/sensory subclasses (see §5, §6) |
| synonyms | str | 0.019 | literature names, e.g. "Yao & Scott 2022: Sugar SEL PN" |
| class | str | 0.125 | |
| rootSide | str | 0.085 | L / R / unknown — **side for sensory neurons (incl. photoreceptors)** |
| somaNeuromere | str | 0.103 | |
| trumanHl | str | 0.093 | hemilineage (Truman) |
| dimorphism | str | 0.011 | |
| matchingNotes | str | 0.016 | |
| entryNerve | str | 0.056 | sensory entry nerve |
| mancSerial, mcnsSerial | float64 | 0.026 / 0.019 | |
| serialMotif | str | 0.004 | |
| fruDsx | str | 0.024 | fru/dsx expression |
| exitNerve | str | 0.005 | motor exit nerve |
| receptorType | str | 0.004 | putative_ppk23 / ppk25 / IR52b (leg pheromone GRNs) |
| somaLocation | object (arrow `list<int64>`) | **0.670** | `[x, y, z]` **voxel** coordinates |
| tosomaLocation | object (arrow `list<int64>`) | 0.005 | |
| status | str | 0.974 | coarse status |

### superclass value_counts

```
ol_intrinsic        89403      ascending_neuron   1846     cb_endocrine          72
NaN                 44877      descending_neuron  1314     ENS                   50
cb_intrinsic        32164      vnc_motor           708     vnc_tbc               38
vnc_intrinsic       13161      visual_centrifugal  563     vnc_sensory_tbc       36
visual_projection    9201      sensory_ascending   537     vnc_endocrine         22
vnc_sensory          6370      cb_motor            107     cb_sensory_tbc        14
ol_sensory           6098      vnc_efferent         94     sensory_descending    12
cb_sensory           4868                                  efferent_ascending 8, efferent_descending 4,
                                                           cb_efferent 4, visual_projection_tbc 2,
                                                           sensory_ascending_tbc 2, descending_neuron_tbc 2
```
Rows with NaN superclass are almost all non-neurons/fragments: status Orphan 15893, Glia 11864, Unimportant 10751, NaN 3470, Assign 1832, Anchor 551, Traced 516. Every row with a `type` has a superclass.

### class value_counts (all 22 values; fewer than 40 exist)

```
NaN                            185064    ALPN     686    hygrosensory        66
visual                           6091    ALLN     420    chemosensory        58
Kenyon_Cell                      4064    DAN      340    SEZPN               27
CX                               2950    ol_bilateral 116 thermosensory      25
olfactory                        2639    MBON      97    ALIN                24
mechanosensory_tactile           2558                    ALON                14
mechanosensory                   1733                    mechanosensory_tbc  11
unknown_sensory                  1712
mechanosensory_proprioceptive    1454
gustatory                        1428
```
`class` is mostly used for sensory modality and a few brain systems (MB, CX, AL). Optic-lobe intrinsic and VPN neurons have **no class** (VPNs: class NaN for all 9203).

### status / statusLabel

```
status:  Traced 165122 | Orphan 15925 | Glia 11864 | Unimportant 10751 | NaN 5472 | Assign 1832 | Anchor 611
statusLabel (top): Roughly traced 71979 | Reviewed 54066 | Prelim Roughly traced 36387 | Orphan 12075 | Glia 11864
  | Unimportant 10751 | Out of scope 4585 | Orphan-artifact 2318 | RT Hard to trace 2019 | 0.5assign 1832
  | Orphan hotknife 1428 | NaN 887 | Leaves 528 | Anchor 283 | Soma Anchor 201 | PRT Orphan 140 | ...
```
All neurons in the sets we care about (T4, L1-3, VPN, DN, MN, KC) are status `Traced`. Photoreceptors: 4107 Traced, 1983 status NaN, 1 Anchor.

### Side columns

```
somaSide:  L 75215 | R 75119 | NaN 60851 | M 392
rootSide:  NaN 193638 | R 9723 | L 7802 | unknown 414
```
- `somaSide` covers ol_intrinsic (44606 L / 44795 R / 2 NaN), visual_projection (4589 L / 4612 R), cb_intrinsic, DNs (656 L / 648 R / 10 M), motor (vnc_motor 355 L / 353 R; cb_motor 54 L / 53 R), KC, DAN, MBON.
- `somaSide` is NaN for **all sensory superclasses** (ol_sensory 6062 of 6098, cb_sensory 4868, vnc_sensory 6368, sensory_ascending 537) — these use **`rootSide`** (ol_sensory 2345 L / 3746 R; vnc_sensory 3186 L / 3170 R; cb_sensory 1961 L / 2494 R / 413 unknown).
- `instance` suffix `_L/_R` agrees with somaSide in 100.0% of 145,960 overlapping rows and with rootSide in 99.99% of 12,031 rows, so `instance[-1]` is a safe third fallback.
- Combined side = somaSide → rootSide → instance suffix: L 83751 / R 85478 / M 392 / NaN 41956 (NaN almost entirely rows with no superclass; only 413 cb_sensory "unknown" + ~80 misc neurons remain unsided).

### somaNeuromere value_counts

```
NaN 189757 | T2 5076 | T1 4296 | T3 3976 | CG 2401 | LB 1313 | A1 933 | TC 729 | DC 568 | MX 500 | MD 372
| A2 273 | A3 241 | A4 212 | A8 198 | A5 179 | A6 164 | A9 143 | A7 116 | GNG 89 | A10 39 | NA 2
```

### Soma position columns

- **`somaLocation`**: `list<int64>` of length 3 `[x, y, z]`, non-null for **141,781 rows (67.0%)**. Ranges: x 2468–93668, y 4758–68996, z 10154–134531 → these are **voxel coordinates** (inferred: MaleCNS native voxel is 8 nm, and 124k z-voxels x 8 nm ≈ 1 mm matches brain+VNC length; no unit metadata in the file — verify before converting to µm). Coverage by superclass: ol_intrinsic 0.907, cb_intrinsic 0.995, visual_projection 0.996, descending 0.995, vnc_motor 0.993, vnc_intrinsic 0.967; **0.0 for all sensory superclasses** (ol_sensory 0.005, cb/vnc_sensory 0.0 — somata are outside the volume).
- `tosomaLocation`: same format, only 995 rows (0.47%) — a point on the neuron toward the soma for cells whose soma is missing.
- Optic-lobe **hex columns `assignedOlHex1`/`assignedOlHex2`**: 23,720 rows, all `ol_intrinsic`, only 15 columnar types: L5 1773, C3 1770, L2 1767, L1 1767, Tm1 1767, T1 1764, Mi1 1762, Mi9 1760, Tm2 1758, Mi4 1758, Tm9 1743, Tm20 1732, **L3 892, C2 874, Tm4 833 (right side only)**. 879 unique (hex1,hex2) column ids on L, 892 on R. L1 hex coverage 99.0% (L) / 100% (R). T4/T5/Tm3/photoreceptors have **no** hex assignment.
- Columns matching hex/col/ol/assigned/position/location/soma: `assignedOlHex1, assignedOlHex2, somaSide, itoleeHl, somaNeuromere, somaLocation, tosomaLocation`.

---

## 2. Neurotransmitters — `body-neurotransmitters-male-cns-v1.0.feather`

Shape **1,835,518 rows x 10 columns**; `body` unique. It covers every body with ≥1 synapse prediction (mostly fragments): only 10.2% of NT bodies are in the annotation file, but **88.4% of annotation bodies have an NT row** (100% for every neuron superclass; only NaN-superclass rows lack it).

| column | dtype | non-null |
|---|---|---|
| body | int64 | 1.0 |
| cell_type | str | 0.090 (identical to annotation `type` where both present: 100% agreement) |
| total_nt_predictions | int32 | 1.0 (median 1; 95th pct 129; max 31,347) |
| predicted_nt_confidence | float64 | 0.9995 |
| predicted_nt | str | 1.0 |
| ground_truth | str | 0.047 |
| celltype_total_nt_predictions | int32 | 1.0 |
| celltype_predicted_nt | str | 1.0 |
| celltype_predicted_nt_confidence | float64 | 0.090 |
| consensus_nt | str | 1.0 |

Label sets and counts:
```
predicted_nt:          unclear 1684226 | acetylcholine 95420 | glutamate 28199 | gaba 20358 | dopamine 4447 | histamine 2259 | serotonin 483 | octopamine 126
celltype_predicted_nt: unclear 1673617 | acetylcholine 98512 | glutamate 29016 | gaba 21483 | histamine 7969 | dopamine 4448 | serotonin 386 | octopamine 87
consensus_nt:          unclear 1671117 | acetylcholine 104193 | glutamate 29443 | gaba 22196 | histamine 8024 | dopamine 396 | octopamine 101 | serotonin 48
ground_truth:          NaN 1750034 | acetylcholine 52779 | glutamate 14397 | gaba 13150 | histamine 4683 | dopamine 380 | octopamine 51 | serotonin 44
```
Confidence column = **`predicted_nt_confidence`**. Quantiles: min 0.187, 1% 0.463, 5% 0.651, 10% 0.863, 25% 0.911, 50% 0.975, 75% 0.975, 90–max 0.975. It saturates at 0.9754 (the "unclear" rows carry the saturated value). Per label mean confidence: ACh 0.943, histamine 0.868, GABA 0.831, glutamate 0.786, dopamine 0.749, octopamine 0.740, serotonin 0.639.
Spot checks: T4a → acetylcholine (1597/1684), L1 → glutamate, L2 → ACh, Mi1 → ACh, Tm9 → ACh, PAM01 → dopamine, MBON01 → glutamate, DNa02/DNp01 → ACh, Dm8a → glutamate.
**Use `consensus_nt` for cells** (see §8: `predicted_nt` calls all 1342 KCg-m "dopamine" while consensus says acetylcholine).

---

## 3. Weights — `connectome-weights-male-cns-v1.0-minconf-0.5.feather`

Schema: `body_pre: int64, body_post: int64, weight: int64` (2318 record batches). **151,856,684 rows**, total weight (synapse count) **311,833,243**.

- weight: min 1, max 2591; quantiles 10/25/50 % = 1, 75 % = 2, 90 % = 3, 95 % = 5, 99 % = 15, 99.9 % = 61, 99.99 % = 169.
- Distribution: weight 1 → 94,185,919 rows; 2 → 34,656,359; 3 → 11,222,655; 4 → 4,168,887; 5 → 1,966,596.
- **rows with weight ≥ 5: 7,622,864 (5.02% of rows, 31.4% of total weight)**; ≥2: 57,670,765 (38%); ≥3: 23,014,406; ≥10: 2,799,910; ≥20: 1,068,256; ≥50: 228,233.
- **Self edges: 123 rows, total weight 542.**
- Duplicate (pre,post) pairs: not checked (would need a 152M-row hash; the full-array attempt exceeded RAM). Treat as possibly present and aggregate with a groupby-sum when building the graph.
- Unique `body_pre`: 1,834,661 (186,159 in annotations); unique `body_post`: **87,576,984** (189,751 in annotations) — the post side includes tens of millions of tiny fragments. Unique bodies overall: 88,384,522.
- **bodyIds are NOT a subset of annotation bodyIds.** Fraction of rows with pre in annotations 0.917; post in annotations 0.202; **pre AND post both in annotations: 0.1714 (26,028,386 rows), carrying 40.2% of total weight**. Both typed: 0.1655 of rows / 39.2% of weight. Both status Traced: 0.168.
- 90.6% of annotation bodies appear in the weights file (88.0% as pre); 9.4% have zero in and zero out.
- Weight originating from pre bodies not in annotations: 14.9M (4.8%). Of annotated-pre output, only 42.2% lands on annotated posts (the rest is on orphan fragments) — so per-neuron "output" totals restricted to the annotated graph are ~0.4x the raw totals.
- Memory: reading the whole table with pyarrow takes ~3.6 GB; adding pandas copies blew a 15 GB machine. Iterating `ipc.open_file(...).get_batch(i)` with `np.searchsorted` + `np.bincount` against the sorted annotation ids stays under 9 GB (peak came from `np.unique` on the post column).

---

## 4. Eye input neurons

**Side column for the optic lobe:** `somaSide` is populated for all `ol_intrinsic` (L1–L5, T4, T5, Mi, Tm, C2/3, Lawf…) and `visual_projection` neurons. Photoreceptors (`superclass == 'ol_sensory'`) have **no somaSide** — use **`rootSide`** (100% populated for R-types, agrees with `instance` suffix in every case). Fallback tested: `instance[-1]` ∈ {L,R} reproduces somaSide 100% and rootSide 99.99%.

Photoreceptor type strings (all `superclass='ol_sensory'`, `class='visual'`), by rootSide:
```
type           L     R
R1-R6       1112  2265
R7y          230   252      R8y          230   251
R7p          173   159      R8p          172   158
R7d           40    42      R8d           35    41
R7_unclear   165   239      R8_unclear   188   254
R7R8_unclear   0    85
HBeyelet       4     3   (somaSide populated for these 7)
```
Note the **left/right asymmetry of R1-R6 (1112 vs 2265)**: 611 L-side and 1372 R-side photoreceptors have status NaN (not Traced); the left eye's outer photoreceptors are under-reconstructed. Total photoreceptors: L 2345, R 3746.

Optic-lobe interneurons (`ol_intrinsic`, all Traced), by somaSide:
```
        L    R          L    R          L     R
L1    884  892   T4a  835  849   Mi1   886   887
L2    886  893   T4b  844  846   Tm1   887   890
L3    880  892   T4c  895  883   Tm2   883   883
L4    879  891   T4d  850  859   Tm3  1017  1037
L5    889  898   T5a  826  838   Tm4   837   833
C2    871  874   T5b  863  852   Tm9   884   887
C3    887  892   T5c  862  858   Lawf1 177  184
                 T5d  812  808   Lawf2 195  188
T4_unclear 3/1, T5a_unclear 0/1
```
T4 total 3424 L / 3437 R; T5 total 3363 L / 3356 R.

**Visual projection neurons:** label is **`superclass == 'visual_projection'`** (9201 cells; L 4589 / R 4612; class is NaN; plus 2 `visual_projection_tbc`). Top types: LC12 498, TmY14 477, LC17 353, LLPC1 285, LC10a 275, MeTu1 250, LLPC2 250, LPC1 224, LC9 219, LLPC3 219, LC10d 214, LC18 208, LPLC2 185, LC16 182, LC13 177 … also MeVP*, LoVP*, LT*, LPT*, HS (8), VS (34), H2 (2).
Prefix counts (all bodies whose `type` starts with the prefix):
```
LC*    4257 (4164 visual_projection, 88 ol_intrinsic, 4 cb_intrinsic, 1 tbc)   L 2114 / R 2143
LPLC*   417 (all visual_projection)                                          L 210 / R 207
LLPC*   761 (all visual_projection)                                          L 378 / R 383
MeTu*  1009 (all visual_projection)                                          L 501 / R 508
MeVP*   917 (882 VP, 35 ol_intrinsic);  LoVP* 692 (676 VP);  LPC* 380 (all VP)
LT*     203 (164 VP, 33 visual_centrifugal); LPT* 302 (209 centrifugal, 91 VP)
```
`visual_centrifugal` (563) is the feedback class (MeVC*, LPT*, DCH/VCH) — not an output of the eye.

---

## 5. Motor readouts

Superclass counts for descending/motor/efferent variants:
```
descending_neuron 1314 | vnc_motor 708 | cb_motor 107 | vnc_efferent 94 | sensory_descending 12
| efferent_ascending 8 | efferent_descending 4 | cb_efferent 4 | descending_neuron_tbc 2
```
Motor neurons = `superclass ∈ {vnc_motor, cb_motor}` → 815 cells (class is NaN for all).

`subclass` of motor neurons (vnc_motor / cb_motor): ad 214/0 (abdominal), **fl 135/0 (front leg)**, hl 130/0 (hind leg), ml 116/0 (middle leg), wm 67/0 (wing), nm 24/20 (neck), hm 16/0 (haltere), xm 6/0, pm 0/67 (proboscis/pharynx), am 0/13 (antennal), rm 0/7.
`somaNeuromere` of motor neurons: T1 173, T2 175, T3 152, A1 56, A2 28, A3 22, A4 22, A9 21, A8 18, A5 16, A10 10, A6 8, A7 6; cb_motor: LB 16, GNG 4, NaN 87.
Cross-tab: fl ↔ T1 exactly (135/135); ml ↔ T2 (116); hl ↔ T3 (130); wm: T2 55 + T1 12; nm: T1 24, LB 14, NaN 6.
Exit nerves of MNs: MetaLN 122, MesoLN 116, ProLN 81, AbN4 72, AbNT 64, AbN2 44, PhN 40, …

**Front-leg motor neurons:** `superclass=='vnc_motor' & subclass=='fl'` (equivalently `somaNeuromere=='T1'` minus neck/wing MNs). Candidates in T1: fl 68 L / 67 R, nm 12/12, wm 6/6, xm 1/1. **Use `subclass=='fl'`: 68 left, 67 right, 135 total** (exitNerve ProLN/DProN/VProN/ProAN). Types (L/R): Acc. ti flexor MN 10/9, Tr flexor MN 8/7, Ti flexor MN 5/5, Ta depressor MN 5/4, Fe reductor MN 4/6, Tergopleural/Pleural promotor MN 4/4, Tergotr. MN 4/4, ltm MN 4/4, Sternal posterior rotator MN 4/2, Acc. tr flexor MN 3/3, Ta levator MN 2/3, Ti extensor MN 2/2, Tr extensor MN 2/2, Sternotrochanter MN 2/2, Sternal anterior rotator MN 2/2, Pleural remotor/abductor MN 2/2, ltm2-femur MN 2/2, ltm1-tibia MN 2/1, Sternal adductor MN 1/1. (Type strings are shared across T1/T2/T3 — always filter by subclass/neuromere.)

Descending neurons: 1314 (`descending_neuron`; L 656 / R 648 / M 10), 480 types. Named DN counts by somaSide:
```
DNa01 1L/1R   DNa02 1L/1R   DNa03 1L/1R   DNb01 1L/1R   DNp01 1L/1R   DNp10 1L/1R
DNg11 3L/3R   DNg13 1L/1R   DNg14 1L/1R   MDN 2L/2R
DNg12 is split into subtypes: DNg12_a 4/4, _b 5/5, _c 4/4, _d 1/1, _e 3/3, _f 2/2, _g 1/1, _h 1/1 (17 per side)
```
Total output weight (sum over all posts) and rank among 480 DN types: DNa02 47,622 (rank 35), DNp10 41,446 (52), DNa01 40,323 (59), MDN 39,119 (68), DNb01 37,839 (75), DNa03 33,588 (103), DNg13 31,606 (119), DNg11 13,587 (346), DNg14 9,930 (398), **DNp01 8,873 (rank 412 — the giant fiber has few chemical output synapses; in_w 43,478)**.

**Top 30 DN types by total output synapse weight** (n cells, out_w all posts, out_w onto annotated bodies, in_w):
```
DNpe053  2 131160  38184  37475      DNg08   23  55384 18143 11808
DNg98    2 119273  44826  16006      DNae009  2  54761 16663 15082
DNg100   2 106295  34601  43069      DNge035  2  54053 20305 11997
pIP1     2  87956  28913  44429      DNge136  4  52367 21090 13664
DNg102   4  84373  28206  20032      DNge050  2  51791 17428 35101
DNg74_a  2  84239  33230  36833      DNge019 11  50950 18159 13989
DNg108   2  72003  26509  34138      DNp42    2  50371 17184 17813
DNp13    2  69270  23427  20780      DNg75    2  50175 17467 37581
aSP22    2  68793  23895  19724      DNge079  2  49740 20466 26611
DNpe031  4  68531  23329  24096      DNg88    2  49729 14436 24833
DNa13    4  67413  21786  49289      DNg97    2  49648 16245 36289
DNd03    2  65542  30297   9561      DNp48    2  49641 18752  8333
DNg74_b  2  60926  23409  22456      DNg106  16  49164 17825 12824
DNg70    2  59713  25766  10518      DNpe056  2  48517 20092 15441
DNpe005  2  58504  16866   6695      DNg93    2  48441 20063 13687
```
(Full table: scratchpad `dn_out.csv`; per-body in/out totals for all annotated bodies: `per_body_weights.parquet`.)

---

## 6. Gustatory

Label is **`class == 'gustatory'`** (1428 cells; superclass vnc_sensory 1073 / cb_sensory 275 / sensory_ascending 80; side via rootSide L 713 / R 715). `subclass`: leg bristle 768, wing bristle 385, **labellar bristle 163**, **taste peg 60**, **pharyngeal sensillum 48**, NaN 4.
**There is no explicit sugar/bitter/Gr64/Gr66/Gr5a label anywhere** (grep of class/type/subclass/instance/synonyms/flywireType/hemibrainType/mancType/receptorType for sugar, bitter, GRN, gustatory, Gr64, Gr66, Gr5a, Gr43, water, ppk28, Ir76b). Hits: `type`/`flywireType` contain "GRN" only for `claw_tpGRN` (50) and `dorsal_tpGRN` (10) (taste pegs); `receptorType` putative_ppk23 269 / ppk25 257 / IR52b 226 (leg pheromone GRNs); `synonyms` "Yao & Scott 2022: Sugar SEL PN" (GNG540 x2, GNG550 x2), "Sugar SEL LN" (GNG056 x2), "Bitter-SEL" (DNg28 x4, superclass efferent_descending) — these are **second-order** neurons, not GRNs.

Labellar GRN types (subclass labellar bristle, entryNerve MxLbN): LB1a 11, LB1b 6, LB1c 16, LB1d 5, LB1e 18 (LB1* = 56); LB2a 4, LB2b 2, LB2c 6, LB2d 4 (LB2* = 16); LB3a 17, LB3b 11, LB3c 23, LB3d 26, LB3 1 (LB3* = 78); LB4a 4, LB4b 7 (LB4* = 11). Pharyngeal: PhG1a–PhG16 (48, aPhN/PhN). Taste pegs: claw_tpGRN 50, dorsal_tpGRN 10.
Connectivity tests (weights file):
- Direct GRN input to the Sugar-SEL neurons is dominated by pharyngeal PhG9 (754) and dorsal_tpGRN (356); labellar bristle types give ≤8. Bitter-SEL (DNg28) receives **no** direct GRN input, so SEL connectivity cannot classify labellar GRNs.
- Hierarchical clustering of downstream-target profiles separates LB1*, LB2*, LB3*, LB4* cleanly (and PhG*/tpGRN differently).
- 2-hop path strength GRN → X → MN9 (Σ w(GRN,X)·w(X,MN9)): LB3d 2153, LB3c 1888, LB3a 946, LB3b 662 ≫ LB1c 741, LB1a 286, LB1b 175, LB1e 159 ≫ LB2* ≤152, LB4* ≤41. Sugar GRNs are the labellar class driving proboscis extension, and LB3* (78 cells ≈ one per labellar bristle per side) is also the most numerous class.
- **Proposed proxies (flagged, unverified against literature):** `grn_sugar` = LB3a–d (78), `grn_bitter` = LB1a–e (56); LB2*/LB4* likely water/salt/other. Verify against MaleCNS paper supplementary tables before use.

**MN9**: `type == 'MN9'`, 2 cells — **bodyId 10331 (MN9_L, somaSide L)** and **bodyId 16949 (MN9_R, somaSide R)**; superclass cb_motor, subclass pm, exitNerve PhN. In-weights: 10331 in 6358 / out 632; 16949 in **633** / out 1145 (right MN9 receives 10x less input — likely truncated reconstruction). Top MN9 input types: DNge062 556, GNG015 478, GNG120 443, GNG095 436, GNG117 413, GNG130 409, GNG108 381, GNG234 362, … No GRN synapses directly onto MN9.

---

## 7. Dopamine / learning sets (by somaSide, L/R)

```
PAM*  316 (158/158), class DAN. PAM01 23/21, PAM02 9/8, PAM03 7/5, PAM04 16/16, PAM05 9/11, PAM06 14/14,
      PAM07 7/7, PAM08 25/25, PAM09 4/5, PAM10 7/8, PAM11 8/7, PAM12 11/11, PAM13 8/8, PAM14 9/9, PAM15 1/3
PPL1* 16 (8/8): PPL101..PPL108 1 per side each.   PPL2* 8 (4/4): PPL201-204.
KC*   4064 (2019/2045), class Kenyon_Cell: KCg-m 1342, KCab-s 657, KCab-m 536, KCab-c 488, KCa'b'-ap2 291,
      KCg-d 206, KCa'b'-m 205, KCa'b'-ap1 199, KCab-p 129, KCg-s1..s4 2 each, KC 2, KCg 1
MBON* 97 (48/49), class MBON: MBON01–MBON35 mostly 1/side; MBON07, 09, 12, 14, 15, 19 2/side; MBON10 4L/5R;
      plus MBON15-like 2/2, MBON17-like 1/1, MBON25-like 2/2 (MBON08 absent)
APL   2 (1/1, class NaN, cb_intrinsic).   DPM 2 (1/1).
```

---

## 8. Surprising / important caveats

1. **Post-side coverage:** only 20% of weight rows have an annotated post body; 87.6M unique post ids are fragments. Restricting to annotated pre AND post keeps 17% of rows but 40% of synapses. Per-neuron output totals in the annotated graph are ~42% of raw totals.
2. **Left eye under-reconstructed:** R1-R6 1112 L vs 2265 R; 1983 photoreceptors have status NaN. Optic-lobe hex columns for **L3, C2, Tm4 exist only on the right side**.
3. **Sensory neurons have no somaSide and no somaLocation** — use `rootSide` (and `entryNerve`); sides for all interneurons/DN/MN come from `somaSide`.
4. **NT `predicted_nt` mislabels Kenyon cells:** all 1342 KCg-m have predicted_nt = dopamine (conf ~0.75) while `consensus_nt` = acetylcholine; `consensus_nt` also collapses dopamine to 396 bodies. Use `consensus_nt` for cell-level NT and `predicted_nt` only for fragments.
5. `predicted_nt_confidence` saturates at 0.9754 and is high for "unclear" rows — it is not a usable filter on its own.
6. **DNp01 (giant fiber) has almost no chemical output** (8,873 total, rank 412/480) — its gap-junction outputs to TTMn/PSI are not in this file. DNg28 (Yao & Scott "Bitter-SEL") is classed `efferent_descending`, not `descending_neuron`.
7. MN9_R (16949) receives 633 input synapses vs 6358 for MN9_L — a strong asymmetry that will bias proboscis readouts.
8. Motor type strings (e.g. "Ti flexor MN") are shared across T1/T2/T3 — always filter by `subclass` (fl/ml/hl) or `somaNeuromere`.
9. Highest-output single neurons are optic-lobe wide-field cells: CT1 (188k/178k), APL (134k/124k), Li39, MeVC11, Am1, LPi21; ascending neurons AN02A002/AN07B004 (~78k each).
10. `DNg12` has no plain type — it is split into `DNg12_a`…`DNg12_h` (17 per side). `DNg11` has 3 cells per side.
11. The weights file is `minconf-0.5` and has 123 self-edges (weight 542 total); duplicates of (pre,post) were not verified — aggregate on load.

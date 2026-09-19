#!/usr/bin/env python3
"""Build the signed sparse MaleCNS v1.0 graph, neuron sets, eye map and soma positions.

Inputs (data/raw/, from src/download.py):
    body-annotations-male-cns-v1.0-minconf-0.5.feather
    body-neurotransmitters-male-cns-v1.0.feather
    connectome-weights-male-cns-v1.0-minconf-0.5.feather

Outputs (data/graph/):
    W.npz          scipy CSC (N, N) float32, W[post, pre] = sign(pre) * synapse count
    ids.npy        int64 bodyId per row / column
    sets.json      name -> list of bodyIds  (eye, motor, DN, GRN, MB sets; from Section 5 of plan.md)
    eye_map.npz    per eye: L1/L2/L3 bodyIds with their optic-lobe hex column coordinates
    positions.npy  float32 (N, 3) soma position in micrometres (NaN if unknown after partner fill)
    groups.npy     int8 (N,) colour groups for the PiP: 0 other, 1 L eye input, 2 R eye input,
                   3 descending, 4 front-leg motor
    build_stats.json

Choices (documented in README):
  * Neurons kept: every annotated body with a non-null `superclass` (as hotocoo/malecns).
  * Edges kept: synapse count >= --min-weight (default 5), both ends kept, self-edges dropped,
    duplicate (pre, post) rows summed.
  * Sign from `consensus_nt` of the presynaptic cell: acetylcholine, dopamine, octopamine,
    serotonin -> +1; gaba, glutamate (GluCl, Liu & Wilson 2013), histamine -> -1;
    unclear / missing -> +1 (documented; --unknown-sign 0 silences them instead).
  * Side: somaSide, then rootSide (sensory neurons), then instance suffix _L/_R.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "graph")
F_ANN = os.path.join(RAW, "body-annotations-male-cns-v1.0-minconf-0.5.feather")
F_NT = os.path.join(RAW, "body-neurotransmitters-male-cns-v1.0.feather")
F_W = os.path.join(RAW, "connectome-weights-male-cns-v1.0-minconf-0.5.feather")

VOXEL_UM = 0.008  # 8 nm isotropic voxels

EXC = {"acetylcholine", "dopamine", "octopamine", "serotonin"}
INH = {"gaba", "glutamate", "histamine"}


def side_of(a: pd.DataFrame) -> pd.Series:
    side = a["somaSide"].copy()
    side = side.where(side.isin(["L", "R", "M"]), a["rootSide"])
    inst = a["instance"].astype("string")
    suffix = inst.str[-2:].map({"_L": "L", "_R": "R"})
    side = side.where(side.isin(["L", "R", "M"]), suffix)
    return side


def load_neurons() -> pd.DataFrame:
    cols = ["bodyId", "type", "instance", "superclass", "class", "subclass", "somaSide", "rootSide",
            "somaNeuromere", "exitNerve", "assignedOlHex1", "assignedOlHex2", "somaLocation", "status"]
    a = pd.read_feather(F_ANN, columns=cols)
    a = a[a["superclass"].notna()].reset_index(drop=True)
    a["side"] = side_of(a)
    nt = pd.read_feather(F_NT, columns=["body", "consensus_nt", "predicted_nt", "predicted_nt_confidence"])
    a = a.merge(nt.rename(columns={"body": "bodyId"}), on="bodyId", how="left")
    return a


def signs(a: pd.DataFrame, unknown_sign: float) -> np.ndarray:
    nt = a["consensus_nt"].fillna("unclear").str.lower()
    s = np.full(len(a), unknown_sign, np.float32)
    s[nt.isin(EXC).to_numpy()] = 1.0
    s[nt.isin(INH).to_numpy()] = -1.0
    return s


def load_edges(body_to_idx: dict, min_weight: int):
    """Stream the 152 M-row weights file, keep w >= min_weight with both ends known."""
    lookup = pd.Series(body_to_idx)
    reader = ipc.open_file(F_W)
    pres, posts, ws = [], [], []
    t0 = time.time()
    for b in range(reader.num_record_batches):
        rb = reader.get_batch(b)
        w = rb.column("weight").to_numpy()
        keep = w >= min_weight
        if not keep.any():
            continue
        pre = rb.column("body_pre").to_numpy()[keep]
        post = rb.column("body_post").to_numpy()[keep]
        pi = lookup.reindex(pre).to_numpy()
        qi = lookup.reindex(post).to_numpy()
        ok = ~(np.isnan(pi) | np.isnan(qi))
        pres.append(pi[ok].astype(np.int32)); posts.append(qi[ok].astype(np.int32)); ws.append(w[keep][ok].astype(np.float32))
        if b % 400 == 0:
            print(f"  batch {b}/{reader.num_record_batches}  kept {sum(len(x) for x in ws):,}  {time.time()-t0:.0f}s", flush=True)
    return np.concatenate(pres), np.concatenate(posts), np.concatenate(ws)


def fill_positions(pos: np.ndarray, W: sp.csc_matrix, rounds: int = 4) -> np.ndarray:
    """Fill unknown soma positions with the mean position of synaptic partners (as hotocoo)."""
    pos = pos.copy()
    A = (abs(W) + abs(W).T).tocsr()  # symmetric partner weights
    for _ in range(rounds):
        known = ~np.isnan(pos[:, 0])
        if known.all():
            break
        P = np.where(known[:, None], pos, 0.0).astype(np.float32)
        num = A @ P
        den = A @ known.astype(np.float32)
        fill = (~known) & (den > 0)
        pos[fill] = num[fill] / den[fill, None]
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-weight", type=int, default=5)
    ap.add_argument("--unknown-sign", type=float, default=1.0)
    ap.add_argument("--sets-draft", default=os.path.join(OUT, "sets_draft.json"))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with open(args.sets_draft) as f:   # fail early if the draft sets are missing
        draft = json.load(f)

    print("loading annotations + NT ...", flush=True)
    a = load_neurons()
    n = len(a)
    ids = a["bodyId"].to_numpy(np.int64)
    body_to_idx = {int(b): i for i, b in enumerate(ids)}
    sign = signs(a, args.unknown_sign)
    print(f"neurons: {n:,}   sign +: {(sign>0).sum():,}  -: {(sign<0).sum():,}  0: {(sign==0).sum():,}")

    print("streaming edges ...", flush=True)
    pre, post, w = load_edges(body_to_idx, args.min_weight)
    self_edges = int((pre == post).sum())
    keep = pre != post
    pre, post, w = pre[keep], post[keep], w[keep]
    W = sp.coo_matrix((w * sign[pre], (post, pre)), shape=(n, n), dtype=np.float32)
    W.sum_duplicates()
    W = W.tocsc()
    W.eliminate_zeros()
    n_syn = int(np.abs(W.data).astype(np.int64).sum())
    print(f"edges: {W.nnz:,}  synapses: {n_syn:,}  inhibitory edge fraction: {(W.data<0).mean():.3f}  self-edges dropped: {self_edges}")
    sp.save_npz(os.path.join(OUT, "W.npz"), W)
    np.save(os.path.join(OUT, "ids.npy"), ids)

    # ---- positions
    loc = a["somaLocation"]
    pos = np.full((n, 3), np.nan, np.float32)
    has = loc.notna().to_numpy()
    pos[has] = np.stack(loc[has].to_numpy()).astype(np.float32) * VOXEL_UM
    n_known = int(has.sum())
    pos = fill_positions(pos, W)
    print(f"positions: {n_known:,} from somaLocation, {int((~np.isnan(pos[:,0])).sum()):,} after partner fill")
    np.save(os.path.join(OUT, "positions.npy"), pos)

    # ---- sets
    sets = {k: [int(b) for b in v if int(b) in body_to_idx] for k, v in draft.items() if not k.startswith("_")}
    bid = a["bodyId"]
    for side_name, side in (("left", "L"), ("right", "R")):
        m = a["side"] == side
        sets[f"mn_frontleg_{side_name}"] = bid[m & (a.superclass == "vnc_motor") & (a.subclass == "fl")].astype(int).tolist()
        sets[f"dn_{side_name}"] = bid[m & (a.superclass == "descending_neuron")].astype(int).tolist()
        sets[f"vpn_{side_name}"] = bid[m & (a.superclass == "visual_projection")].astype(int).tolist()
        sets[f"ol_{side_name}"] = bid[m & (a.superclass == "ol_intrinsic")].astype(int).tolist()
        for t in ("L1", "L2", "L3", "L4", "L5", "Mi1", "Tm1", "Tm2", "Tm9"):
            sets[f"eye_{side_name}_{t}"] = bid[m & (a.type == t)].astype(int).tolist()
        sets[f"t4_{side_name}"] = bid[m & a.type.astype("string").str.match(r"^T4[a-d]$", na=False)].astype(int).tolist()
        sets[f"t5_{side_name}"] = bid[m & a.type.astype("string").str.match(r"^T5[a-d]$", na=False)].astype(int).tolist()
        sets[f"photoreceptors_{side_name}"] = bid[m & (a.superclass == "ol_sensory") & a.type.astype("string").str.startswith("R", na=False)].astype(int).tolist()
    sets["mn_all"] = bid[a.superclass.isin(["vnc_motor", "cb_motor"])].astype(int).tolist()
    sets["mn_vnc"] = bid[a.superclass == "vnc_motor"].astype(int).tolist()
    sets["dn_all"] = bid[a.superclass == "descending_neuron"].astype(int).tolist()
    sets["MN9"] = bid[a.type == "MN9"].astype(int).tolist()
    with open(os.path.join(OUT, "sets.json"), "w") as f:
        json.dump(sets, f)
    print("sets:", {k: len(v) for k, v in sets.items() if k.startswith(("eye_left_L", "eye_right_L", "mn_front", "dn_", "vpn", "grn_s", "MN9", "t4"))})

    # ---- eye map (hex columns of lamina neurons, per side)
    em = {}
    for side_name, side in (("L", "L"), ("R", "R")):
        for t in ("L1", "L2", "L3"):
            m = (a["side"] == side) & (a.type == t) & a.assignedOlHex1.notna()
            em[f"{side_name}_{t}_bodies"] = bid[m].to_numpy(np.int64)
            em[f"{side_name}_{t}_hex"] = a.loc[m, ["assignedOlHex1", "assignedOlHex2"]].to_numpy(np.float32)
    np.savez(os.path.join(OUT, "eye_map.npz"), **em)
    print("eye map:", {k: len(v) for k, v in em.items() if k.endswith("bodies")})

    # ---- groups for the PiP
    groups = np.zeros(n, np.int8)
    idx = lambda name: np.fromiter((body_to_idx[b] for b in sets[name]), np.int64)
    for name in ("ol_left", "vpn_left"):
        groups[idx(name)] = 1
    for name in ("ol_right", "vpn_right"):
        groups[idx(name)] = 2
    groups[idx("dn_all")] = 3
    groups[idx("mn_frontleg_left")] = 4
    groups[idx("mn_frontleg_right")] = 4
    np.save(os.path.join(OUT, "groups.npy"), groups)

    stats = dict(n_neurons=int(n), n_edges=int(W.nnz), n_synapses=n_syn,
                 min_weight=args.min_weight, unknown_sign=args.unknown_sign,
                 inhibitory_edge_fraction=float((W.data < 0).mean()),
                 sign_counts=dict(exc=int((sign > 0).sum()), inh=int((sign < 0).sum()), zero=int((sign == 0).sum())),
                 nt_counts=a["consensus_nt"].fillna("missing").value_counts().to_dict(),
                 self_edges_dropped=self_edges, positions_known=n_known,
                 positions_after_fill=int((~np.isnan(pos[:, 0])).sum()),
                 set_sizes={k: len(v) for k, v in sets.items()})
    with open(os.path.join(OUT, "build_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print("done ->", OUT)


if __name__ == "__main__":
    main()

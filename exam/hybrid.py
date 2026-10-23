"""Hybrid of the selected PPR and MF configurations: score fusion and reciprocal-rank fusion.

Both component score matrices are computed once on the validation split and cached in results/.
Fusion is done over candidate-pool items only, per user:
  score fusion : w * minmax_u(MF) + (1 - w) * minmax_u(PPR)
  rank fusion  : w / (k0 + rank_MF) + (1 - w) / (k0 + rank_PPR),  k0 = 60 (Cormack et al. 2009)
"""
import json
import os
import sys

import numpy as np

from common import SPLITS, Data, ground_truth, mapk
from methods import mf_scores, ppr_scores, rank_top5
from tune import setup

PPR_CFG = {"alpha": 0.4, "transform": "raw", "lam": 0.05}
MF_CFG = {"factors": 64, "conf_a": 3.0, "reg": 0.02, "steps": 300, "target": "log", "seed": 42}


def component_scores(d, split):
    R_ppr, items, pool, fb, active = setup(d, split, PPR_CFG["lam"])
    S_ppr, info_ppr = ppr_scores(R_ppr, alpha=PPR_CFG["alpha"], transform=PPR_CFG["transform"])
    R_mf, items_mf, *_ = setup(d, split, 0.0)
    assert np.array_equal(items, items_mf)  # recency weights change values, not the item set
    S_mf, info_mf = mf_scores(R_mf, **MF_CFG)
    return S_ppr, S_mf, items, pool, fb, active, info_ppr, info_mf


def minmax_rows(S, mask):
    Sm = np.where(mask, S, np.nan)
    lo, hi = np.nanmin(Sm, axis=1, keepdims=True), np.nanmax(Sm, axis=1, keepdims=True)
    out = (S - lo) / np.where(hi > lo, hi - lo, 1.0)
    return np.where(mask, out, 0.0)


def ranks_rows(S, mask):
    Sm = np.where(mask, S, -np.inf)
    order = np.argsort(-Sm, axis=1, kind="stable")
    r = np.empty_like(order)
    np.put_along_axis(r, order, np.arange(S.shape[1])[None, :].repeat(S.shape[0], 0), axis=1)
    return r + 1  # 1-based rank


def fuse(S_ppr, S_mf, items, pool, w, kind, k0=60):
    mask = np.isin(items, pool)[None, :]
    if kind == "score":
        return w * minmax_rows(S_mf, mask) + (1 - w) * minmax_rows(S_ppr, mask) + 1e-9
    return w / (k0 + ranks_rows(S_mf, mask)) + (1 - w) / (k0 + ranks_rows(S_ppr, mask))


if __name__ == "__main__":
    d = Data()
    truth = ground_truth(d, SPLITS["val"][1])
    S_ppr, S_mf, items, pool, fb, active, ip, im = component_scores(d, "val")
    ev = lambda S, ms=0.0: mapk(truth, (lambda t: {u: t[u] for u in truth})(rank_top5(S, items, pool, fb, active, min_score=ms)))
    rows = [{"kind": "ppr_only", "map5": ev(S_ppr)}, {"kind": "mf_only", "map5": ev(S_mf, -np.inf)}]
    print(rows, flush=True)
    for kind in ["score", "rank"]:
        for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            rows.append({"kind": kind, "w_mf": w, "map5": ev(fuse(S_ppr, S_mf, items, pool, w, kind))})
            print(rows[-1], flush=True)
    best = max(rows, key=lambda r: r["map5"])
    print("BEST", best)
    os.makedirs("results", exist_ok=True)
    with open("results/val_hybrid.json", "w") as fh:
        json.dump({"ppr": PPR_CFG, "mf": MF_CFG, "ppr_info": ip, "mf_final_loss": im["final_loss"],
                   "rows": rows, "best": best}, fh, indent=1)

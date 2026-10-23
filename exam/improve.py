"""Bounded improvement round on validation (each idea is one variable on top of the selected hybrid).

A. MF seed ensemble: average MF scores over seeds 42..46 (variance reduction).
B. MF with recency-weighted counts (lam in {0.05, 0.1}).
C. Schedule-aware re-ranking: multiply the fused score by (A_target / A_hist)^beta, beta small.
"""
import json

import numpy as np

from common import SPLITS, Data, ground_truth, mapk
from hybrid import MF_CFG, PPR_CFG, fuse
from methods import mf_scores, ppr_scores, rank_top5
from tune import setup

W_MF = 0.8

d = Data()
hist, target = SPLITS["val"]
truth = ground_truth(d, target)
R0, items, pool, fb, active = setup(d, "val", 0.0)
R_ppr, *_ = setup(d, "val", PPR_CFG["lam"])
S_ppr, _ = ppr_scores(R_ppr, alpha=PPR_CFG["alpha"], transform=PPR_CFG["transform"])


def ev(S, ms=0.0):
    top = rank_top5(S, items, pool, fb, active, min_score=ms)
    return mapk(truth, {u: top[u] for u in truth})


rows = []
mfs = []
for seed in range(42, 47):
    S, _ = mf_scores(R0, **{**MF_CFG, "seed": seed})
    mfs.append(S)
    rows.append({"idea": "A", "seed": seed, "mf_map5": ev(S, -np.inf),
                 "hybrid_map5": ev(fuse(S_ppr, S, items, pool, W_MF, "score"))})
    print(rows[-1], flush=True)
S_ens = np.mean(mfs, axis=0)
rows.append({"idea": "A", "seed": "mean42-46", "mf_map5": ev(S_ens, -np.inf),
             "hybrid_map5": ev(fuse(S_ppr, S_ens, items, pool, W_MF, "score"))})
print(rows[-1], flush=True)

for lam in [0.05, 0.1]:
    R_l, *_ = setup(d, "val", lam)
    S, _ = mf_scores(R_l, **MF_CFG)
    rows.append({"idea": "B", "lam": lam, "mf_map5": ev(S, -np.inf),
                 "hybrid_map5": ev(fuse(S_ppr, S, items, pool, W_MF, "score"))})
    print(rows[-1], flush=True)

H = fuse(S_ppr, mfs[0], items, pool, W_MF, "score")
m = np.isin(d.a["week"], np.asarray(hist))
ah = np.bincount(np.searchsorted(items, d.a["show"][m][np.isin(d.a["show"][m], items)]), minlength=len(items))
at_s = d.airing_counts(target)
at = at_s.reindex(items).fillna(0).to_numpy()
ratio = np.where(ah > 0, at / np.maximum(ah, 1), 0.0)
for beta in [0.1, 0.2, 0.3, 0.5]:
    rows.append({"idea": "C", "beta": beta, "hybrid_map5": ev(H * ratio[None, :] ** beta + 1e-12)})
    print(rows[-1], flush=True)

with open("results/val_improve.json", "w") as fh:
    json.dump(rows, fh, indent=1)

"""Fifth bounded round (val + val2): recency signals on top of the final strategy.
(a) MF fitted on recency-weighted counts (lam), (b) third fused component = counts in the last 4 history weeks."""
import json
import numpy as np
from common import SPLITS, Data, ground_truth, mapk
from hybrid import MF_CFG, PPR_CFG, fuse, minmax_rows
from methods import mf_scores, ppr_scores, rank_top5
from rerank import user_channel_ratio
from tune import setup

d = Data(); rows = []
for split in ["val", "val2"]:
    hist, target = SPLITS[split]; truth = ground_truth(d, target)
    R0, items, pool, fb, active = setup(d, split, 0.0)
    R_ppr, *_ = setup(d, split, PPR_CFG["lam"])
    S_ppr, _ = ppr_scores(R_ppr, alpha=PPR_CFG["alpha"], transform=PPR_CFG["transform"])
    ratio = user_channel_ratio(d, items, hist, target) ** 0.3
    mask = np.isin(items, pool)[None, :]
    ev = lambda S: (lambda t: mapk(truth, {u: t[u] for u in truth}))(rank_top5(S, items, pool, fb, active))
    mfs = {}
    for lam in [0.0, 0.05, 0.1]:
        R_l, *_ = setup(d, split, lam)
        mfs[lam], _ = mf_scores(R_l, **MF_CFG)
        rows.append({"split": split, "idea": "mf_lam", "lam": lam,
                     "map5": ev(fuse(S_ppr, mfs[lam], items, pool, 0.8, "score") * ratio + 1e-12)}); print(rows[-1], flush=True)
    c = d.counts(range(hist.stop - 4, hist.stop))
    Rr = np.zeros_like(R0); c = c[c["show"].isin(items)]
    Rr[c["user"].to_numpy(), np.searchsorted(items, c["show"].to_numpy())] = c["n"].to_numpy()
    H = fuse(S_ppr, mfs[0.0], items, pool, 0.8, "score")
    for w3 in [0.1, 0.2, 0.3]:
        H3 = (1 - w3) * H + w3 * minmax_rows(Rr, mask)
        rows.append({"split": split, "idea": "recent4_component", "w3": w3, "map5": ev(H3 * ratio + 1e-12)}); print(rows[-1], flush=True)
json.dump(rows, open("results/val_improve5.json", "w"), indent=1)

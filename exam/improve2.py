"""Second bounded round: schedule-aware re-ranking variants on validation.

ratio_s = (airings of s per week in target) / (airings of s per week in the last `recent` history weeks)
final   = fused_score * ratio^beta   (schedule of the target window only: what airs, never who watched)
"""
import itertools
import json

import numpy as np

from common import SPLITS, Data, ground_truth, mapk
from hybrid import MF_CFG, PPR_CFG, fuse
from methods import mf_scores, ppr_scores, rank_top5
from rerank import schedule_ratio
from tune import setup


if __name__ == "__main__":
    d = Data()
    hist, target = SPLITS["val"]
    truth = ground_truth(d, target)
    R0, items, pool, fb, active = setup(d, "val", 0.0)
    R_ppr, *_ = setup(d, "val", PPR_CFG["lam"])
    S_ppr, _ = ppr_scores(R_ppr, alpha=PPR_CFG["alpha"], transform=PPR_CFG["transform"])
    S_mf0, _ = mf_scores(R0, **MF_CFG)
    R_l, *_ = setup(d, "val", 0.1)
    S_mf1, _ = mf_scores(R_l, **MF_CFG)

    def ev(S, ms=0.0):
        top = rank_top5(S, items, pool, fb, active, min_score=ms)
        return mapk(truth, {u: top[u] for u in truth})

    rows = []
    comps = {"hyb_mf_lam0": fuse(S_ppr, S_mf0, items, pool, 0.8, "score"),
             "hyb_mf_lam0.1": fuse(S_ppr, S_mf1, items, pool, 0.8, "score")}
    for (name, H), recent, beta in itertools.product(comps.items(), [None, 4, 2], [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]):
        r = schedule_ratio(d, items, hist, target, recent)
        rows.append({"base": name, "recent_weeks": recent or len(hist), "beta": beta,
                     "map5": ev(H * r[None, :] ** beta + 1e-12)})
        print(rows[-1], flush=True)
    # the same re-ranking on each single method, at the best (recent, beta) of the lam0 hybrid
    best = max((x for x in rows if x["base"] == "hyb_mf_lam0"), key=lambda x: x["map5"])
    r = schedule_ratio(d, items, hist, target, None if best["recent_weeks"] == len(hist) else best["recent_weeks"])
    for name, S, ms in [("ppr", S_ppr, 0.0), ("mf_lam0", S_mf0, -np.inf), ("popularity", np.broadcast_to(R0.sum(0), R0.shape), 0.0)]:
        Sn = S - S.min() + 1e-6 if ms == -np.inf else S  # MF scores can be negative: shift before scaling
        rows.append({"base": name + "+schedule", "recent_weeks": best["recent_weeks"], "beta": best["beta"],
                     "map5": ev(Sn * r[None, :] ** best["beta"])})
        print(rows[-1], flush=True)
    print("BEST", max(rows, key=lambda x: x["map5"]))
    with open("results/val_improve2.json", "w") as fh:
        json.dump(rows, fh, indent=1)

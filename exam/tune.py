"""Validation sweeps (history weeks 11-20 -> target weeks 21-30). Results go to results/val_<method>.json.

usage: python tune.py baselines | ppr | mf1 | mf2 <factors>
"""
import itertools
import json
import os
import sys
import time

import numpy as np

from common import SPLITS, Data, ground_truth, mapk
from methods import interaction_matrix, mf_scores, popularity_scores, ppr_scores, rank_top5


def setup(d, split, lam=0.0):
    hist, target = SPLITS[split]
    pool = d.pool(target)
    R, items = interaction_matrix(d, hist, lam=lam)
    pop = R.sum(axis=0)
    order = np.argsort(-pop, kind="stable")
    in_pool = np.isin(items, pool)
    fallback = [int(items[j]) for j in order if in_pool[j]]
    return R, items, pool, fallback, R.sum(axis=1) > 0


def main(which):
    os.makedirs("results", exist_ok=True)
    d = Data()
    truth = ground_truth(d, SPLITS["val"][1])
    ev = lambda preds: mapk(truth, {u: preds[u] for u in truth})
    rows = []
    t_all = time.time()
    cache = {}

    def get(lam):
        if lam not in cache:
            cache[lam] = setup(d, "val", lam)
        return cache[lam]

    if which == "baselines":
        R, items, pool, fb, active = get(0.0)
        rows.append({"method": "popularity", "map5": ev(rank_top5(popularity_scores(R), items, pool, fb,
                                                                   np.ones_like(active)))})
        rows.append({"method": "personal_frequency", "map5": ev(rank_top5(R, items, pool, fb, active))})
    elif which == "ppr":
        grid = list(itertools.product([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85], ["raw", "log"], [0.0]))
        grid += list(itertools.product([0.2, 0.3, 0.4], ["raw", "log"], [0.05, 0.1]))
        for alpha, tr, lam in grid:
            R, items, pool, fb, active = get(lam)
            t = time.time()
            S, info = ppr_scores(R, alpha=alpha, transform=tr)
            rows.append({"alpha": alpha, "transform": tr, "lam": lam, "map5": ev(rank_top5(S, items, pool, fb, active)),
                         "seconds": round(time.time() - t, 2), **info})
            print(rows[-1], flush=True)
    elif which.startswith("mf"):
        # mf1: target x factors x reg at conf_a=1; mf2: reg x confidence around the mf1 winner
        if which == "mf1":
            grid = [("binary", 32, 1e-2, 10.0, 300)]
            grid += list(itertools.product(["log"], [16, 32, 64], [1e-2, 3e-2, 1e-1, 3e-1], [1.0], [300]))
        else:
            f0 = int(sys.argv[2])
            grid = list(itertools.product(["log"], [f0], [5e-3, 1e-2, 2e-2], [0.0, 1.0, 3.0], [300]))
        for target, f, reg, a, steps in grid:
            R, items, pool, fb, active = get(0.0)
            t = time.time()
            S, info = mf_scores(R, factors=f, conf_a=a, reg=reg, steps=steps, target=target)
            rows.append({"target": target, "factors": f, "conf_a": a, "reg": reg, "steps": steps, "lr": 0.05,
                         "seed": 42,
                         "map5": ev(rank_top5(S, items, pool, fb, active, min_score=-np.inf)),
                         "final_loss": info["final_loss"], "seconds": round(time.time() - t, 2)})
            print(rows[-1], flush=True)
    best = max(rows, key=lambda r: r["map5"])
    print("BEST", best, f"total {time.time() - t_all:.1f}s")
    with open(f"results/val_{which}.json", "w") as fh:
        json.dump({"split": "history 11-20 -> target 21-30", "n_eval_users": len(truth), "rows": rows,
                   "best": best}, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])

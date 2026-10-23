"""Third bounded round (validation): w_MF x beta with the global schedule ratio, and a per-user
schedule ratio restricted to the channels the user watched in the history window."""
import itertools
import json

import numpy as np

from common import SPLITS, Data, ground_truth, mapk
from hybrid import MF_CFG, PPR_CFG, fuse
from methods import mf_scores, ppr_scores, rank_top5
from rerank import schedule_ratio, user_channel_ratio
from tune import setup


if __name__ == "__main__":
    d = Data()
    hist, target = SPLITS["val"]
    truth = ground_truth(d, target)
    R0, items, pool, fb, active = setup(d, "val", 0.0)
    R_ppr, *_ = setup(d, "val", PPR_CFG["lam"])
    S_ppr, _ = ppr_scores(R_ppr, alpha=PPR_CFG["alpha"], transform=PPR_CFG["transform"])
    S_mf, _ = mf_scores(R0, **MF_CFG)

    def ev(S):
        top = rank_top5(S, items, pool, fb, active)
        return mapk(truth, {u: top[u] for u in truth})

    rg = schedule_ratio(d, items, hist, target)
    ru = user_channel_ratio(d, items, hist, target)
    rows = []
    for w, beta in itertools.product([0.5, 0.6, 0.7, 0.8, 0.9], [0.3, 0.4, 0.5]):
        H = fuse(S_ppr, S_mf, items, pool, w, "score")
        rows.append({"ratio": "global", "w_mf": w, "beta": beta, "map5": ev(H * rg[None, :] ** beta + 1e-12)})
        print(rows[-1], flush=True)
    H = fuse(S_ppr, S_mf, items, pool, 0.8, "score")
    for beta in [0.1, 0.2, 0.3, 0.4, 0.5]:
        rows.append({"ratio": "user_channels", "w_mf": 0.8, "beta": beta, "map5": ev(H * ru ** beta + 1e-12)})
        print(rows[-1], flush=True)
    print("BEST", max(rows, key=lambda x: x["map5"]))
    with open("results/val_improve3.json", "w") as fh:
        json.dump(rows, fh, indent=1)

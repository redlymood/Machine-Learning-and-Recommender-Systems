"""Fourth bounded round: (channel, hour) slot ratio vs channel ratio, on val and the val2 robustness split."""
import json
import sys

import numpy as np

from common import SPLITS, Data, ground_truth, mapk
from hybrid import MF_CFG, PPR_CFG, fuse
from methods import mf_scores, ppr_scores, rank_top5
from rerank import schedule_ratio, user_channel_ratio, user_slot_ratio
from tune import setup

d = Data()
rows = []
for split in sys.argv[1:] or ["val"]:
    hist, target = SPLITS[split]
    truth = ground_truth(d, target)
    R0, items, pool, fb, active = setup(d, split, 0.0)
    R_ppr, *_ = setup(d, split, PPR_CFG["lam"])
    S_ppr, _ = ppr_scores(R_ppr, alpha=PPR_CFG["alpha"], transform=PPR_CFG["transform"])
    S_mf, _ = mf_scores(R0, **MF_CFG)
    H = fuse(S_ppr, S_mf, items, pool, 0.8, "score")

    def ev(S, ms=0.0):
        top = rank_top5(S, items, pool, fb, active, min_score=ms)
        return mapk(truth, {u: top[u] for u in truth})

    base = {"popularity": ev(np.broadcast_to(R0.sum(0), R0.shape).copy()), "ppr": ev(S_ppr),
            "mf": ev(S_mf, -np.inf), "hybrid": ev(H)}
    rows.append({"split": split, "n_users": len(truth), **base}); print(rows[-1], flush=True)
    ratios = {"global": schedule_ratio(d, items, hist, target)[None, :],
              "user_channels": user_channel_ratio(d, items, hist, target),
              "slot_binary": user_slot_ratio(d, items, hist, target, "binary"),
              "slot_time": user_slot_ratio(d, items, hist, target, "time")}
    for name, r in ratios.items():
        for beta in [0.2, 0.3, 0.4, 0.5]:
            rows.append({"split": split, "ratio": name, "beta": beta, "map5": ev(H * r ** beta + 1e-12)})
            print(rows[-1], flush=True)
with open("results/val_improve4.json", "w") as fh:
    json.dump(rows, fh, indent=1)

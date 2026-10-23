"""Refit every method on history weeks 11-30 with the validation-selected settings and write
submissions/<method>.csv in the template's user order (pool = labelled shows airing in weeks 31-42).
final = hybrid re-ranked with the user-channel schedule ratio (rerank.py).
"""
import json
import os
import time

import numpy as np
import pandas as pd

from common import K, SPLITS, Data
from hybrid import MF_CFG, PPR_CFG, fuse
from methods import mf_scores, popularity_scores, ppr_scores, rank_top5
from rerank import user_channel_ratio
from tune import setup

W_MF = 0.8   # hybrid weight selected on validation (results/val_hybrid.json)
BETA = 0.3   # schedule re-ranking exponent selected on validation (results/val_improve3.json)


def write_submission(path, test_users, test_idx, top, pool):
    rows = []
    pool_set = set(int(s) for s in pool)
    for uid, u in zip(test_users, test_idx):
        rec = top[u]
        assert len(rec) == K and len(set(rec)) == K and all(s in pool_set for s in rec), (uid, rec)
        rows.append((str(uid), " ".join(str(s) for s in rec)))
    pd.DataFrame(rows, columns=["user_id", "tv_show_id"]).to_csv(path, index=False)


def main():
    t0 = time.time()
    os.makedirs("submissions", exist_ok=True)
    d = Data()
    tmpl = pd.read_csv("data/submission.csv", dtype={"user_id": "uint64"})
    test_users = tmpl["user_id"].to_numpy()
    test_idx = np.searchsorted(d.users, test_users)
    assert np.all(d.users[test_idx] == test_users)

    timing = {}
    R0, items, pool, fb, active = setup(d, "test", 0.0)
    R_ppr, items_p, *_ = setup(d, "test", PPR_CFG["lam"])
    assert np.array_equal(items, items_p)

    t = time.time(); top_pop = rank_top5(popularity_scores(R0), items, pool, fb, np.ones_like(active))
    timing["popularity"] = time.time() - t
    t = time.time(); S_ppr, info_ppr = ppr_scores(R_ppr, alpha=PPR_CFG["alpha"], transform=PPR_CFG["transform"])
    top_ppr = rank_top5(S_ppr, items, pool, fb, active); timing["ppr"] = time.time() - t
    t = time.time(); S_mf, info_mf = mf_scores(R0, **MF_CFG)
    top_mf = rank_top5(S_mf, items, pool, fb, active, min_score=-np.inf); timing["mf"] = time.time() - t
    t = time.time(); H = fuse(S_ppr, S_mf, items, pool, W_MF, "score")
    top_hyb = rank_top5(H, items, pool, fb, active); timing["hybrid_fusion"] = time.time() - t
    hist, target = SPLITS["test"]
    t = time.time(); ratio = user_channel_ratio(d, items, hist, target)
    top_fin = rank_top5(H * ratio ** BETA + 1e-12, items, pool, fb, active); timing["final_rerank"] = time.time() - t

    tops = {"popularity": top_pop, "ppr": top_ppr, "mf": top_mf, "hybrid": top_hyb, "final": top_fin}
    for name, top in tops.items():
        write_submission(f"submissions/{name}.csv", test_users, test_idx, top, pool)
    np.savez("results/test_scores.npz", S_ppr=S_ppr, S_mf=S_mf, items=items, pool=pool,
             fallback=np.array(fb), active=active)
    with open("results/test_lists.json", "w") as fh:
        json.dump({n: {str(u): t[u] for u in test_idx} for n, t in tops.items()}, fh)

    hist_rows = R0[test_idx].sum(axis=1)
    info = {"history_weeks": "11-30", "items_scored": int(len(items)), "pool_size": int(len(pool)),
            "test_users_without_history": int((hist_rows == 0).sum()),
            "ppr": {**PPR_CFG, **info_ppr}, "mf": {**MF_CFG, "final_loss": info_mf["final_loss"]},
            "hybrid": {"kind": "score", "w_mf": W_MF},
            "final": {"base": "hybrid", "rerank": "user_channel_ratio", "beta": BETA}, "seconds": {k: round(v, 2) for k, v in timing.items()},
            "total_seconds": round(time.time() - t0, 1)}
    with open("results/test_run.json", "w") as fh:
        json.dump(info, fh, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()

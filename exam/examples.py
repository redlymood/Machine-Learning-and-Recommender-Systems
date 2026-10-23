"""Two example users on the validation split (history 11-20 -> target 21-30), every method's Top-5
with id, title and category, next to the user's actual top-5 and history. Writes results/examples.json.

Choice rule (fixed before looking at any list): among evaluable validation users who are also Kaggle
test users, 'heavy' = closest to the 90th percentile of qualifying airings in the history window,
'light' = closest to the 10th percentile; ties -> smaller user index.
"""
import json

import numpy as np
import pandas as pd

from common import SPLITS, Data, apk, ground_truth
from hybrid import MF_CFG, PPR_CFG, fuse
from methods import mf_scores, popularity_scores, ppr_scores, rank_top5
from rerank import user_channel_ratio
from tune import setup

d = Data()
hist, target = SPLITS["val"]
truth = ground_truth(d, target)
R0, items, pool, fb, active = setup(d, "val", 0.0)
R_ppr, *_ = setup(d, "val", PPR_CFG["lam"])
S_ppr, _ = ppr_scores(R_ppr, alpha=PPR_CFG["alpha"], transform=PPR_CFG["transform"])
S_mf, _ = mf_scores(R0, **MF_CFG)
H = fuse(S_ppr, S_mf, items, pool, 0.8, "score")
F = H * user_channel_ratio(d, items, hist, target) ** 0.3 + 1e-12
lists = {"popularity": rank_top5(popularity_scores(R0), items, pool, fb, np.ones_like(active)),
         "ppr": rank_top5(S_ppr, items, pool, fb, active),
         "mf": rank_top5(S_mf, items, pool, fb, active, min_score=-np.inf),
         "hybrid": rank_top5(H, items, pool, fb, active),
         "final": rank_top5(F, items, pool, fb, active)}

test_users = pd.read_csv("data/submission.csv", dtype={"user_id": "uint64"})["user_id"].to_numpy()
test_idx = set(np.searchsorted(d.users, test_users).tolist())
cand = sorted(u for u in truth if u in test_idx and R0[u].sum() > 0)
tot = np.array([R0[u].sum() for u in cand])
pick = {}
for name, q in [("heavy", 90), ("light", 10)]:
    target_v = np.percentile(tot, q)
    pick[name] = cand[int(np.argmin(np.abs(tot - target_v)))]


def show(s):
    r = d.shows.loc[s]
    return {"id": int(s), "title": r["tv_show_title"], "category": r["tv_show_category"]}


out = {"rule": __doc__.split("Choice rule")[1].strip(), "users": {}}
for name, u in pick.items():
    seen = R0[u]
    hist_top = [(int(items[j]), float(seen[j])) for j in np.argsort(-seen, kind="stable")[:8] if seen[j] > 0]
    tc = d.counts(target); tc = tc[tc["user"] == u].sort_values(["n", "show"], ascending=[False, True])
    rec = {}
    for m, top in lists.items():
        l = top[u]
        rec[m] = {"top5": [dict(show(s), hit=s in truth[u], in_history=bool(seen[np.searchsorted(items, s)] > 0)
                                if s in set(items.tolist()) else False) for s in l],
                  "ap5": apk(set(truth[u]), l)}
    out["users"][name] = {
        "user_idx": int(u), "user_id": str(d.users[u]),
        "history_qualifying_airings": float(seen.sum()), "history_distinct_shows": int((seen > 0).sum()),
        "history_top": [dict(show(s), n=n) for s, n in hist_top],
        "actual_top5_weeks21_30": [dict(show(s), n=int(tc[tc['show'] == s]['n'].iloc[0])) for s in truth[u]],
        "recommendations": rec}
json.dump(out, open("results/examples.json", "w"), ensure_ascii=False, indent=1)
for name, x in out["users"].items():
    print(f"== {name}: user_idx {x['user_idx']} id {x['user_id']} | history {x['history_qualifying_airings']:.0f} qualifying airings, {x['history_distinct_shows']} shows")
    print("   actual weeks 21-30:", [(a['id'], a['title'][:28], a['n']) for a in x["actual_top5_weeks21_30"]])
    print("   history top:", [(a['id'], a['title'][:28], a['n']) for a in x["history_top"][:6]])
    for m, r in x["recommendations"].items():
        print(f"   {m:10s} AP@5={r['ap5']:.4f}", [(a['id'], a['title'][:22], a['category'], 'HIT' if a['hit'] else '', 'H' if a['in_history'] else 'new') for a in r['top5']])

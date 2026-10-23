"""Where is the headroom? Truth items seen in history, oracle bounds, hit composition (validation)."""
import json
import numpy as np
from common import SPLITS, Data, ground_truth, mapk, apk
from hybrid import MF_CFG, PPR_CFG, fuse
from methods import mf_scores, ppr_scores, rank_top5
from rerank import user_channel_ratio
from tune import setup

d = Data(); hist, target = SPLITS["val"]; truth = ground_truth(d, target)
R0, items, pool, fb, active = setup(d, "val", 0.0)
R_ppr, *_ = setup(d, "val", PPR_CFG["lam"])
S_ppr, _ = ppr_scores(R_ppr, alpha=PPR_CFG["alpha"], transform=PPR_CFG["transform"])
S_mf, _ = mf_scores(R0, **MF_CFG)
H = fuse(S_ppr, S_mf, items, pool, 0.8, "score")
F = H * user_channel_ratio(d, items, hist, target) ** 0.3 + 1e-12
seen = {u: set(items[R0[u] > 0].tolist()) for u in truth}
n_t = sum(len(t) for t in truth.values())
in_hist = sum(len(set(t) & seen[u]) for u, t in truth.items())
# oracle 1: perfect recall of the truth items that appear in history, in truth order; rest empty
oracle_hist = mapk(truth, {u: [s for s in t if s in seen[u]] + [-1] * 5 for u, t in truth.items()})
out = {"truth_items": n_t, "truth_items_in_history_share": in_hist / n_t, "oracle_history_only_map5": oracle_hist}
for name, S in [("popularity", np.broadcast_to(R0.sum(0), R0.shape).copy()), ("ppr", S_ppr), ("mf", S_mf), ("hybrid", H), ("final", F)]:
    top = rank_top5(S, items, pool, fb, active, min_score=-np.inf if name == "mf" else 0.0)
    rec = {u: top[u] for u in truth}
    hits = [(s in set(truth[u]), s in seen[u]) for u in truth for s in rec[u]]
    hits_rep = sum(h and r for h, r in hits); hits_new = sum(h and not r for h, r in hits)
    rec_seen = sum(r for _, r in hits) / len(hits)
    out[name] = {"map5": mapk(truth, rec), "hits_on_history_items": hits_rep, "hits_on_new_items": hits_new,
                 "share_of_recs_from_history": rec_seen, "precision@5": (hits_rep + hits_new) / len(hits)}
print(json.dumps(out, indent=1))
json.dump(out, open("results/val_diagnose.json", "w"), indent=1)

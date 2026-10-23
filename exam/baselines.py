"""Data facts for the split design + popularity and personal-frequency baselines on validation."""
import time

import numpy as np
import pandas as pd

from common import SPLITS, Data, ground_truth, mapk, top_k_with_fill

t0 = time.time()
d = Data()
sub = pd.read_csv("data/submission.csv", dtype={"user_id": "uint64"})
test_idx = np.searchsorted(d.users, sub["user_id"].to_numpy())
assert np.all(d.users[test_idx] == sub["user_id"].to_numpy())

for name, (hist, target) in SPLITS.items():
    c = d.counts(hist)
    has = set(c["user"].unique())
    print(f"[{name}] history weeks {hist.start}-{hist.stop - 1}: users with >=1 qualifying view {len(has):,}; "
          f"test users with history {sum(u in has for u in test_idx)}/{len(test_idx)}; "
          f"pool size {len(d.pool(target)):,}")

hist, target = SPLITS["val"]
truth = ground_truth(d, target)
print(f"[val] evaluable users (>=5 distinct qualifying shows in weeks 21-30): {len(truth):,}; "
      f"of them test users: {len(set(truth) & set(test_idx))}")

c = d.counts(hist)
pool = d.pool(target)
c = c[c["show"].isin(pool)]

# Popularity: total qualifying airings per show in the history window, pool only.
pop = c.groupby("show")["n"].sum().sort_values(ascending=False)
pop_list = pop.index.tolist()
# Alternative popularity: number of distinct users with a qualifying view.
pop_u = c.groupby("show")["user"].nunique().sort_values(ascending=False).index.tolist()

preds_pop = {u: pop_list[:5] for u in truth}
preds_popu = {u: pop_u[:5] for u in truth}

# Personal frequency (repeats allowed): user's own counts, ties by global popularity, fill by popularity.
rank_pop = pd.Series(np.arange(len(pop_list)), index=pop_list)
c["pr"] = c["show"].map(rank_pop)
c = c.sort_values(["user", "n", "pr"], ascending=[True, False, True])
own = c.groupby("user")["show"].apply(list).to_dict()
preds_own = {u: top_k_with_fill(own.get(u, []), pop_list) for u in truth}

print(f"[val] MAP@5 popularity (qualifying airings): {mapk(truth, preds_pop):.5f}")
print(f"[val] MAP@5 popularity (distinct users):     {mapk(truth, preds_popu):.5f}")
print(f"[val] MAP@5 personal frequency + pop fill:   {mapk(truth, preds_own):.5f}")
tu = {u: t for u, t in truth.items() if u in set(test_idx)}
print(f"[val, test-user subset n={len(tu)}] popularity {mapk(tu, preds_pop):.5f}, personal {mapk(tu, preds_own):.5f}")
print(f"done in {time.time() - t0:.1f}s")

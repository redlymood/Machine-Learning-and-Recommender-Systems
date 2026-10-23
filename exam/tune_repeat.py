"""Validation sweep for the schedule-aware personal-frequency score.

score(u, s) = n_w(u, s) / A_hist(s)^beta * A_target(s)^beta
  n_w      : user's qualifying airings of s in the history window, weight exp(-lam * age_in_weeks)
  A_hist   : airings of s in the history window (schedule), same recency weights
  A_target : airings of s in the target window (schedule only)
"""
import itertools
import time

import numpy as np
import pandas as pd

from common import SPLITS, Data, ground_truth, mapk, top_k_with_fill


def repeat_scores(d, hist, target, lam, beta):
    last = hist.stop - 1
    wfun = lambda w: np.exp(-lam * (last - w))
    c = d.counts(hist, weight=wfun)
    pool = d.pool(target)
    c = c[c["show"].isin(pool)].copy()
    if beta:
        m = np.isin(d.a["week"], np.asarray(hist))
        ah = pd.Series(wfun(d.a["week"][m])).groupby(d.a["show"][m]).sum()
        at = d.airing_counts(target)
        c["n"] = c["n"] * (at.reindex(c["show"]).to_numpy() / ah.reindex(c["show"]).to_numpy()) ** beta
    return c


def popularity(d, hist, target):
    c = d.counts(hist)
    c = c[c["show"].isin(d.pool(target))]
    return c.groupby("show")["n"].sum().sort_values(ascending=False).index.tolist()


def rank_lists(c, pop_list):
    rank_pop = pd.Series(np.arange(len(pop_list)), index=pop_list)
    c = c.assign(pr=c["show"].map(rank_pop)).sort_values(["user", "n", "pr"], ascending=[True, False, True])
    return c.groupby("user")["show"].apply(list).to_dict()


if __name__ == "__main__":
    t0 = time.time()
    d = Data()
    hist, target = SPLITS["val"]
    truth = ground_truth(d, target)
    pop_list = popularity(d, hist, target)
    rows = []
    for lam, beta in itertools.product([0.0, 0.05, 0.1, 0.2, 0.3, 0.5], [0.0, 0.5, 1.0]):
        own = rank_lists(repeat_scores(d, hist, target, lam, beta), pop_list)
        preds = {u: top_k_with_fill(own.get(u, []), pop_list) for u in truth}
        rows.append((lam, beta, mapk(truth, preds)))
        print(f"lam={lam:<4} beta={beta:<3} MAP@5={rows[-1][2]:.5f}")
    best = max(rows, key=lambda r: r[2])
    print(f"best lam={best[0]} beta={best[1]} MAP@5={best[2]:.5f}  ({time.time() - t0:.1f}s)")

"""Shared data access, the qualifying-view rule, ground truth construction and MAP@5."""
import numpy as np
import pandas as pd

K = 5
SCREEN_MIN = 0.8  # the competition's own "high screen time" cut (data description page)

# Splits: (history weeks, target weeks). Validation mirrors the test layout inside weeks 11-30.
SPLITS = {
    "val": (range(11, 21), range(21, 31)),
    "test": (range(11, 31), range(31, 43)),
    "val2": (range(11, 25), range(25, 31)),  # robustness check only (later cut, shorter target)
}


class Data:
    def __init__(self):
        p = dict(np.load("cache/pairs.npz"))
        self.a = dict(np.load("cache/airings.npz"))
        v = np.load("cache/views.npz")
        self.users = v["users"]                       # uint64 user ids, index = user_idx
        q = p["screen"] >= SCREEN_MIN
        self.q = {k: x[q] for k, x in p.items()}      # qualifying (user, airing) pairs only
        self.shows = pd.read_csv("cache/shows.csv", index_col="tv_show_id", keep_default_na=False)
        self.n_users = len(self.users)

    def counts(self, weeks, weight=None):
        """DataFrame (user, show, n): number of qualifying airings per user and show in `weeks`.
        `weight` maps a week array to per-row weights (recency); default 1 per airing."""
        m = np.isin(self.q["week"], np.asarray(weeks))
        w = np.ones(m.sum()) if weight is None else weight(self.q["week"][m])
        df = pd.DataFrame({"user": self.q["user"][m], "show": self.q["show"][m], "n": w})
        return df.groupby(["user", "show"], as_index=False)["n"].sum()

    def pool(self, weeks):
        """Candidate pool: labelled shows with at least one airing in `weeks` (schedule only)."""
        return np.unique(self.a["show"][np.isin(self.a["week"], np.asarray(weeks))])

    def airing_counts(self, weeks):
        m = np.isin(self.a["week"], np.asarray(weeks))
        s, c = np.unique(self.a["show"][m], return_counts=True)
        return pd.Series(c, index=s)


def ground_truth(data, weeks):
    """Competition rule applied to `weeks`: per user the 5 most frequent qualifying shows,
    users with fewer than 5 distinct qualifying shows are not evaluable.
    Ties at the cut are broken by smaller show id (the hosts' tie rule is not published)."""
    c = data.counts(weeks)
    c = c.sort_values(["user", "n", "show"], ascending=[True, False, True])
    c["rank"] = c.groupby("user").cumcount()
    top = c[c["rank"] < K]
    sizes = top.groupby("user").size()
    ok = sizes.index[sizes == K]
    top = top[top["user"].isin(ok)]
    return {u: g["show"].tolist() for u, g in top.groupby("user")}


def apk(actual, predicted, k=K):
    """Kaggle-style AP@k (same formula as ml_metrics.apk)."""
    predicted = list(predicted)[:k]
    score, hits = 0.0, 0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            hits += 1
            score += hits / (i + 1.0)
    return score / min(len(actual), k) if actual else 0.0


def mapk(truth, preds, k=K):
    return float(np.mean([apk(set(t), preds[u], k) for u, t in truth.items()]))


def top_k_with_fill(scored, fallback, k=K):
    """scored: list of show ids already in rank order; fill to k from `fallback` without duplicates."""
    out = list(dict.fromkeys(scored))[:k]
    for s in fallback:
        if len(out) == k:
            break
        if s not in out:
            out.append(s)
    return out

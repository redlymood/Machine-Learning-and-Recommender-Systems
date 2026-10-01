"""Reference implementation of week3 CF (same formulas as script.js) for cross-checking.

Strategy C: co-rated cosine weighted by min(common, beta) / beta, beta = 50.
User-Based with N=20, Item-Based with K=20; Top-5 needs support >= MIN_SUPPORT and
at least MIN_ITEM_RATINGS ratings per movie.
Tie-breaking matches script.js: neighbours by (weighted sim desc, id asc), recommendations
by (Math.round(score * 1e6) desc, support desc, id asc). Similarity sums are integer-valued,
so they are exact in float64 and the similarities are bit-identical to the browser.

Run: python3 check_cf.py
"""
import math
import time
from pathlib import Path

import numpy as np

N_NEIGHBORS = 20
K_NEIGHBORS = 20
MIN_SUPPORT = 3
MIN_ITEM_RATINGS = 20
SIGNIFICANCE_BETA = 50

HERE = Path(__file__).resolve().parent


def load():
    titles = {}
    for line in (HERE / "u.item").read_bytes().decode("latin-1").splitlines():
        fields = line.split("|")
        titles[int(fields[0])] = fields[1]
    data = np.loadtxt(HERE / "u.data", dtype=np.int64)
    num_users, num_movies = int(data[:, 0].max()), len(titles)
    R = np.zeros((num_users + 1, num_movies + 1))
    R[data[:, 0], data[:, 1]] = data[:, 2]
    return R, titles


def corated_cosine(M):
    """Pairwise co-rated cosine between the rows of M. Returns (sim, common)."""
    B = (M > 0).astype(float)
    S = M * M
    dot = M @ M.T
    norm_a = S @ B.T          # sum of a_k^2 over k co-rated with b
    norm_b = B @ S.T          # sum of b_k^2 over k co-rated with a
    common = B @ B.T
    denom = np.sqrt(norm_a) * np.sqrt(norm_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(denom == 0, 0.0, dot / denom)
    return sim, common.astype(int)


def weighted(sim, common):
    """sim * min(common, beta) / beta, same operation order as weightedSimilarity in script.js."""
    return sim * np.minimum(common, SIGNIFICANCE_BETA) / SIGNIFICANCE_BETA


def js_round(x):
    """Math.round: half rounds up (Python's round() would round half to even)."""
    return math.floor(x + 0.5)


def weighted_average(used, rating_of):
    num = den = 0.0
    for nid, sim in used:          # same order as script.js
        num += sim * rating_of(nid)
        den += sim
    return num / den


class CF:
    def __init__(self, R):
        self.R = R
        self.num_users = R.shape[0] - 1
        self.num_movies = R.shape[1] - 1
        t0 = time.perf_counter()
        self.user_raw, self.user_common = corated_cosine(R)
        self.item_raw, self.item_common = corated_cosine(R.T)
        self.user_sim = weighted(self.user_raw, self.user_common)
        self.item_sim = weighted(self.item_raw, self.item_common)
        self.item_ratings = (R > 0).sum(0)
        self.sim_time = time.perf_counter() - t0
        self._neighbors = {}

    def neighbors(self, u):
        if u not in self._neighbors:
            ids = [v for v in range(1, self.num_users + 1) if v != u and self.user_sim[u, v] > 0]
            ids.sort(key=lambda v: (-self.user_sim[u, v], v))
            self._neighbors[u] = [(v, self.user_sim[u, v]) for v in ids]
        return self._neighbors[u]

    def predict_user_based(self, u, i, N=N_NEIGHBORS):
        if not (self.R[:, i] > 0).any():
            return None, 0, 0.0, "nobody rated the movie"
        used = []
        for v, s in self.neighbors(u):
            if self.R[v, i] > 0:
                used.append((v, s))
                if len(used) == N:
                    break
        if not used:
            return None, 0, 0.0, "no similar user rated the movie"
        avg_common = float(np.mean([self.user_common[u, v] for v, _ in used]))
        return weighted_average(used, lambda v: self.R[v, i]), len(used), avg_common, None

    def predict_item_based(self, u, i, K=K_NEIGHBORS):
        if not (self.R[:, i] > 0).any():
            return None, 0, 0.0, "nobody rated the movie"
        rated = [j for j in np.nonzero(self.R[u] > 0)[0] if j != i]
        similar = [(int(j), self.item_sim[i, j]) for j in rated if self.item_sim[i, j] > 0]
        if not similar:
            return None, 0, 0.0, "no rated movie is similar"
        similar.sort(key=lambda x: (-x[1], x[0]))
        used = similar[:K]
        avg_common = float(np.mean([self.item_common[i, j] for j, _ in used]))
        return weighted_average(used, lambda j: self.R[u, j]), len(used), avg_common, None

    def recommend(self, u, predict, top_k=5):
        candidates = []
        for i in range(1, self.num_movies + 1):
            if self.R[u, i] > 0 or self.item_ratings[i] < MIN_ITEM_RATINGS:
                continue
            score, support, _, _ = predict(u, i)
            if score is None or support < MIN_SUPPORT:
                continue
            candidates.append((i, score, support))
        candidates.sort(key=lambda c: (-js_round(c[1] * 1e6), -c[2], c[0]))
        return candidates[:top_k], candidates


def fmt(score):
    return "null" if score is None else f"{score:.4f} (shown as {score:.1f})"


def main():
    R, titles = load()
    cf = CF(R)
    print(f"users={cf.num_users} movies={cf.num_movies} ratings={int((R > 0).sum())}")
    print(f"full similarity matrices (reference only): {cf.sim_time:.2f} s")
    nan_count = int(np.isnan(cf.user_sim).sum() + np.isnan(cf.item_sim).sum())
    print(f"NaN in similarities: {nan_count}")

    print("\n=== Single predictions ===")
    for u, i in [(1, 1), (1, 300), (13, 50), (405, 1500)]:
        actual = int(R[u, i])
        print(f"\nuser {u}, movie {i} \"{titles[i]}\" ({cf.item_ratings[i]} ratings) — actual rating: "
              f"{actual if actual else 'not rated'}")
        for name, predict in [("User-Based", cf.predict_user_based), ("Item-Based", cf.predict_item_based)]:
            score, support, avg_common, reason = predict(u, i)
            print(f"  {name}: {fmt(score)}, support {support}, avg common {avg_common:.1f}"
                  + (f", reason: {reason}" if reason else ""))

    for user in (1, 405):
        for name, predict in [("User-Based", cf.predict_user_based), ("Item-Based", cf.predict_item_based)]:
            t0 = time.perf_counter()
            top, all_candidates = cf.recommend(user, predict)
            elapsed = time.perf_counter() - t0
            print(f"\n=== {name} Top-5 for user {user} ({elapsed:.2f} s, {len(all_candidates)} candidates) ===")
            for rank, (i, score, support) in enumerate(top, 1):
                print(f"  {rank}. [{i}] {titles[i]} — score {score:.6f}, support {support}, "
                      f"{cf.item_ratings[i]} ratings")
            if len(all_candidates) > 5:
                i, score, support = all_candidates[5]
                print(f"  (6th: [{i}] {titles[i]} — score {score:.6f}, support {support})")

    print("\n=== Diagnostics for user 1 ===")
    nb = cf.neighbors(1)
    print("top-20 neighbours (id, weighted sim, common):",
          [(v, round(float(s), 4), int(cf.user_common[1, v])) for v, s in nb[:N_NEIGHBORS]])


if __name__ == "__main__":
    main()

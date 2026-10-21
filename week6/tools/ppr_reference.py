"""Independent float64 numpy reference for HW6 global PageRank and PPR.

Builds the graph from week6/u.data with this folder's own split code
(independent_stats.split; graph.js is not read), using the readme rules:
nodes = users and movies observed in TRAIN, R[u, i] = 1 iff train rating >= 4,
users first then movies, zero-degree nodes kept, and the readme 3.3 update
  p_next = alpha * (Q^T p + d v) + (1 - alpha) v,   p_0 = v.
Compares with the page vectors exported by graph_check.py
(results/page_vectors.json) by raw id, at alpha = 0.85:
  - "same rule": float64 with the page's stop rule (L1 < 1e-6, cap 500);
  - "fixed point": float64 iterated to L1 < 1e-14 (cap 20000).
Reports max |delta p| and whether the 10 highest movie-node scores (no
exclusions; ties by ascending id) are the same ids in the same order.

Usage: python3 ppr_reference.py   (writes results/ppr_reference.txt)
"""
import json
import numpy as np

from independent_stats import load, split
from paths import RESULTS

ALPHA = 0.85


def build_graph():
    ratings, _ = load()
    train, _, _, _ = split(ratings)
    users = sorted({r[0] for r in train})
    movies = sorted({r[1] for r in train})
    u_index = {u: k for k, u in enumerate(users)}
    m_index = {m: k for k, m in enumerate(movies)}
    R = np.zeros((len(users), len(movies)))
    for u, m, r, _ in train:
        if r >= 4.0:
            R[u_index[u], m_index[m]] = 1.0
    return users, movies, R


def iterate(R, v, alpha, tol, max_iter):
    n_u, n_m = R.shape
    deg_u, deg_m = R.sum(axis=1), R.sum(axis=0)
    inv_u = np.divide(1.0, deg_u, out=np.zeros_like(deg_u), where=deg_u > 0)
    inv_m = np.divide(1.0, deg_m, out=np.zeros_like(deg_m), where=deg_m > 0)
    dangling = np.concatenate([deg_u == 0, deg_m == 0])
    p = v.copy()
    for k in range(1, max_iter + 1):
        link = np.concatenate([R @ (p[n_u:] * inv_m), R.T @ (p[:n_u] * inv_u)])
        d = p[dangling].sum()
        nxt = alpha * (link + d * v) + (1 - alpha) * v
        res = np.abs(nxt - p).sum()
        p = nxt
        if res < tol:
            return p, k, res, True
    return p, max_iter, res, False


def top10_movies(p, movie_ids, n_users):
    scores = p[n_users:]
    order = sorted(range(len(movie_ids)), key=lambda i: (-scores[i], movie_ids[i]))
    return [movie_ids[i] for i in order[:10]], [scores[i] for i in order[:11]]


def main():
    users, movies, R = build_graph()
    n_u, n_m = R.shape
    n = n_u + n_m
    page = json.loads((RESULTS / "page_vectors.json").read_text())
    assert page["alpha"] == ALPHA
    # Map the page's node order onto ours by raw id (independent of graph.js internals).
    page_order = ([users.index(u) for u in page["userIds"]] +
                  [n_u + movies.index(m) for m in page["movieIds"]])
    assert sorted(page_order) == list(range(n)), "page node universe differs from the reference"

    runs = {"pagerank": np.full(n, 1.0 / n)}
    for u in ("1", "98"):
        v = np.zeros(n)
        v[users.index(int(u))] = 1.0
        runs["ppr user " + u] = v

    lines = [f"float64 reference vs page ({page['backend']}), alpha {ALPHA}; graph {n_u} users + {n_m} movies, "
             f"{int(R.sum())} positive edges",
             "run | page it / ref it (same rule) | ref it (fixed point) | max|dp| vs same rule | "
             "max|dp| vs fixed point | |sum-1| page | top-10 movies same ids+order"]
    for name, v in runs.items():
        got_raw = page["pagerank"] if name == "pagerank" else page["ppr"][name.split()[-1]]
        got = np.zeros(n)
        got[page_order] = np.asarray(got_raw["p"])
        same, k_same, _, _ = iterate(R, v, ALPHA, 1e-6, 500)
        exact, k_exact, res_exact, ok = iterate(R, v, ALPHA, 1e-14, 20000)
        top_page, s_page = top10_movies(got, movies, n_u)
        top_ref, s_ref = top10_movies(exact, movies, n_u)
        gaps = np.diff(-np.asarray(s_ref))  # score gaps between ranks 1..11 of the reference
        lines.append(
            f"{name} | {got_raw['iterations']} / {k_same} | {k_exact} (res {res_exact:.1e}, conv {ok}) | "
            f"{np.abs(got - same).max():.3e} | {np.abs(got - exact).max():.3e} | {abs(got.sum() - 1):.2e} | "
            f"{top_page == top_ref}")
        lines.append(f"    top-10 page {top_page}")
        lines.append(f"    top-10 ref  {top_ref}  (smallest gap between ranks 1-11: {gaps.min():.3e})")
    text = "\n".join(lines) + "\n"
    (RESULTS / "ppr_reference.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

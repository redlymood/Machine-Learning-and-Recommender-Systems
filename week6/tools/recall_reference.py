"""Part D: independent float64 checks of the page's retrieval numbers (Python + numpy).

D1  Recall@10 for popularity, global PageRank and PPR at the selected alpha on
    the shared TEST cohort, from this folder's own split / graph code
    (hw6_data, ppr_reference.build_graph), compared user by user with the
    page's Top-10 lists and hits (results/page_lists.json). For lists that
    differ, the float64 score gap between the swapped movies shows whether it
    is a float32 near-tie.
D2  The users whose alpha = 0.95 PPR did not converge on the page
    (results/alpha095_page.json): float32 page vector vs float64 (same stop
    rule, and the fixed point), max |delta p| and Top-10 movie ids.
D3  Hand-check values for users 98 and 1 (PPR at the selected alpha, test).
Propagation: p_next = alpha * (Q^T p + d v) + (1 - alpha) v, p_0 = v, per-column
stop at L1 < tol (frozen afterwards), cap max_iter - the readme rule.

Usage: python3 recall_reference.py   (writes results/recall_reference.txt)
"""
import json

import numpy as np

from hw6_data import RESULTS, HW6Data, page_lists
from ppr_reference import build_graph


def batch_iterate(R, V, alpha, tol, max_iter):
    n_u = R.shape[0]
    deg_u, deg_m = R.sum(axis=1), R.sum(axis=0)
    inv_u = np.divide(1.0, deg_u, out=np.zeros_like(deg_u), where=deg_u > 0)
    inv_m = np.divide(1.0, deg_m, out=np.zeros_like(deg_m), where=deg_m > 0)
    dangling = np.concatenate([deg_u == 0, deg_m == 0])
    P = V.copy()
    active = np.ones(V.shape[1], dtype=bool)
    iterations = np.zeros(V.shape[1], dtype=int)
    residual = np.zeros(V.shape[1])
    for k in range(1, max_iter + 1):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        Pa, Va = P[:, idx], V[:, idx]
        link = np.vstack([R @ (Pa[n_u:] * inv_m[:, None]), R.T @ (Pa[:n_u] * inv_u[:, None])])
        nxt = alpha * link + (alpha * Pa[dangling].sum(axis=0) + 1 - alpha) * Va
        res = np.abs(nxt - Pa).sum(axis=0)
        P[:, idx], iterations[idx], residual[idx] = nxt, k, res
        active[idx[res < tol]] = False
    return P, iterations, residual, ~active


def compare_lists(name, data, page_test, ref_lists, ref_scores):
    page_lists_ = page_test["lists"][name]
    page_hits = {x["userId"]: x["hit"] for x in page_test["perUser"][name]}
    cohort = page_test["cohort"]
    diff_order = diff_set = diff_hits = 0
    gaps = []
    recalls = []
    for u in cohort:
        ref = ref_lists[u]
        got = page_lists_[str(u)]
        rel = data.relevant("test", u)
        hits = sum(m in rel for m in ref)
        recalls.append(hits / len(rel))
        if ref != got:
            diff_order += 1
        if set(ref) != set(got):
            diff_set += 1
            s = ref_scores(u)
            only_page, only_ref = set(got) - set(ref), set(ref) - set(got)
            gaps.append(max(abs(s[a] - s[b]) for a in only_page for b in only_ref))
        if hits != page_hits[u]:
            diff_hits += 1
    return np.mean(recalls), diff_order, diff_set, diff_hits, gaps


def main():
    data = HW6Data()
    page = page_lists()
    alpha = page["alpha"]
    test = page["test"]
    cohort = test["cohort"]
    assert cohort == data.cohort("test"), "page test cohort differs from the Python cohort"
    users, movies, R = build_graph()
    assert movies == data.pool
    n_u, n = len(users), len(users) + len(movies)
    u_index = {u: k for k, u in enumerate(users)}
    page_recall = {x["method"]: x["recall"] for x in test["audit"]}

    lines = [f"D1 float64 Recall@10 on the TEST cohort ({len(cohort)} users, == page cohort), alpha {alpha}",
             "method | page recall | float64 recall | users: list differs (order) | set differs | hits differ | "
             "max float64 gap of swapped movies"]
    # Popularity: integer counts.
    pop_lists = {u: data.top_k(data.popularity, u, "test")[0] for u in cohort}
    # Global PageRank, one uniform column.
    pr, pr_it, pr_res, pr_conv = batch_iterate(R, np.full((n, 1), 1.0 / n), alpha, 1e-6, 500)
    pr_scores = dict(zip(movies, pr[n_u:, 0]))
    pr_lists = {u: data.top_k(pr_scores, u, "test")[0] for u in cohort}
    # PPR, one column per cohort user.
    V = np.zeros((n, len(cohort)))
    V[[u_index[u] for u in cohort], np.arange(len(cohort))] = 1.0
    ppr, ppr_it, ppr_res, ppr_conv = batch_iterate(R, V, alpha, 1e-6, 500)
    col = {u: c for c, u in enumerate(cohort)}
    ppr_scores = lambda u: dict(zip(movies, ppr[n_u:, col[u]]))
    ppr_lists = {u: data.top_k(ppr_scores(u), u, "test")[0] for u in cohort}
    for name, lists, scores in (("popularity", pop_lists, lambda u: data.popularity),
                                ("pagerank", pr_lists, lambda u: pr_scores),
                                ("ppr", ppr_lists, ppr_scores)):
        rec, d_order, d_set, d_hits, gaps = compare_lists(name, data, test, lists, scores)
        lines.append(f"{name} | {page_recall[name]:.6f} | {rec:.6f} | {d_order} | {d_set} | {d_hits} | "
                     f"{max(gaps) if gaps else 0:.3e}")
    lines.append(f"  float64 iterations: pagerank {pr_it[0]} (converged {pr_conv[0]}); PPR min/median/max "
                 f"{ppr_it.min()}/{int(np.median(ppr_it))}/{ppr_it.max()}, not converged {(~ppr_conv).sum()}")

    # D2: alpha = 0.95 users that did not converge on the page.
    a95 = json.loads((RESULTS / "alpha095_page.json").read_text())
    assert a95["userIds"] == users and a95["movieIds"] == movies
    stuck = [x["user"] for x in a95["users"]]
    V = np.zeros((n, len(stuck)))
    V[[u_index[u] for u in stuck], np.arange(len(stuck))] = 1.0
    same, same_it, _, same_conv = batch_iterate(R, V, 0.95, 1e-6, 500)
    exact, exact_it, _, exact_conv = batch_iterate(R, V, 0.95, 1e-14, 20000)
    lines.append(f"D2 alpha 0.95, {len(stuck)} users not converged on the page (float32, residual "
                 f"{min(x['residual'] for x in a95['users']):.3e}..{max(x['residual'] for x in a95['users']):.3e})")
    worst_same = worst_exact = 0.0
    same_top = same_top_test = 0
    for c, x in enumerate(a95["users"]):
        got = np.asarray(x["p"])
        worst_same = max(worst_same, np.abs(got - same[:, c]).max())
        worst_exact = max(worst_exact, np.abs(got - exact[:, c]).max())
        top = lambda p: sorted(range(len(movies)), key=lambda i: (-p[n_u + i], movies[i]))[:10]
        same_top += top(got) == top(exact[:, c])
        sg = dict(zip(movies, got[n_u:]))
        se = dict(zip(movies, exact[n_u:, c]))
        same_top_test += data.top_k(sg, x["user"], "test")[0] == data.top_k(se, x["user"], "test")[0]
    lines.append(f"  float64 same rule: iterations {same_it.min()}..{same_it.max()}, all converged {same_conv.all()}; "
                 f"fixed point {exact_it.min()}..{exact_it.max()} iterations")
    lines.append(f"  max |delta p| page vs float64 same rule {worst_same:.3e}, vs fixed point {worst_exact:.3e}")
    lines.append(f"  identical Top-10 movie ids (no exclusions): {same_top}/{len(stuck)}; identical test Top-10 "
                 f"(train+val excluded): {same_top_test}/{len(stuck)}")

    # D3: hand check.
    lines.append(f"D3 hand check (PPR, alpha {alpha}, test exclusions)")
    for u in (98, 1):
        h = page["hand"][str(u)]
        _, ranking = data.top_k(ppr_scores(u), u, "test")
        ref_rank = {m: i + 1 for i, m in enumerate(ranking)}
        lines.append(f"  user {u}: {h['eligibleCandidates']} eligible candidates; page Recall@10 {h['recall']:.6f}; "
                     f"hits {h['hits']}; score at rank 10 / 11: {h['score10']:.6g} / {h['score11']:.6g}")
        for x in h["relevant"]:
            lines.append(f"    relevant {x['movieId']:>4} {x['title']} | page rank {x['rank']} | float64 rank "
                         f"{ref_rank[x['movieId']]} | score {x['score']:.6g} | pop {x['popularity']}")
    text = "\n".join(lines) + "\n"
    (RESULTS / "recall_reference.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

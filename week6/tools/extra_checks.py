"""Part A: extra checks for the HW6 analysis (Python float64, own split and graph code).

1  Cohort breakdown for test and validation (readme 7.1 / 7.2), from all users.
2  Global PageRank vs training popularity: Spearman correlation over the 1,611
   pool movies and mean common Top-10 (test lists) per alpha in the grid.
3  At the selected alpha, over the test cohort: mean list popularity, mean
   overlaps with PPR, and catalogue coverage per method (popularity, global
   PageRank, PPR, provided MF, HW5 MF).
4  Test Recall@10 by user activity: light / heavy split at the median number of
   positive training edges of the test cohort.
5  IMPROVE, validation only and outside the readme grid (not used for any
   decision or test number): PPR validation Recall@10 at alpha 0.10 and 0.30.
The provided MF is modelled as a constant score (it collapsed to mu; on the page
its lists equal the lowest-id eligible movies for all 861 test users). Only the
page's selected alpha and its exported results.csv are read, for comparison.

Usage: python3 extra_checks.py   (writes results/extra_checks.txt)
"""
import csv
import json
from collections import Counter

import numpy as np

from hw6_data import RESULTS, HW6Data, page_lists
from paths import EXPORTS, FACTORS
from ppr_reference import build_graph
from recall_reference import batch_iterate

GRID = [0.50, 0.70, 0.85, 0.95]


def ranks(x):
    """Average ranks (1-based) with ties sharing the mean rank."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = np.asarray(x)[order]
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def recall(data, lists, users, split_name):
    return float(np.mean([sum(m in data.relevant(split_name, u) for m in lists[u]) /
                          len(data.relevant(split_name, u)) for u in users]))


def cohort_breakdown(data, split_name, lines):
    rows = data.test if split_name == "test" else data.val
    all_users = sorted({u for u, _, _, _ in data.train} | {u for u, _, _, _ in rows})
    split_users = {u for u, _, _, _ in rows}
    relevant = [(u, m) for u, m, r, _ in rows if r >= 4.0]
    not_in_pool = [(u, m) for u, m in relevant if m not in data.pool_set]
    rated_earlier = [(u, m) for u, m in relevant if m in data.pool_set and m in data.excluded(split_name, u)]
    no_positive = [u for u in all_users if u not in data.positive_users]
    remaining = [u for u in all_users if u in data.positive_users]
    eligible = data.relevant_all(split_name)
    no_eligible = [u for u in remaining if not eligible.get(u)]
    no_ge4 = [u for u in no_eligible if not any(uu == u for uu, _ in relevant)]
    only_out_of_pool = [u for u in no_eligible if u not in no_ge4]
    cohort = [u for u in remaining if eligible.get(u)]
    lines.append(f"  {split_name}: users {len(all_users)} (with {split_name} ratings: {len(split_users)}) "
                 f"-> minus no positive training edge {len(no_positive)} {no_positive} = {len(remaining)} "
                 f"-> minus no eligible relevant movie {len(no_eligible)} = cohort {len(cohort)}")
    lines.append(f"     no eligible relevant: {len(no_ge4)} users have no {split_name} rating >= 4; {len(only_out_of_pool)} "
                 f"have ratings >= 4 only on movies outside the pool")
    lines.append(f"     relevant {split_name} ratings (>= 4): {len(relevant)}; dropped, not in the training pool: "
                 f"{len(not_in_pool)} (movies {sorted({m for _, m in not_in_pool})}); dropped, rated earlier: "
                 f"{len(rated_earlier)}")
    lines.append(f"     user 685 in {split_name}: ratings >= 4 eligible {len(eligible.get(685, set()))} "
                 f"(counted by the page's excluded_no_eligible over all split users: "
                 f"{sum(1 for u in split_users if not eligible.get(u))})")
    return cohort


def main():
    data = HW6Data()
    page = page_lists()
    alpha = page["alpha"]
    users, movies, R = build_graph()
    n_u, n = len(users), len(users) + len(movies)
    u_index = {u: k for k, u in enumerate(users)}
    lines = ["Extra checks (Python float64, own split and graph code; no page code)", "A1 cohort breakdown"]

    pairs = [(u, m) for u, m, _, _ in data.train + data.val + data.test]
    lines.append(f"  duplicate (user, movie) pairs in u.data: {len(pairs) - len(set(pairs))}")
    test_cohort = cohort_breakdown(data, "test", lines)
    val_cohort = cohort_breakdown(data, "val", lines)
    assert test_cohort == page["test"]["cohort"] and val_cohort == page["val"]["cohort"]
    lines.append("  both cohorts equal the page's exported cohorts")

    def ppr_lists(cohort, a, split_name):
        V = np.zeros((n, len(cohort)))
        V[[u_index[u] for u in cohort], np.arange(len(cohort))] = 1.0
        P, its, _, conv = batch_iterate(R, V, a, 1e-6, 500)
        out = {u: data.top_k(dict(zip(movies, P[n_u:, c])), u, split_name)[0] for c, u in enumerate(cohort)}
        return out, its, conv

    lines.append("A2 global PageRank vs training popularity (movie degree = popularity)")
    lines.append("  alpha | Spearman(PageRank, popularity) over 1,611 movies | mean common Top-10 with popularity "
                 "(test, 861 users) | identical lists")
    pop = np.asarray([data.popularity[m] for m in movies], dtype=float)
    pop_lists = {u: data.top_k(data.popularity, u, "test")[0] for u in test_cohort}
    pr_lists = {}
    for a in GRID:
        p, _, _, _ = batch_iterate(R, np.full((n, 1), 1.0 / n), a, 1e-6, 500)
        scores = dict(zip(movies, p[n_u:, 0]))
        lists_a = {u: data.top_k(scores, u, "test")[0] for u in test_cohort}
        pr_lists[a] = lists_a
        common = [len(set(lists_a[u]) & set(pop_lists[u])) for u in test_cohort]
        same = sum(lists_a[u] == pop_lists[u] for u in test_cohort)
        lines.append(f"  {a:.2f} | {spearman(p[n_u:, 0], pop):.4f} | {np.mean(common):.2f} | {same}")

    lines.append(f"A3 test cohort ({len(test_cohort)} users) at the selected alpha {alpha}")
    f = json.loads(FACTORS.read_text())
    P5, Q5 = np.asarray(f["userFactors"]), np.asarray(f["itemFactors"])
    pool = np.asarray(movies)
    method_lists = {
        "popularity": pop_lists,
        "pagerank": pr_lists[alpha],
        "ppr": ppr_lists(test_cohort, alpha, "test")[0],
        "mf (provided, collapsed)": {u: data.top_k(dict.fromkeys(movies, 0.0), u, "test")[0] for u in test_cohort},
        "mf (HW5, extra row)": {u: data.top_k(dict(zip(movies, f["globalMean"] + Q5[pool] @ P5[u])), u, "test")[0]
                                for u in test_cohort},
    }
    page_pop = {}
    with open(EXPORTS / "results.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == "test" and row["mean_popularity"]:
                page_pop[row["method"]] = float(row["mean_popularity"])
    lines.append("  method | mean list popularity (page) | catalogue coverage (distinct movies) | Recall@10")
    for name, lists in method_lists.items():
        mean_pop = np.mean([np.mean([data.popularity[m] for m in lists[u]]) for u in test_cohort])
        key = name.split()[0]
        page_value = f"{page_pop[key]:.3f}" if name != "mf (HW5, extra row)" else "-"
        coverage = len({m for u in test_cohort for m in lists[u]})
        lines.append(f"  {name} | {mean_pop:.3f} ({page_value}) | {coverage} | "
                     f"{recall(data, lists, test_cohort, 'test'):.6f}")
    ppr = method_lists["ppr"]
    for other in ("popularity", "pagerank", "mf (HW5, extra row)"):
        common = [len(set(ppr[u]) & set(method_lists[other][u])) for u in test_cohort]
        lines.append(f"  mean common Top-10, ppr vs {other}: {np.mean(common):.2f} "
                     f"(distribution {sorted(Counter(common).items())})")

    lines.append("A4 test Recall@10 by user activity (positive training edges)")
    edges = Counter(u for u, m, r, _ in data.train if r >= 4.0)
    activity = np.asarray([edges[u] for u in test_cohort])
    median = float(np.median(activity))
    light = [u for u in test_cohort if edges[u] <= median]
    heavy = [u for u in test_cohort if edges[u] > median]
    lines.append(f"  median positive edges {median:g}; light (<= median) {len(light)} users, heavy (> median) "
                 f"{len(heavy)} users")
    lines.append("  method | light | heavy | all")
    for name in ("popularity", "pagerank", "ppr", "mf (HW5, extra row)"):
        lists = method_lists[name]
        lines.append(f"  {name} | {recall(data, lists, light, 'test'):.6f} | {recall(data, lists, heavy, 'test'):.6f} | "
                     f"{recall(data, lists, test_cohort, 'test'):.6f}")

    lines.append("A5 IMPROVE - validation only, outside the readme grid, not used for any decision or test number")
    lines.append("  alpha | PPR validation Recall@10 (846 users) | iterations min/max | not converged")
    for a in (0.10, 0.30, 0.50):
        lists, its, conv = ppr_lists(val_cohort, a, "val")
        lines.append(f"  {a:.2f} | {recall(data, lists, val_cohort, 'val'):.6f} | {its.min()}/{its.max()} | "
                     f"{int((~conv).sum())}" + ("   (grid value; page 0.114483)" if a == 0.50 else ""))
    text = "\n".join(lines) + "\n"
    (RESULTS / "extra_checks.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

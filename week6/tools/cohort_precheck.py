"""Readme 7.1 cohort precheck for users 1 and 98 (the HW5 analysis users), in Python.

A user is evaluable for a split when the user has >= 1 positive training edge
and >= 1 relevant eligible movie: rating >= 4 in the split, in the training
candidate pool, and not rated in train (validation) or in train + val (test).
Uses this folder's own split code (independent_stats.split), not the page.

Usage: python3 cohort_precheck.py   (writes results/cohort_precheck.txt)
"""
from collections import defaultdict

from independent_stats import load, split
from paths import RESULTS

USERS = [1, 98]


def main():
    ratings, _ = load()
    train, val, test, _ = split(ratings)
    pool = {m for _, m, _, _ in train}
    positive = defaultdict(int)
    rated = {"train": defaultdict(set), "val": defaultdict(set)}
    for u, m, r, _ in train:
        rated["train"][u].add(m)
        positive[u] += r >= 4.0
    for u, m, _, _ in val:
        rated["val"][u].add(m)

    def eligible(rows, user, exclude):
        rel = [m for u, m, r, _ in rows if u == user and r >= 4.0]
        out_pool = [m for m in rel if m not in pool]
        seen = set().union(*(rated[s][user] for s in exclude))
        ok = sorted(m for m in rel if m in pool and m not in seen)
        return len(rel), len(out_pool), ok

    lines = ["user | positive train edges | split | relevant (>= 4) | not in pool | eligible relevant | evaluable"]
    for user in USERS:
        for name, rows, exclude in (("val", val, ["train"]), ("test", test, ["train", "val"])):
            n_rel, n_out, ok = eligible(rows, user, exclude)
            lines.append(f"{user} | {positive[user]} | {name} | {n_rel} | {n_out} | {len(ok)} {ok} | "
                         f"{positive[user] > 0 and len(ok) > 0}")
    text = "\n".join(lines) + "\n"
    (RESULTS / "cohort_precheck.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

"""Part C: the HW5 MF as an extra, disclosed MF row under the HW6 protocol (Python).

Scores mu + p_u . q_i (unclamped, as HW5 ranked) from results/hw5_mf_factors.json
over the same 1,611 training candidates, with the HW6 test exclusions (train +
val), score desc / ascending id, on the shared TEST cohort exported from the
page (results/page_lists.json). Reports Recall@10 and the Top-10 of users 1
and 98 in the B4 format. Scoring runs in float64 on the float32 factors.

Usage: python3 hw5_mf_recall.py   (writes results/hw5_mf_recall.txt)
"""
import json

import numpy as np

from hw6_data import RESULTS, HW6Data, page_lists
from paths import FACTORS

USERS = [1, 98]


def main():
    data = HW6Data()
    f = json.loads(FACTORS.read_text())
    mu, P, Q = f["globalMean"], np.asarray(f["userFactors"]), np.asarray(f["itemFactors"])
    page = page_lists()
    cohort = page["test"]["cohort"]
    assert cohort == data.cohort("test"), "page cohort differs from the Python cohort"

    pool = np.asarray(data.pool)
    recalls, hit_users, lists = [], 0, {}
    for user in cohort:
        values = mu + Q[pool] @ P[user]
        scores = dict(zip(data.pool, values))
        top, _ = data.top_k(scores, user, "test")
        rel = data.relevant("test", user)
        hits = sum(m in rel for m in top)
        recalls.append(hits / len(rel))
        hit_users += hits > 0
        lists[user] = (top, scores, rel)

    lines = [f"HW5 MF (epoch {f['epoch']}, val RMSE {f['valRMSE']:.6f}, test RMSE {f['testRMSE']:.6f}) under the HW6 "
             f"protocol: test cohort {len(cohort)} (== page cohort), 1,611 training candidates, train+val exclusions",
             f"Recall@10 = {np.mean(recalls):.6f}; users with >= 1 hit: {hit_users}"]
    for user in USERS:
        top, scores, rel = lists[user]
        pops = [data.popularity[m] for m in top]
        lines.append(f"user {user}: hits {sum(m in rel for m in top)}/{len(rel)}, mean pop {np.mean(pops):.1f}")
        for rank, m in enumerate(top, 1):
            lines.append(f"  {rank:>2} | {m:>4} | {data.title(m)} | {';'.join(data.genres(m))} | {scores[m]:.4f} | "
                         f"pop {data.popularity[m]} | hit {m in rel}")
    common = set(lists[1][0]) & set(lists[98][0])
    lines.append(f"common movies users 1 vs 98: {len(common)} {sorted(common)}")
    text = "\n".join(lines) + "\n"
    (RESULTS / "hw5_mf_recall.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

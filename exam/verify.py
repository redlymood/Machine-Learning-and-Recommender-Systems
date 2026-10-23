"""Independent checks of the pipeline. Writes results/verify.txt.

1. Join: one (user, airing) pair recomputed from the raw CSV text (no caches, no numpy time parsing).
2. AP@5: hand-computed values for the two example users.
3. PPR: TF float32 batch result vs a float64 numpy power iteration written separately, 25 users.
"""
import csv
import datetime as dt
import json

import numpy as np

from common import SPLITS, Data, apk
from hybrid import PPR_CFG
from methods import ppr_scores
from tune import setup

out = []
log = lambda s: (print(s), out.append(s))

# ---------------------------------------------------------------- 1. join from raw text
USER = "8377619604347126107"   # heavy example user
d = Data()
u = int(np.searchsorted(d.users, np.uint64(USER)))
allp = dict(np.load("cache/pairs.npz"))


def check_pair(idx, label):
    ai = int(allp["airing"][idx])
    ch, st, show = int(d.a["channel"][ai]), int(d.a["start"][ai]), str(int(d.a["show"][ai]))
    st_txt = dt.datetime.fromtimestamp(st, dt.UTC).strftime("%d.%m.%Y %H:%M:%S")
    row = None
    with open("data/export_arh_21-30-final.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["channel_id"] == str(ch) and r["start_time"] == st_txt and r["tv_show_id"] == show:
                row = r
                break
    log(f"[join:{label}] schedule row: channel {row['channel_id']} ({row['channel_title']}), start {row['start_time']}, "
        f"duration {row['duration']} s, '{row['tv_show_title']}', id {row['tv_show_id']}")
    a0 = dt.datetime.strptime(row["start_time"], "%d.%m.%Y %H:%M:%S")
    a1 = a0 + dt.timedelta(seconds=int(row["duration"]))
    overlap = 0
    with open("data/dataset11-30.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["user_id"] != USER or r["vsetv_id"] != row["channel_id"]:
                continue
            v0 = dt.datetime.strptime(r["start_time"], "%Y-%m-%d %H:%M:%S")
            v1 = dt.datetime.strptime(r["stop_time"], "%Y-%m-%d %H:%M:%S")
            ov = (min(v1, a1) - max(v0, a0)).total_seconds()
            if ov > 0:
                overlap += ov
                log(f"[join:{label}]   view {r['start_time']} -> {r['stop_time']} overlaps {ov:.0f} s")
    screen = overlap / int(row["duration"])
    log(f"[join:{label}] by hand: overlap {overlap:.0f} s / {row['duration']} s = screen {screen:.4f} "
        f"(qualifies at >= 0.8: {screen >= 0.8}); pipeline: overlap {int(allp['overlap'][idx])} s, "
        f"screen {float(allp['screen'][idx]):.4f}")
    assert abs(screen - float(allp["screen"][idx])) < 1e-9


mine = np.flatnonzero((allp["user"] == u) & (allp["week"] >= 21) & (allp["week"] <= 30))
full = mine[(allp["show"][mine] == 2400480) & (allp["screen"][mine] >= 0.8)][0]          # 'ТСН.' qualifying
part = mine[(allp["screen"][mine] > 0.5) & (allp["screen"][mine] < 0.8)][0]              # partial, below the cut
check_pair(full, "qualifying")
check_pair(part, "partial")

# ---------------------------------------------------------------- 2. AP@5 by hand
ex = json.load(open("results/examples.json"))
heavy, light = ex["users"]["heavy"], ex["users"]["light"]
hand = {("heavy", "popularity"): (1 / 1 + 2 / 2 + 3 / 4 + 4 / 5) / 5,   # hits at ranks 1,2,4,5
        ("heavy", "mf"): (1 + 1 + 1 + 1) / 5,                           # hits at ranks 1-4
        ("light", "final"): (1 / 5) / 5}                                # one hit at rank 5
for (who, meth), val in hand.items():
    code = ex["users"][who]["recommendations"][meth]["ap5"]
    log(f"[ap@5] {who}/{meth}: hand {val:.4f}, code {code:.4f}")
    assert abs(val - code) < 1e-12
assert apk({1, 2, 3, 4, 5}, [9, 9, 9, 9, 9]) == 0.0 and apk({1, 2, 3, 4, 5}, [1, 2, 3, 4, 5]) == 1.0

# ---------------------------------------------------------------- 3. PPR reference in float64
R, items, pool, fb, active = setup(d, "val", PPR_CFG["lam"])
S_tf, info = ppr_scores(R, alpha=PPR_CFG["alpha"], transform=PPR_CFG["transform"])
W = R.astype(np.float64)
P = (W / W.sum(0, keepdims=True).clip(min=1e-300)).T @ (W / W.sum(1, keepdims=True).clip(min=1e-300))
# P[i, j] = sum_u W[u,i]/d_i * W[u,j]/d_u  (item -> user -> item), rows sum to 1 for items with views
rng = np.random.default_rng(0)
users = rng.choice(np.flatnonzero(active), 25, replace=False)
max_diff, same_top5 = 0.0, 0
for uu in users:
    e = W[uu] / W[uu].sum()
    p = e.copy()
    for _ in range(200):
        p_new = (1 - PPR_CFG["alpha"]) * e + PPR_CFG["alpha"] * p @ P
        if np.abs(p_new - p).sum() < 1e-12:
            p = p_new
            break
        p = p_new
    max_diff = max(max_diff, float(np.abs(p - S_tf[uu]).max()))
    same_top5 += list(np.argsort(-p, kind="stable")[:5]) == list(np.argsort(-S_tf[uu], kind="stable")[:5])
log(f"[ppr] row sums of P: min {P.sum(1).min():.6f} max {P.sum(1).max():.6f}; TF iterations {info['iterations']}")
log(f"[ppr] float64 reference vs TF batch on 25 users: max |diff| {max_diff:.2e}, identical top-5 order {same_top5}/25")
open("results/verify.txt", "w").write("\n".join(out) + "\n")

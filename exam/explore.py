"""Quick facts about the caches: date ranges, user overlap, channel overlap, clock alignment."""
import numpy as np
import pandas as pd

v = dict(np.load("cache/views.npz"))  # dict(): NpzFile re-reads an array on every [] access
sch = {k: dict(np.load(f"cache/schedule_{k}.npz")) for k in ["11-20", "21-30", "31-42"]}
sub = pd.read_csv("data/submission.csv", dtype={"user_id": "uint64"})


def day(ts):
    return pd.to_datetime(ts, unit="s")


print("views start range", day(v["start"].min()), "->", day(v["start"].max()))
print("views stop  range", day(v["stop"].min()), "->", day(v["stop"].max()))
for k, s in sch.items():
    print(f"schedule {k}", day(s["start"].min()), "->", day(s["start"].max() + s["duration"][s["start"].argmax()]))

test_users = sub["user_id"].to_numpy()
in_train = np.isin(test_users, v["users"])
print(f"test users {len(test_users)}, unique {len(np.unique(test_users))}, in view log {in_train.sum()}")

vc = np.unique(v["channel"])
for k, s in sch.items():
    sc = np.unique(s["channel"])
    print(f"channels: views {len(vc)}, schedule {k} {len(sc)}, views-channels missing from schedule {len(np.setdiff1d(vc, sc))}")

# duraton == stop - start ?
print("duraton == stop-start:", np.mean(v["duraton"] == v["stop"] - v["start"]))
print("view length quantiles (s):", np.percentile(v["duraton"], [10, 25, 50, 75, 90, 99]).astype(int), "max", v["duraton"].max())

# Overlaps inside the schedule: on one channel, does an airing start before the previous one ends?
for k, s in sch.items():
    o = np.lexsort((s["start"], s["channel"]))
    ch, st, du = s["channel"][o], s["start"][o], s["duration"][o]
    same = ch[1:] == ch[:-1]
    gap = st[1:] - (st[:-1] + du[:-1])
    print(f"schedule {k}: overlapping consecutive airings {np.sum(same & (gap < 0)):,}, "
          f"gaps>0 {np.sum(same & (gap > 0)):,}, exact back-to-back {np.sum(same & (gap == 0)):,}, "
          f"duplicate (channel,start) {np.sum(same & (st[1:] == st[:-1])):,}")

# Clock alignment: view stops vs airing boundaries on the same channel, for shifts -3h..+3h.
s = sch["11-20"]
m = v["stop"] < s["start"].max()
vch, vstop = v["channel"][m], v["stop"][m]
for shift_h in range(-3, 4):
    hits = total = 0
    for c in np.unique(s["channel"]):
        sm = s["channel"] == c
        b = np.unique(np.concatenate([s["start"][sm], s["start"][sm] + s["duration"][sm]]))
        t = vstop[vch == c] + shift_h * 3600
        j = np.clip(np.searchsorted(b, t), 1, len(b) - 1)
        d = np.minimum(np.abs(t - b[j - 1]), np.abs(b[j] - t))
        hits += int(np.sum(d <= 60)); total += len(t)
    print(f"shift {shift_h:+d}h: view stops within 60 s of an airing boundary: {hits / total:.4f} of {total:,}")

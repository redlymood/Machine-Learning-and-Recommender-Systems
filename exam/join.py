"""Join the view log to the labelled schedule (same channel + interval overlap).

Output cache/airings.npz  : every labelled airing of weeks 11-42 (channel, start, duration, show, week)
       cache/pairs.npz    : one row per (user, labelled airing) with overlap seconds summed over
                            that user's view intervals; screen = overlap / airing duration.
Rows with tv_show_id = 0 are dropped before the join (the data page says "drop them").
"""
import time

import numpy as np

WEEK0 = 1583712000  # 2020-03-09 00:00:00, Monday of ISO week 11 (naive wall-clock seconds)
WEEK = 7 * 24 * 3600


def load_airings():
    parts = [dict(np.load(f"cache/schedule_{k}.npz")) for k in ["11-20", "21-30", "31-42"]]
    a = {k: np.concatenate([p[k] for p in parts]) for k in ["channel", "start", "duration", "show"]}
    lab = a["show"] != 0
    a = {k: x[lab] for k, x in a.items()}
    # The files overlap by a few boundary airings; keep one row per (channel, start).
    key = a["channel"].astype(np.int64) * 10**10 + a["start"]
    _, first = np.unique(key, return_index=True)
    order = np.lexsort((a["start"][first], a["channel"][first]))
    keep = first[order]
    a = {k: x[keep] for k, x in a.items()}
    a["week"] = ((a["start"] - WEEK0) // WEEK + 11).astype(np.int16)
    return a, int(lab.sum()) - len(keep)


def join(v, a):
    """Return arrays (user, airing_index, overlap) for every positive overlap."""
    users, airs, overlaps = [], [], []
    a_end = a["start"] + a["duration"]
    for c in np.unique(v["channel"]):
        am = np.flatnonzero(a["channel"] == c)
        if len(am) == 0:
            continue
        vm = np.flatnonzero(v["channel"] == c)
        st, en = a["start"][am], a_end[am]
        vs, ve = v["start"][vm], v["stop"][vm]
        lo = np.searchsorted(en, vs, side="right")  # first airing ending after the view starts
        hi = np.searchsorted(st, ve, side="left")   # airings starting before the view stops
        cnt = np.maximum(hi - lo, 0)
        if cnt.sum() == 0:
            continue
        rep = np.repeat(np.arange(len(vm)), cnt)
        offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        ai = lo[rep] + offs
        ov = np.minimum(ve[rep], en[ai]) - np.maximum(vs[rep], st[ai])
        ok = ov > 0
        users.append(v["user_idx"][vm][rep][ok])
        airs.append(am[ai][ok])
        overlaps.append(ov[ok])
    return np.concatenate(users), np.concatenate(airs), np.concatenate(overlaps)


def main():
    t0 = time.time()
    a, dup = load_airings()
    print(f"labelled airings weeks 11-42: {len(a['show']):,} (dropped {dup} cross-file duplicates); "
          f"weeks {a['week'].min()}..{a['week'].max()}")
    v = dict(np.load("cache/views.npz"))
    u, ai, ov = join(v, a)
    print(f"raw view x airing overlaps: {len(u):,}  ({time.time() - t0:.1f}s)")

    # Sum overlaps of the same user on the same airing (several view intervals can cover one airing).
    key = u.astype(np.int64) * len(a["show"]) + ai
    ukey, inv = np.unique(key, return_inverse=True)
    tot = np.bincount(inv, weights=ov).astype(np.int64)
    pu = (ukey // len(a["show"])).astype(np.int32)
    pa = (ukey % len(a["show"])).astype(np.int64)
    screen = tot / a["duration"][pa]
    np.savez("cache/airings.npz", **a)
    np.savez("cache/pairs.npz", user=pu, airing=pa, overlap=tot, screen=screen,
             show=a["show"][pa], week=a["week"][pa])
    print(f"user x airing pairs: {len(pu):,}; screen>=0.8: {(screen >= 0.8).sum():,}; "
          f"screen>1.0 (overlapping view intervals): {(screen > 1.0).sum():,}  ({time.time() - t0:.1f}s)")
    q = np.percentile(screen, [10, 25, 50, 75, 90])
    print("screen quantiles 10/25/50/75/90:", np.round(q, 3))


if __name__ == "__main__":
    main()

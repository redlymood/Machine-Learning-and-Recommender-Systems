"""Schedule-aware re-ranking (uses the target-window schedule only: which show airs where and when).

final(u, s) = fused(u, s) * ratio(u, s) ** beta
"""
import numpy as np

WEEK0, WEEK = 1583712000, 7 * 24 * 3600  # 2020-03-09 00:00 = start of ISO week 11


def schedule_ratio(d, items, hist, target, recent=None):
    """ratio[s] = per-week airings of s in target / per-week airings of s in the (last `recent`) history weeks."""
    weeks_h = list(hist)[-recent:] if recent else list(hist)
    ah = d.airing_counts(weeks_h).reindex(items).fillna(0).to_numpy() / len(weeks_h)
    at = d.airing_counts(target).reindex(items).fillna(0).to_numpy() / len(target)
    # shows absent from the recent history schedule: treat as one airing over the window (finite boost)
    return at / np.maximum(ah, 1.0 / len(weeks_h))


def user_channel_ratio(d, items, hist, target):
    """ratio[u, s] = per-week airings of s in target on u's history channels
                     / max(per-week airings of s in history on u's channels, 1 airing per window)."""
    v = dict(np.load("cache/views.npz"))
    wk = (v["start"] - WEEK0) // WEEK + 11
    m = np.isin(wk, np.asarray(hist))
    chans = np.unique(d.a["channel"])
    uc = np.zeros((d.n_users, len(chans)), dtype=np.float32)
    ci = np.searchsorted(chans, v["channel"][m])
    # a user's channels = channels with any view interval in the history weeks
    ok = (ci < len(chans)) & (chans[np.minimum(ci, len(chans) - 1)] == v["channel"][m])
    uc[v["user_idx"][m][ok], ci[ok]] = 1.0

    def ch_show(weeks):
        mm = np.isin(d.a["week"], np.asarray(weeks)) & np.isin(d.a["show"], items)
        M = np.zeros((len(chans), len(items)), dtype=np.float32)
        np.add.at(M, (np.searchsorted(chans, d.a["channel"][mm]), np.searchsorted(items, d.a["show"][mm])), 1.0)
        return M / len(weeks)

    ah = uc @ ch_show(hist)
    at = uc @ ch_show(target)
    return at / np.maximum(ah, 1.0 / len(hist))


def user_slot_ratio(d, items, hist, target, weight="binary", max_hours=24):
    """Like user_channel_ratio, but over (channel, hour-of-day) slots.

    A user's slot profile comes from every view interval in the history weeks (labelled or not):
    each view is spread over the clock hours it covers (capped at `max_hours`).
      weight="binary": slot = 1 if the user had any view in it;  "time": share of the user's
      viewing hours that fell into the slot.
    An airing belongs to the slot of its start hour on its channel.
    ratio[u, s] = per-week airings of s in target in u's slots (weighted) / same for history,
    floored at one airing per history window.
    """
    v = dict(np.load("cache/views.npz"))
    wk = (v["start"] - WEEK0) // WEEK + 11
    m = np.isin(wk, np.asarray(hist))
    chans = np.unique(d.a["channel"])
    ch = v["channel"][m]
    ci = np.searchsorted(chans, ch)
    ok = (ci < len(chans)) & (chans[np.minimum(ci, len(chans) - 1)] == ch)
    users, ci = v["user_idx"][m][ok], ci[ok]
    h0 = (v["start"][m][ok] // 3600)
    h1 = (np.maximum(v["stop"][m][ok] - 1, v["start"][m][ok]) // 3600)
    n_h = np.minimum(h1 - h0 + 1, max_hours)
    rep = np.repeat(np.arange(len(h0)), n_h)
    hour = (h0[rep] + (np.arange(n_h.sum()) - np.repeat(np.cumsum(n_h) - n_h, n_h))) % 24
    slot = ci[rep] * 24 + hour
    n_slots = len(chans) * 24
    up = np.zeros((d.n_users, n_slots), dtype=np.float32)
    np.add.at(up, (users[rep], slot), 1.0)
    if weight == "binary":
        up = (up > 0).astype(np.float32)
    else:
        up = up / np.maximum(up.sum(axis=1, keepdims=True), 1.0)

    def slot_show(weeks):
        mm = np.isin(d.a["week"], np.asarray(weeks)) & np.isin(d.a["show"], items)
        s = np.searchsorted(chans, d.a["channel"][mm]) * 24 + (d.a["start"][mm] // 3600) % 24
        M = np.zeros((n_slots, len(items)), dtype=np.float32)
        np.add.at(M, (s, np.searchsorted(items, d.a["show"][mm])), 1.0)
        return M / len(weeks)

    ah = up @ slot_show(hist)
    at = up @ slot_show(target)
    floor = (1.0 / len(hist)) * (1.0 if weight == "binary" else 1.0 / n_slots)
    return at / np.maximum(ah, floor)

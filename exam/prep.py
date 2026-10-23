"""Parse the raw CSVs once into numpy caches (cache/*.npz, cache/shows.csv).

Times become int64 seconds since 1970-01-01 read as naive wall-clock time
(no timezone shift); the join script checks the alignment of the two sources.
"""
import os
import time

import numpy as np
import pandas as pd

DATA = "data"
CACHE = "cache"
SCHEDULES = {
    "11-20": "export_arh_11-20-final.csv",
    "21-30": "export_arh_21-30-final.csv",
    "31-42": "export_arh_31-42-final.csv",
}
META_COLS = ["tv_show_title", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2",
             "tv_show_genre_3", "year_of_production", "director", "actors"]


def to_seconds(series, fmt):
    # pandas 3 parses to datetime64[us]; cast to seconds explicitly before taking int64
    return pd.to_datetime(series, format=fmt).astype("datetime64[s]").astype("int64")


def main():
    os.makedirs(CACHE, exist_ok=True)
    t0 = time.time()

    views = pd.read_csv(os.path.join(DATA, "dataset11-30.csv"),
                        dtype={"user_id": "uint64", "vsetv_id": "int32", "duraton": "int64"})
    users, user_idx = np.unique(views["user_id"].to_numpy(), return_inverse=True)
    start = to_seconds(views["start_time"], "%Y-%m-%d %H:%M:%S").to_numpy()
    stop = to_seconds(views["stop_time"], "%Y-%m-%d %H:%M:%S").to_numpy()
    np.savez(os.path.join(CACHE, "views.npz"), users=users, user_idx=user_idx.astype(np.int32),
             channel=views["vsetv_id"].to_numpy(), start=start, stop=stop,
             duraton=views["duraton"].to_numpy())
    print(f"views: {len(views):,} rows, {len(users):,} users, {time.time() - t0:.1f}s")

    meta_frames = []
    for key, fname in SCHEDULES.items():
        sch = pd.read_csv(os.path.join(DATA, fname), dtype={"tv_show_id": "int64"}, low_memory=False)
        st = to_seconds(sch["start_time"], "%d.%m.%Y %H:%M:%S").to_numpy()
        np.savez(os.path.join(CACHE, f"schedule_{key}.npz"),
                 channel=sch["channel_id"].to_numpy().astype(np.int32), start=st,
                 duration=sch["duration"].to_numpy().astype(np.int64),
                 show=sch["tv_show_id"].to_numpy())
        lab = sch[sch["tv_show_id"] != 0]
        meta_frames.append(lab[["tv_show_id"] + META_COLS].assign(src=key))
        print(f"schedule {key}: {len(sch):,} rows, {len(lab):,} labelled, "
              f"{lab['tv_show_id'].nunique():,} shows, {time.time() - t0:.1f}s")

    # One metadata row per show: the most frequent title/category (first-seen on ties).
    meta = pd.concat(meta_frames, ignore_index=True)
    shows = (meta.groupby("tv_show_id")[META_COLS]
             .agg(lambda s: s.mode(dropna=True).iloc[0] if s.notna().any() else ""))
    shows["airings"] = meta.groupby("tv_show_id").size()
    shows.to_csv(os.path.join(CACHE, "shows.csv"))
    print(f"shows: {len(shows):,} distinct labelled ids, total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

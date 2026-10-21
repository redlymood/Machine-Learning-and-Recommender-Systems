"""Independent HW6 data / graph numbers, computed in Python from the raw files.

Reads week6/u.data and week6/u.item directly (no page code, no data.js) and
re-implements the split rule stated in the task:
  per user, sort by (timestamp, movieId); nTrain = floor(0.8 n),
  nVal = floor(0.1 n), nTest = the rest; users with < 5 ratings are skipped.
Then derives the HW6 graph quantities (readme 3.1): positive training edges
(rating >= 4 in TRAIN), the node universe (users / movies observed in TRAIN),
zero-degree nodes and the training candidate pool (readme 5). The shared
evaluation cohorts (readme 7.1) are added because they set the number of PPR
runs in the cost estimate.

Usage: python3 independent_stats.py   (prints and writes results/independent_stats.txt)
"""
from collections import defaultdict
from paths import RESULTS, WEEK6

OUT = RESULTS / "independent_stats.txt"
MIN_RATINGS = 5


def load():
    ratings = []
    for line in (WEEK6 / "u.data").read_text().splitlines():
        if line.strip():
            u, m, r, t = line.split("\t")
            ratings.append((int(u), int(m), float(r), int(t)))
    item_ids = []
    # u.item is Latin-1; only the id field is needed here.
    for line in (WEEK6 / "u.item").read_text(encoding="latin-1").splitlines():
        fields = line.split("|")
        if line.strip() and len(fields) >= 24:
            item_ids.append(int(fields[0]))
    return ratings, item_ids


def split(ratings):
    by_user = defaultdict(list)
    for row in ratings:
        by_user[row[0]].append(row)
    train, val, test, skipped = [], [], [], []
    for user, rows in by_user.items():
        n = len(rows)
        if n < MIN_RATINGS:
            skipped.append(user)
            continue
        rows = sorted(rows, key=lambda r: (r[3], r[1]))
        n_train, n_val = (8 * n) // 10, n // 10  # integer floors of 0.8 n and 0.1 n
        train += rows[:n_train]
        val += rows[n_train:n_train + n_val]
        test += rows[n_train + n_val:]
    return train, val, test, skipped


def cohort(split_rows, split_name, train, val, candidate_pool, positive_users):
    """Readme 7.1 / 7.2 cohort: positive training edge + >= 1 eligible relevant movie."""
    rated = defaultdict(set)
    for u, m, _, _ in train:
        rated[u].add(m)
    if split_name == "test":
        for u, m, _, _ in val:
            rated[u].add(m)
    relevant = defaultdict(int)
    unseen = 0
    for u, m, r, _ in split_rows:
        if r < 4.0:
            continue
        if m not in candidate_pool:
            unseen += 1
            continue
        if m in rated[u]:
            continue
        relevant[u] += 1
    users = sorted(u for u in relevant if u in positive_users)
    split_users = {u for u, _, _, _ in split_rows}
    no_eligible = sum(1 for u in split_users if relevant.get(u, 0) == 0)
    return users, unseen, no_eligible, sum(relevant[u] for u in users)


def main():
    ratings, item_ids = load()
    users = {r[0] for r in ratings}
    rated_movies = {r[1] for r in ratings}
    train, val, test, skipped = split(ratings)

    train_users = {r[0] for r in train}
    train_movies = {r[1] for r in train}
    positive_pairs = {(r[0], r[1]) for r in train if r[2] >= 4.0}
    pos_users = {u for u, _ in positive_pairs}
    pos_movies = {m for _, m in positive_pairs}
    zero_users = sorted(train_users - pos_users)
    zero_movies = sorted(train_movies - pos_movies)
    n_nodes = len(train_users) + len(train_movies)

    n_r, n_u, n_m = len(ratings), len(users), len(item_ids)
    max_u, max_m = max(users), max(item_ids)
    val_users, val_unseen, val_noel, val_rel = cohort(val, "val", train, val, train_movies, pos_users)
    test_users, test_unseen, test_noel, test_rel = cohort(test, "test", train, val, train_movies, pos_users)

    lines = [
        "HW6 independent stats (Python, raw week6/u.data + u.item; no page code)",
        "",
        "[dataset]",
        f"ratings                       {n_r}",
        f"users (distinct in u.data)    {n_u}   (max id {max_u})",
        f"movies in u.item              {n_m}   (max id {max_m}); distinct rated movies {len(rated_movies)}",
        f"sparsity 1 - R/(U*M)          {1 - n_r / (n_u * n_m):.6f}  ({100 * (1 - n_r / (n_u * n_m)):.2f}%)",
        f"sparsity, page formula        {1 - (len(train) + len(val) + len(test)) / ((max_u + 1) * (max_m + 1)):.6f}"
        "  (1 - split total / ((maxUser+1)*(maxMovie+1)), script.js:1960-1961)",
        "",
        "[split: per-user chronological 80/10/10, floor counts, ties by movieId, skip n < 5]",
        f"train / val / test            {len(train)} / {len(val)} / {len(test)}   (sum {len(train) + len(val) + len(test)})",
        f"excluded users (n < 5)        {len(skipped)}",
        "",
        "[graph, readme 3.1]",
        f"positive training edges       {len(positive_pairs)}   (distinct (user, movie) with train rating >= 4)",
        f"graph nodes                   {n_nodes} = {len(train_users)} users + {len(train_movies)} movies (no padding)",
        f"directed transitions (2|E|)   {2 * len(positive_pairs)}",
        f"density of R                  {len(positive_pairs) / (len(train_users) * len(train_movies)):.6f}",
        f"zero-degree users             {len(zero_users)}   ids {zero_users}",
        f"zero-degree movies            {len(zero_movies)}",
        f"zero-degree nodes total       {len(zero_users) + len(zero_movies)}",
        f"users with >= 1 positive edge {len(pos_users)}; movies with >= 1 positive edge {len(pos_movies)}",
        "",
        "[candidate pool, readme 5]",
        f"training candidate pool       {len(train_movies)} movies (observed in train)",
        f"metadata movies not in pool   {n_m - len(train_movies)}",
        "",
        "[shared cohorts, readme 7.1 / 7.2 (for the PPR cost estimate)]",
        f"val  cohort users             {len(val_users)}   relevant eligible {val_rel}; "
        f"relevant ratings absent from pool {val_unseen}; split users without eligible relevant {val_noel}",
        f"test cohort users             {len(test_users)}   relevant eligible {test_rel}; "
        f"relevant ratings absent from pool {test_unseen}; split users without eligible relevant {test_noel}",
        f"val cohort ∪ test cohort      {len(set(val_users) | set(test_users))}",
    ]
    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

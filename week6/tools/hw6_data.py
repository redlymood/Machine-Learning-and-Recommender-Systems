"""Shared Python view of the HW6 data, built from the raw files with this folder's own split.

`HW6Data()` holds the split (independent_stats.split), the training candidate
pool, training popularity (distinct users with a train rating >= 4), the
previously-rated exclusion sets (val: train; test: train + val), the relevant
eligible movies per split (readme 7.2), the shared cohorts (readme 7.1) and
titles / genres from u.item (Latin-1; the 18 real genre flags are fields
6..23, as in data.js). No page code is used.
"""
import json
from collections import defaultdict

from independent_stats import load, split
from paths import RESULTS, WEEK6

GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama",
          "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
          "Western"]


def load_items():
    items = {}
    for line in (WEEK6 / "u.item").read_text(encoding="latin-1").splitlines():
        fields = line.split("|")
        if line.strip() and len(fields) >= 24:
            flags = fields[6:24]
            items[int(fields[0])] = (fields[1], [g for g, f in zip(GENRES, flags) if f == "1"])
    return items


class HW6Data:
    def __init__(self):
        ratings, _ = load()
        self.train, self.val, self.test, _ = split(ratings)
        self.items = load_items()
        self.pool = sorted({m for _, m, _, _ in self.train})
        self.pool_set = set(self.pool)
        positive = defaultdict(set)
        for u, m, r, _ in self.train:
            if r >= 4.0:
                positive[m].add(u)
        self.popularity = {m: len(positive[m]) for m in self.pool}
        self.positive_users = {u for users in positive.values() for u in users}
        self.rated = {"train": defaultdict(set), "val": defaultdict(set)}
        for u, m, _, _ in self.train:
            self.rated["train"][u].add(m)
        for u, m, _, _ in self.val:
            self.rated["val"][u].add(m)

    def excluded(self, split_name, user):
        out = set(self.rated["train"][user])
        if split_name == "test":
            out |= self.rated["val"][user]
        return out

    def relevant(self, split_name, user):
        rows = self.test if split_name == "test" else self.val
        ex = self.excluded(split_name, user)
        return {m for u, m, r, _ in rows if u == user and r >= 4.0 and m in self.pool_set and m not in ex}

    def relevant_all(self, split_name):
        rows = self.test if split_name == "test" else self.val
        out = defaultdict(set)
        for u, m, r, _ in rows:
            if r >= 4.0 and m in self.pool_set:
                out[u].add(m)
        return {u: ms - self.excluded(split_name, u) for u, ms in out.items()}

    def cohort(self, split_name):
        rel = self.relevant_all(split_name)
        return sorted(u for u, ms in rel.items() if ms and u in self.positive_users)

    def top_k(self, scores, user, split_name, k=10):
        """scores: dict movieId -> score over the pool; score desc, ties by ascending id."""
        ex = self.excluded(split_name, user)
        eligible = [m for m in self.pool if m not in ex]
        eligible.sort(key=lambda m: (-scores[m], m))
        return eligible[:k], eligible

    def title(self, m):
        return self.items[m][0]

    def genres(self, m):
        return self.items[m][1]


def page_lists():
    return json.loads((RESULTS / "page_lists.json").read_text())

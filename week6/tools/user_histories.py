"""B6: training histories of users 1 and 98 (Python, own split; no page code).

For each user: training ratings and positive ones (>= 4), genre shares among
the training-rated and the positive movies (multi-label: share of movies that
carry the genre), mean training popularity of the positive movies, and the 5
most / 5 least popular positive movies (ties by ascending id). Popularity =
distinct users with a train rating >= 4 (readme 7.3).

Usage: python3 user_histories.py   (writes results/user_histories.txt)
"""
from collections import Counter

from hw6_data import RESULTS, HW6Data

USERS = [1, 98]


def shares(data, movies, top=6):
    counts = Counter(g for m in movies for g in data.genres(m))
    return ", ".join(f"{g} {100 * c / len(movies):.0f}%" for g, c in counts.most_common(top))


def main():
    data = HW6Data()
    pops = [data.popularity[m] for m in data.pool]
    edge_pops = [data.popularity[m] for u, m, r, _ in data.train if r >= 4.0]
    lines = [f"context: mean popularity over the 1,611 pool movies {sum(pops) / len(pops):.1f}; "
             f"over the 45,602 positive training edges {sum(edge_pops) / len(edge_pops):.1f}"]
    for user in USERS:
        rated = sorted({m for u, m, _, _ in data.train if u == user})
        positive = sorted({m for u, m, r, _ in data.train if u == user and r >= 4.0})
        by_pop = sorted(positive, key=lambda m: (-data.popularity[m], m))
        lines.append(f"user {user}: {len(rated)} training ratings, {len(positive)} positive (>= 4)")
        lines.append(f"  genre shares, all training ratings: {shares(data, rated)}")
        lines.append(f"  genre shares, positive movies:      {shares(data, positive)}")
        lines.append(f"  mean popularity of positive movies: {sum(data.popularity[m] for m in positive) / len(positive):.1f}")
        lines.append("  5 most popular positive movies:  " +
                     "; ".join(f"{m} {data.title(m)} (pop {data.popularity[m]})" for m in by_pop[:5]))
        lines.append("  5 least popular positive movies: " +
                     "; ".join(f"{m} {data.title(m)} (pop {data.popularity[m]})" for m in by_pop[-5:]))
    text = "\n".join(lines) + "\n"
    (RESULTS / "user_histories.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

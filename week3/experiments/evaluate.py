"""HW3 part 2: offline evaluation of the collaborative-filtering strategies.

Numpy only for the computations; matplotlib (from week3/.venv) is used for the
plots when available. All randomness goes through SEED, so reruns are identical.

Run from week3/:
    .venv/bin/python experiments/evaluate.py

Results: experiments/results/*.csv and *.png
"""
import csv
import time
from pathlib import Path

import numpy as np

SEED = 42
TEST_FRACTION = 0.2
N_NEIGHBORS = 20          # User-Based
K_NEIGHBORS = 20          # Item-Based
MIN_SUPPORT = 3           # same reliability threshold as script.js (Top-5 only)
BETAS = (10, 25, 50)      # significance-weighting caps for strategy C
MAIN_BETA = 50
TOP_K = 5
TOP5_USERS = 50
POPULARITY_THRESHOLDS = (20, 50)   # extra Top-5 variants: min train ratings of a candidate movie
MF_PARAMS = dict(k=20, lr=0.01, reg=0.05, epochs=30)
MF_VALIDATION_FRACTION = 0.1   # carved out of train; the best epoch is chosen on it, never on test
NEW_USER_LEVELS = (0, 1, 3, 5, 10)
NEW_USER_COUNT = 5
NEW_USER_MIN_RATINGS = 50

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent
OUT = HERE / "results"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data():
    data = np.loadtxt(DATA_DIR / "u.data", dtype=np.int64)[:, :3]
    titles = {}
    for line in (DATA_DIR / "u.item").read_bytes().decode("latin-1").splitlines():
        fields = line.split("|")
        titles[int(fields[0])] = fields[1]
    return data, titles


def split_per_user(data, rng, test_fraction=TEST_FRACTION):
    """Random holdout inside every user: round(20%) of each user's ratings go to test."""
    is_test = np.zeros(len(data), dtype=bool)
    for u in np.unique(data[:, 0]):
        idx = np.nonzero(data[:, 0] == u)[0]
        n_test = int(round(len(idx) * test_fraction))
        is_test[rng.choice(idx, n_test, replace=False)] = True
    return data[~is_test], data[is_test]


def to_matrix(rows, num_users, num_items):
    R = np.zeros((num_users + 1, num_items + 1))
    R[rows[:, 0], rows[:, 1]] = rows[:, 2]
    return R


def row_means(M, fallback):
    """Mean of the non-zero entries of every row (fallback for empty rows)."""
    counts = (M > 0).sum(1)
    sums = M.sum(1)
    return np.where(counts > 0, sums / np.maximum(counts, 1), fallback)


# ---------------------------------------------------------------------------
# Similarities (rows of M are the entities: users for UB, items for IB)
# ---------------------------------------------------------------------------
def corated_cosine(M):
    """Strategy A: cosine over co-rated entries only. Returns (sim, common); diagonal = 0."""
    B = (M > 0).astype(float)
    S = M * M
    dot = M @ M.T
    norm_a = S @ B.T            # sum of a_k^2 over k co-rated with b
    norm_b = B @ S.T
    common = B @ B.T
    denom = np.sqrt(norm_a) * np.sqrt(norm_b)
    sim = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
    np.fill_diagonal(sim, 0.0)
    common = common.astype(np.int64)
    np.fill_diagonal(common, 0)
    return sim, common


def mean_imputed_cosine(M, fill):
    """Strategy B: missing entries replaced by the row mean, cosine over full vectors."""
    X = np.where(M > 0, M, fill[:, None])
    X[0, :] = 0.0               # id 0 is unused
    X[:, 0] = 0.0
    norms = np.linalg.norm(X, axis=1)
    Xn = np.divide(X, norms[:, None], out=np.zeros_like(X), where=norms[:, None] > 0)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, 0.0)
    return sim


def significance_weighted(sim, common, beta):
    """Strategy C: sim * min(common, beta) / beta."""
    return sim * np.minimum(common, beta) / beta


# ---------------------------------------------------------------------------
# Predictors: same formula as script.js, vectorised over a list of items
# ---------------------------------------------------------------------------
def neighbor_order(sim_row):
    """Ids with sim > 0, sorted by sim desc then id asc (as in script.js)."""
    order = np.lexsort((np.arange(len(sim_row)), -sim_row))
    return order[sim_row[order] > 0]


def predict_user_based(S, R, u, items, N=N_NEIGHBORS):
    """Top-N most similar users (sim > 0) who rated each item; sum(sim*r)/sum(sim)."""
    order = neighbor_order(S[u])
    if len(order) == 0:
        return np.full(len(items), np.nan), np.zeros(len(items), dtype=int)
    sims = S[u, order]
    ratings = R[np.ix_(order, items)]
    rated = ratings > 0
    use = rated & (np.cumsum(rated, axis=0) <= N)
    weights = sims[:, None] * use
    den = weights.sum(0)
    num = (weights * ratings).sum(0)
    support = use.sum(0)
    pred = np.divide(num, den, out=np.full(len(items), np.nan), where=support > 0)
    return pred, support


def predict_item_based(S, R, u, items, K=K_NEIGHBORS):
    """Top-K movies rated by u most similar (sim > 0) to each item; sum(sim*r_uj)/sum(sim)."""
    rated_items = np.nonzero(R[u] > 0)[0]
    if len(rated_items) == 0:
        return np.full(len(items), np.nan), np.zeros(len(items), dtype=int)
    sims = S[np.ix_(items, rated_items)]
    top = np.argsort(-sims, axis=1, kind="stable")[:, :K]     # stable => id asc on ties
    top_sims = np.take_along_axis(sims, top, axis=1)
    top_ratings = R[u, rated_items][top]
    use = top_sims > 0
    weights = top_sims * use
    den = weights.sum(1)
    num = (weights * top_ratings).sum(1)
    support = use.sum(1)
    pred = np.divide(num, den, out=np.full(len(items), np.nan), where=support > 0)
    return pred, support


def predict_test(predictor, S, R, test):
    """Predict every test pair, grouped by user. Returns (pred with NaN for null, support)."""
    pred = np.full(len(test), np.nan)
    support = np.zeros(len(test), dtype=int)
    users = test[:, 0]
    for u in np.unique(users):
        idx = np.nonzero(users == u)[0]
        p, s = predictor(S, R, u, test[idx, 1])
        pred[idx] = p
        support[idx] = s
    return pred, support


# ---------------------------------------------------------------------------
# Matrix factorisation (strategy D)
# ---------------------------------------------------------------------------
def train_mf(train, val, test, num_users, num_items, rng, k, lr, reg, epochs):
    """SGD MF with biases; returns the model from the epoch with the lowest validation RMSE."""
    mu = train[:, 2].mean()
    bu = np.zeros(num_users + 1)
    bi = np.zeros(num_items + 1)
    P = rng.normal(0, 0.1, (num_users + 1, k))
    Q = rng.normal(0, 0.1, (num_items + 1, k))
    history = []

    def predict(rows):
        p = mu + bu[rows[:, 0]] + bi[rows[:, 1]] + np.einsum("ij,ij->i", P[rows[:, 0]], Q[rows[:, 1]])
        return np.clip(p, 1, 5)

    best = None
    users, items, ratings = train[:, 0].tolist(), train[:, 1].tolist(), train[:, 2].astype(float).tolist()
    for epoch in range(1, epochs + 1):
        for t in rng.permutation(len(train)):
            u, i, r = users[t], items[t], ratings[t]
            pu, qi = P[u], Q[i]
            e = r - (mu + bu[u] + bi[i] + pu @ qi)
            bu[u] += lr * (e - reg * bu[u])
            bi[i] += lr * (e - reg * bi[i])
            pu_old = pu.copy()
            P[u] += lr * (e * qi - reg * pu)
            Q[i] += lr * (e * pu_old - reg * qi)
        history.append((epoch, rmse(predict(train), train[:, 2]), rmse(predict(val), val[:, 2]),
                        rmse(predict(test), test[:, 2])))
        print(f"    MF epoch {epoch:2d}: train {history[-1][1]:.4f}, val {history[-1][2]:.4f}, test {history[-1][3]:.4f}")
        if best is None or history[-1][2] < best[1]:
            best = (epoch, history[-1][2], bu.copy(), bi.copy(), P.copy(), Q.copy())

    # Early stopping: restore the parameters of the best validation epoch
    best_epoch = best[0]
    bu[:], bi[:], P[:], Q[:] = best[2], best[3], best[4], best[5]
    return predict, history, best_epoch


# ---------------------------------------------------------------------------
# Metrics and helpers
# ---------------------------------------------------------------------------
def rmse(pred, truth):
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def mae(pred, truth):
    return float(np.mean(np.abs(pred - truth)))


def with_fallback(pred, fallback):
    return np.where(np.isnan(pred), fallback, pred)


def write_csv(name, header, rows):
    with open(OUT / name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def bin_label(value, edges):
    for low, high, label in edges:
        if low <= value <= high:
            return label
    raise ValueError(value)


ITEM_BINS = [(0, 0, "0"), (1, 5, "1-5"), (6, 20, "6-20"), (21, 100, "21-100"), (101, 10**9, ">100")]
USER_BINS = [(20, 30, "20-30"), (31, 60, "31-60"), (61, 150, "61-150"), (151, 10**9, ">150")]
COMMON_BINS = [(0, 0, "0"), (1, 1, "1"), (2, 4, "2-4"), (5, 19, "5-19"), (20, 49, "20-49"), (50, 10**9, ">=50")]


def upper_pairs(n):
    """Index arrays of all pairs a < b among ids 1..n."""
    return np.triu_indices(n + 1, k=1)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    data, titles = load_data()
    num_users, num_items = int(data[:, 0].max()), len(titles)

    # ---- 0. Split --------------------------------------------------------
    train, test = split_per_user(data, rng)
    R = to_matrix(train, num_users, num_items)
    global_mean = train[:, 2].mean()
    user_mean = row_means(R, global_mean)
    item_counts = (R > 0).sum(0)
    user_counts = (R > 0).sum(1)
    item_mean = row_means(R.T, global_mean)
    truth = test[:, 2].astype(float)
    fallback = user_mean[test[:, 0]]

    new_items = item_counts[test[:, 1]] == 0
    print(f"[0] train {len(train)}, test {len(test)}; users in train: {int((user_counts[1:] > 0).sum())}/{num_users}")
    print(f"    test pairs with a movie unseen in train: {int(new_items.sum())} "
          f"({int(len(np.unique(test[new_items, 1])))} distinct movies)")
    write_csv("split.csv", ["metric", "value"], [
        ["seed", SEED], ["train_ratings", len(train)], ["test_ratings", len(test)],
        ["users_in_train", int((user_counts[1:] > 0).sum())],
        ["test_pairs_with_new_item", int(new_items.sum())],
        ["distinct_new_items_in_test", int(len(np.unique(test[new_items, 1])))],
        ["items_with_zero_train_ratings", int((item_counts[1:] == 0).sum())],
    ])

    # ---- 1. Strategies x methods -----------------------------------------
    print("[1] strategies x methods")
    summary = []
    test_preds = {}   # name -> prediction with fallback (for the bins in part 3)

    def record(name, method, strategy, pred, prep_s, pred_s):
        covered = ~np.isnan(pred)
        final = with_fallback(pred, fallback)
        test_preds[name] = final
        row = [method, strategy, rmse(final, truth), mae(final, truth), float(covered.mean()),
               rmse(pred[covered], truth[covered]) if covered.any() else float("nan"), prep_s, pred_s]
        summary.append(row)
        print(f"    {method:10s} {strategy:28s} RMSE {row[2]:.4f} MAE {row[3]:.4f} cov {row[4]:.3f} "
              f"prep {prep_s:.2f}s pred {pred_s:.2f}s")

    # Baselines
    t0 = time.perf_counter()
    record("global", "Baseline", "global mean", np.full(len(test), global_mean), 0.0, time.perf_counter() - t0)
    record("user_mean", "Baseline", "user mean", user_mean[test[:, 0]].astype(float), 0.0, 0.0)
    record("item_mean", "Baseline", "item mean",
           np.where(item_counts[test[:, 1]] > 0, item_mean[test[:, 1]], np.nan), 0.0, 0.0)

    # Similarity matrices, timed separately
    t0 = time.perf_counter(); user_sim_a, user_common = corated_cosine(R); t_user_a = time.perf_counter() - t0
    t0 = time.perf_counter(); item_sim_a, item_common = corated_cosine(R.T); t_item_a = time.perf_counter() - t0
    t0 = time.perf_counter(); user_sim_b = mean_imputed_cosine(R, user_mean); t_user_b = time.perf_counter() - t0
    t0 = time.perf_counter(); item_sim_b = mean_imputed_cosine(R.T, item_mean); t_item_b = time.perf_counter() - t0

    variants = [
        ("UB-A", "User-Based", "A co-rated only", predict_user_based, user_sim_a, t_user_a),
        ("UB-B", "User-Based", "B mean imputation", predict_user_based, user_sim_b, t_user_b),
        ("IB-A", "Item-Based", "A co-rated only", predict_item_based, item_sim_a, t_item_a),
        ("IB-B", "Item-Based", "B mean imputation", predict_item_based, item_sim_b, t_item_b),
    ]
    for beta in BETAS:
        t0 = time.perf_counter(); s = significance_weighted(user_sim_a, user_common, beta); extra = time.perf_counter() - t0
        variants.append((f"UB-C{beta}", "User-Based", f"C weighted beta={beta}", predict_user_based, s, t_user_a + extra))
        t0 = time.perf_counter(); s = significance_weighted(item_sim_a, item_common, beta); extra = time.perf_counter() - t0
        variants.append((f"IB-C{beta}", "Item-Based", f"C weighted beta={beta}", predict_item_based, s, t_item_a + extra))

    sims = {}
    for name, method, strategy, predictor, S, prep in variants:
        sims[name] = S
        t0 = time.perf_counter()
        pred, _ = predict_test(predictor, S, R, test)
        record(name, method, strategy, pred, prep, time.perf_counter() - t0)

    # Matrix factorisation
    print("    training MF ...")
    t0 = time.perf_counter()
    val_rng = np.random.default_rng(SEED)
    is_val = val_rng.random(len(train)) < MF_VALIDATION_FRACTION
    mf_train, mf_val = train[~is_val], train[is_val]
    mf_predict, mf_history, mf_best_epoch = train_mf(mf_train, mf_val, test, num_users, num_items,
                                                     np.random.default_rng(SEED), **MF_PARAMS)
    t_mf = time.perf_counter() - t0
    print(f"    MF: {len(mf_train)} train / {len(mf_val)} validation ratings, best validation epoch {mf_best_epoch}")
    t0 = time.perf_counter()
    mf_pred = np.where(item_counts[test[:, 1]] > 0, mf_predict(test), np.nan)   # unseen movie => null
    record("MF", "MF (SGD)", f"D k={MF_PARAMS['k']}, early stop @ epoch {mf_best_epoch} (val)", mf_pred, t_mf,
           time.perf_counter() - t0)
    write_csv("mf_epochs.csv", ["epoch", "train_rmse", "val_rmse", "test_rmse"], mf_history)

    write_csv("summary.csv", ["method", "strategy", "rmse", "mae", "coverage", "rmse_covered_only",
                              "prep_seconds", "predict_seconds"], summary)

    # ---- 2. Top-5 quality -------------------------------------------------
    print("[2] Top-5 quality")
    eval_users = np.sort(np.random.default_rng(SEED).choice(np.arange(1, num_users + 1), TOP5_USERS, replace=False))
    relevant = {u: set(test[(test[:, 0] == u) & (test[:, 2] >= 4), 1].tolist()) for u in eval_users}
    top5_configs = [
        ("UB-A", predict_user_based, sims["UB-A"], 0),
        ("UB-C50", predict_user_based, sims[f"UB-C{MAIN_BETA}"], 0),
        ("IB-A", predict_item_based, sims["IB-A"], 0),
        ("IB-C50", predict_item_based, sims[f"IB-C{MAIN_BETA}"], 0),
    ]
    for threshold in POPULARITY_THRESHOLDS:
        top5_configs += [
            (f"UB-A pop>={threshold}", predict_user_based, sims["UB-A"], threshold),
            (f"UB-C50 pop>={threshold}", predict_user_based, sims[f"UB-C{MAIN_BETA}"], threshold),
            (f"IB-A pop>={threshold}", predict_item_based, sims["IB-A"], threshold),
            (f"IB-C50 pop>={threshold}", predict_item_based, sims[f"IB-C{MAIN_BETA}"], threshold),
        ]
    top5_rows = []
    top5_examples = []
    all_items = np.arange(1, num_items + 1)

    def top5_metrics(label, recommend):
        hits, pops, rare, n_recs, empty = [], [], 0, 0, 0
        for u in eval_users:
            recs = recommend(u)
            if len(recs) == 0:
                empty += 1
            hits.append(len(set(recs) & relevant[u]) / TOP_K)
            pops += item_counts[recs].tolist()
            rare += int((item_counts[recs] <= 5).sum())
            n_recs += len(recs)
            if u == eval_users[0]:
                top5_examples.append([label, int(u), " | ".join(f"{titles[i]} ({item_counts[i]})" for i in recs)])
        row = [label, float(np.mean(hits)), float(np.mean(pops)), float(np.median(pops)), rare / max(n_recs, 1), empty]
        top5_rows.append(row)
        print(f"    {label:22s} P@5 {row[1]:.3f}  mean pop {row[2]:6.1f}  median pop {row[3]:5.0f}  share<=5 {row[4]:.2f}")

    for label, predictor, S, min_pop in top5_configs:
        def recommend(u, predictor=predictor, S=S, min_pop=min_pop):
            candidates = all_items[(R[u, 1:] == 0) & (item_counts[1:] >= min_pop)]
            pred, support = predictor(S, R, u, candidates)
            ok = ~np.isnan(pred) & (support >= MIN_SUPPORT)
            c, p, s = candidates[ok], pred[ok], support[ok]
            order = np.lexsort((c, -s, -p))          # score desc, support desc, id asc
            return c[order][:TOP_K]
        top5_metrics(label, recommend)

    def recommend_mf(u):
        candidates = all_items[(R[u, 1:] == 0) & (item_counts[1:] > 0)]
        p = mf_predict(np.column_stack([np.full(len(candidates), u), candidates]))
        return candidates[np.lexsort((candidates, -p))][:TOP_K]
    top5_metrics("MF (reference)", recommend_mf)

    def recommend_popular(u):
        candidates = all_items[R[u, 1:] == 0]
        return candidates[np.lexsort((candidates, -item_counts[candidates]))][:TOP_K]
    top5_metrics("Most popular (ref.)", recommend_popular)

    write_csv("top5.csv", ["config", "precision_at_5", "mean_popularity", "median_popularity",
                           "share_recs_with_le5_ratings", "users_with_empty_list"], top5_rows)
    write_csv("top5_example_user.csv", ["config", "user", "top5 (train ratings)"], top5_examples)

    # ---- 3a. RMSE by bins -------------------------------------------------
    print("[3] cold start and sparsity")
    bin_methods = ["UB-A", "IB-A", f"IB-C{MAIN_BETA}", "MF"]
    item_bin = np.array([bin_label(c, ITEM_BINS) for c in item_counts[test[:, 1]]])
    total_user_counts = np.bincount(data[:, 0], minlength=num_users + 1)   # user bins use all ratings (min 20)
    user_bin = np.array([bin_label(c, USER_BINS) for c in total_user_counts[test[:, 0]]])
    bin_rows = []
    for kind, labels, edges in [("item_train_ratings", item_bin, ITEM_BINS), ("user_total_ratings", user_bin, USER_BINS)]:
        for _, _, label in edges:
            mask = labels == label
            row = [kind, label, int(mask.sum())]
            row += [rmse(test_preds[m][mask], truth[mask]) if mask.any() else float("nan") for m in bin_methods]
            bin_rows.append(row)
    write_csv("rmse_by_bins.csv", ["bin_type", "bin", "test_pairs"] + bin_methods, bin_rows)

    # ---- 3b. User pairs: common vs similarity ------------------------------
    a_idx, b_idx = upper_pairs(num_users)
    keep = a_idx >= 1
    a_idx, b_idx = a_idx[keep], b_idx[keep]
    pair_common = user_common[a_idx, b_idx]
    pair_sim = user_sim_a[a_idx, b_idx]
    common_rows = []
    for low, high, label in COMMON_BINS:
        mask = (pair_common >= low) & (pair_common <= high)
        common_rows.append([label, int(mask.sum()), float(mask.mean()),
                            float(pair_sim[mask].mean()) if mask.any() else float("nan")])
    one = pair_common == 1
    perfect_one = one & (pair_sim == 1.0)

    top20_short = {"A": [], f"C{MAIN_BETA}": []}
    for u in range(1, num_users + 1):
        for key, S in [("A", user_sim_a), (f"C{MAIN_BETA}", sims[f"UB-C{MAIN_BETA}"])]:
            top = neighbor_order(S[u])[:N_NEIGHBORS]
            top20_short[key].append(int((user_common[u, top] < 5).sum()))
    pair_stats = [
        ["user_pairs", int(len(pair_common))],
        ["pairs_common_eq_1", int(one.sum())],
        ["pairs_common_eq_1_and_sim_eq_1", int(perfect_one.sum())],
        ["share_of_all_pairs_common1_sim1", float(perfect_one.mean())],
        ["share_of_common1_pairs_with_sim1", float(perfect_one.sum() / max(one.sum(), 1))],
        ["top20_neighbours_common_lt5_mean_A", float(np.mean(top20_short["A"]))],
        ["top20_neighbours_common_lt5_median_A", float(np.median(top20_short["A"]))],
        ["users_with_any_top20_common_lt5_A", float(np.mean(np.array(top20_short["A"]) > 0))],
        [f"top20_neighbours_common_lt5_mean_C{MAIN_BETA}", float(np.mean(top20_short[f"C{MAIN_BETA}"]))],
        [f"top20_neighbours_common_lt5_median_C{MAIN_BETA}", float(np.median(top20_short[f"C{MAIN_BETA}"]))],
    ]
    write_csv("user_pairs_common.csv", ["common_bin", "pairs", "share", "mean_sim_A"], common_rows)
    write_csv("user_pairs_stats.csv", ["metric", "value"], pair_stats)
    for row in pair_stats:
        print(f"    {row[0]}: {row[1]}")

    # ---- 3c. New-user simulation ------------------------------------------
    sim_rng = np.random.default_rng(SEED)
    full_counts = np.bincount(data[:, 0], minlength=num_users + 1)
    target_users = np.sort(sim_rng.choice(np.nonzero(full_counts >= NEW_USER_MIN_RATINGS)[0], NEW_USER_COUNT, replace=False))
    others = data[~np.isin(data[:, 0], target_users)]
    kept, held = [], []
    for u in target_users:
        rows = data[data[:, 0] == u]
        rows = rows[sim_rng.permutation(len(rows))]
        kept.append(rows[:max(NEW_USER_LEVELS)])      # nested: level n keeps the first n
        held.append(rows[max(NEW_USER_LEVELS):])       # same test set for every level
    held = np.vstack(held)
    held_truth = held[:, 2].astype(float)
    new_user_rows = []
    for n in NEW_USER_LEVELS:
        train_n = np.vstack([others] + [k[:n] for k in kept])
        Rn = to_matrix(train_n, num_users, num_items)
        gm = train_n[:, 2].mean()
        um = row_means(Rn, gm)                          # 0 ratings => global mean
        us, uc = corated_cosine(Rn)
        isim, ic = corated_cosine(Rn.T)
        row = [n]
        for name, predictor, S in [("UB-A", predict_user_based, us), ("IB-A", predict_item_based, isim),
                                   (f"UB-C{MAIN_BETA}", predict_user_based, significance_weighted(us, uc, MAIN_BETA)),
                                   (f"IB-C{MAIN_BETA}", predict_item_based, significance_weighted(isim, ic, MAIN_BETA))]:
            pred, _ = predict_test(predictor, S, Rn, held)
            covered = ~np.isnan(pred)
            final = with_fallback(pred, um[held[:, 0]])
            row += [float(covered.mean()), rmse(final, held_truth),
                    rmse(pred[covered], held_truth[covered]) if covered.any() else float("nan")]
        row.append(rmse(um[held[:, 0]], held_truth))
        new_user_rows.append(row)
        print(f"    new user with {n:2d} ratings: UB-A cov {row[1]:.2f} RMSE {row[2]:.3f} | IB-A cov {row[4]:.2f} RMSE {row[5]:.3f}")
    nu_header = ["ratings_kept"]
    for name in ["UB-A", "IB-A", f"UB-C{MAIN_BETA}", f"IB-C{MAIN_BETA}"]:
        nu_header += [f"{name}_coverage", f"{name}_rmse_with_fallback", f"{name}_rmse_covered_only"]
    nu_header.append("fallback_only_rmse")
    write_csv("new_user.csv", nu_header, new_user_rows)
    write_csv("new_user_setup.csv", ["metric", "value"], [
        ["target_users", " ".join(map(str, target_users))], ["held_out_ratings", len(held)]])

    # ---- 4. Users vs items ------------------------------------------------
    print("[4] users vs items")
    i_a, i_b = upper_pairs(num_items)
    keep = i_a >= 1
    i_a, i_b = i_a[keep], i_b[keep]
    item_pair_common = item_common[i_a, i_b]

    def best_of(fn, repeats=3):
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter(); fn(); times.append(time.perf_counter() - t0)
        return min(times)

    eff_rows = [
        ["entities", num_users, num_items],
        ["ordered_pairs (n^2)", num_users ** 2, num_items ** 2],
        ["unordered_pairs n(n-1)/2", num_users * (num_users - 1) // 2, num_items * (num_items - 1) // 2],
        ["mean_train_ratings_per_entity", float(user_counts[1:].mean()), float(item_counts[1:].mean())],
        ["median_train_ratings_per_entity", float(np.median(user_counts[1:])), float(np.median(item_counts[1:]))],
        ["mean_common_over_all_pairs", float(pair_common.mean()), float(item_pair_common.mean())],
        ["mean_common_over_pairs_with_common>=1", float(pair_common[pair_common > 0].mean()),
         float(item_pair_common[item_pair_common > 0].mean())],
        ["share_pairs_with_common=0", float((pair_common == 0).mean()), float((item_pair_common == 0).mean())],
        ["full_corated_matrix_seconds (best of 3)", best_of(lambda: corated_cosine(R)), best_of(lambda: corated_cosine(R.T))],
    ]

    # Stability: similarities on two random halves of train
    half_rng = np.random.default_rng(SEED)
    halves = half_rng.permutation(len(train)) < len(train) // 2
    R1 = to_matrix(train[halves], num_users, num_items)
    R2 = to_matrix(train[~halves], num_users, num_items)
    stability_rows = []
    scatter = {}
    for kind, n, M1, M2 in [("user-user", num_users, R1, R2), ("item-item", num_items, R1.T, R2.T)]:
        s1, c1 = corated_cosine(M1)
        s2, c2 = corated_cosine(M2)
        pa, pb = upper_pairs(n)
        keep = pa >= 1
        pa, pb = pa[keep], pb[keep]
        for strategy in ["A", f"C{MAIN_BETA}"]:
            x, y = s1[pa, pb], s2[pa, pb]
            if strategy != "A":
                x = significance_weighted(x, c1[pa, pb], MAIN_BETA)
                y = significance_weighted(y, c2[pa, pb], MAIN_BETA)
            mask = (c1[pa, pb] >= 5) & (c2[pa, pb] >= 5)
            corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
            stability_rows.append([kind, strategy, int(mask.sum()), corr,
                                   float(np.mean(np.abs(x[mask] - y[mask]))), float(np.std(x[mask]))])
            if strategy == "A":
                scatter[kind] = (x[mask], y[mask])
            print(f"    stability {kind} {strategy}: pairs {int(mask.sum())}, Pearson r {corr:.3f}")
    write_csv("efficiency.csv", ["metric", "users", "items"], eff_rows)
    write_csv("stability.csv", ["pairs", "strategy", "pairs_common_ge5_both_halves", "pearson_r",
                                "mean_abs_diff", "std_sim_half1"], stability_rows)
    for row in eff_rows:
        print(f"    {row[0]}: users {row[1]}, items {row[2]}")

    make_plots(mf_history, mf_best_epoch, bin_rows, bin_methods, common_rows, new_user_rows, top5_rows, scatter)


# ---------------------------------------------------------------------------
# Plots (optional: needs matplotlib from week3/.venv)
# ---------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed categorical order
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def make_plots(mf_history, mf_best_epoch, bin_rows, bin_methods, common_rows, new_user_rows, top5_rows, scatter):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: skipping plots (run with week3/.venv/bin/python)")
        return

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
    })
    saved = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(OUT / name, dpi=150)
        plt.close(fig)
        saved.append(name)

    # MF learning curve
    epochs = [h[0] for h in mf_history]
    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, (label, col) in enumerate([("train", 1), ("validation", 2), ("test", 3)]):
        values = [h[col] for h in mf_history]
        ax.plot(epochs, values, color=SERIES[idx], linewidth=2, label=label)
    ax.axvline(mf_best_epoch, color=MUTED, linewidth=1, linestyle="--")
    best = mf_history[mf_best_epoch - 1]
    ax.annotate(f"best validation epoch {mf_best_epoch}\nval {best[2]:.3f}, test {best[3]:.3f}",
                (mf_best_epoch, best[3]), textcoords="offset points", xytext=(6, 18), color=INK)
    ax.set_xlabel("epoch"); ax.set_ylabel("RMSE")
    ax.set_title("Matrix factorisation (SGD): RMSE by epoch, early stopping on validation", loc="left")
    ax.legend()
    save(fig, "mf_learning_curve.png")

    # RMSE by bins
    for kind, filename, xlabel in [("item_train_ratings", "rmse_by_item_popularity.png", "train ratings of the movie"),
                                   ("user_total_ratings", "rmse_by_user_activity.png", "total ratings of the user")]:
        rows = [r for r in bin_rows if r[0] == kind]
        x = np.arange(len(rows))
        width = 0.8 / len(bin_methods)
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for m_idx, method in enumerate(bin_methods):
            values = [r[3 + m_idx] for r in rows]
            ax.bar(x + (m_idx - (len(bin_methods) - 1) / 2) * width, values, width * 0.92,
                   color=SERIES[m_idx], label=method)
        ax.set_xticks(x, [f"{r[1]}\n(n={r[2]})" for r in rows])
        ax.set_xlabel(xlabel); ax.set_ylabel("RMSE (with user-mean fallback)")
        ax.set_title(f"Test RMSE by {xlabel}", loc="left")
        ax.legend(ncols=len(bin_methods), loc="upper right")
        save(fig, filename)

    # User pairs: share of pairs and mean similarity by common
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [r[0] for r in common_rows]
    axes[0].bar(labels, [r[2] * 100 for r in common_rows], color=SERIES[0], width=0.6)
    axes[0].set_title("User pairs by number of co-rated movies", loc="left")
    axes[0].set_xlabel("co-rated movies (common)"); axes[0].set_ylabel("% of user pairs")
    for i, r in enumerate(common_rows):
        axes[0].annotate(f"{r[2] * 100:.1f}%", (i, r[2] * 100), textcoords="offset points", xytext=(0, 3), ha="center")
    sim_rows = [r for r in common_rows if r[0] != "0"]
    axes[1].plot([r[0] for r in sim_rows], [r[3] for r in sim_rows], color=SERIES[0], linewidth=2, marker="o", markersize=8)
    axes[1].set_title("Mean co-rated cosine (strategy A)", loc="left")
    axes[1].set_xlabel("co-rated movies (common)"); axes[1].set_ylabel("mean similarity")
    for r in sim_rows:
        axes[1].annotate(f"{r[3]:.3f}", (r[0], r[3]), textcoords="offset points", xytext=(0, 8), ha="center")
    save(fig, "user_pairs_common.png")

    # New-user simulation
    levels = [r[0] for r in new_user_rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for idx, (name, offset) in enumerate([("UB-A", 1), ("IB-A", 4)]):
        axes[0].plot(levels, [r[offset] * 100 for r in new_user_rows], color=SERIES[idx], linewidth=2,
                     marker="o", markersize=8, label=name)
        axes[1].plot(levels, [r[offset + 1] for r in new_user_rows], color=SERIES[idx], linewidth=2,
                     marker="o", markersize=8, label=name)
    axes[1].plot(levels, [r[-1] for r in new_user_rows], color=MUTED, linewidth=2, linestyle="--", label="fallback only")
    axes[0].set_title("Coverage for a new user (UB-A and IB-A coincide)", loc="left")
    axes[0].set_xlabel("ratings the user has given"); axes[0].set_ylabel("% of held-out pairs predicted by CF")
    axes[1].set_title("RMSE for a new user (with fallback)", loc="left")
    axes[1].set_xlabel("ratings the user has given"); axes[1].set_ylabel("RMSE")
    for ax in axes:
        ax.set_xticks(levels)
        ax.legend()
    save(fig, "new_user.png")

    # Top-5: precision and popularity
    base = [r for r in top5_rows if "pop>=" not in r[0]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = [r[0] for r in base]
    axes[0].barh(names, [r[1] for r in base], color=SERIES[0], height=0.6)
    axes[0].set_title("Precision@5 (50 users)", loc="left"); axes[0].invert_yaxis()
    axes[1].barh(names, [r[4] * 100 for r in base], color=SERIES[1], height=0.6)
    axes[1].set_title("% of recommended movies with <= 5 train ratings", loc="left"); axes[1].invert_yaxis()
    axes[1].set_yticklabels([])
    for ax, col, fmt in [(axes[0], 1, "{:.3f}"), (axes[1], 4, "{:.0%}")]:
        for i, r in enumerate(base):
            ax.annotate(fmt.format(r[col]), (r[col] if col == 1 else r[col] * 100, i),
                        textcoords="offset points", xytext=(4, 0), va="center")
    save(fig, "top5_quality.png")

    # Similarity stability
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, kind in zip(axes, ["user-user", "item-item"]):
        x, y = scatter[kind]
        hb = ax.hexbin(x, y, gridsize=40, cmap="Blues", mincnt=1, bins="log")
        r = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"{kind}: half 1 vs half 2 (r = {r:.2f}, n = {len(x)})", loc="left")
        ax.set_xlabel("similarity on half 1"); ax.set_ylabel("similarity on half 2")
        ax.grid(False)
        fig.colorbar(hb, ax=ax, label="pairs (log)")
    save(fig, "similarity_stability.png")

    print("plots:", ", ".join(saved))


if __name__ == "__main__":
    main()

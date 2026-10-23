"""Popularity baseline, Personalized PageRank (TF ops) and implicit-feedback MF (TF model).

Every method returns a dense score matrix [n_users, n_items] over the same item index; `rank_top5`
turns any score matrix into exactly five show ids per user with one shared rule.
"""
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import tensorflow as tf

from common import K


# ---------------------------------------------------------------- interaction matrix
def interaction_matrix(d, hist, lam=0.0):
    """R[u, i] = recency-weighted count of qualifying airings of show i by user u in `hist`.
    Items = shows with at least one qualifying view in `hist` (nothing else can be scored)."""
    last = hist.stop - 1
    c = d.counts(hist, weight=lambda w: np.exp(-lam * (last - w)))
    items = np.unique(c["show"].to_numpy())
    col = np.searchsorted(items, c["show"].to_numpy())
    R = np.zeros((d.n_users, len(items)), dtype=np.float32)
    R[c["user"].to_numpy(), col] = c["n"].to_numpy()
    return R, items


def popularity_scores(R):
    """Total qualifying airings per item, the same vector for every user."""
    return np.broadcast_to(R.sum(axis=0, keepdims=True), R.shape)


# ---------------------------------------------------------------- Personalized PageRank
def ppr_scores(R, alpha=0.3, tol=1e-6, max_iter=100, transform="raw"):
    """PPR on the user-show bipartite graph, one restart vector per user, all users in one batch.

    Item-to-item transition = item -> user -> item walk with edge weights w(u, i):
        P = D_i^-1 W^T D_u^-1 W      (row-stochastic over items; includes i -> i via the same user)
    Iteration (row vectors):  p <- (1 - alpha) * e_u + alpha * p P,  e_u = W_u / sum(W_u).
    `alpha` is the continuation probability, 1 - alpha the restart probability.
    """
    W = np.log1p(R) if transform == "log" else R
    W = tf.constant(W, dtype=tf.float32)
    du = tf.reduce_sum(W, axis=1, keepdims=True)             # [U, 1]
    di = tf.reduce_sum(W, axis=0, keepdims=True)             # [1, I]
    inv_du = tf.math.divide_no_nan(1.0, du)
    inv_di = tf.math.divide_no_nan(1.0, di)
    E = W * inv_du                                           # restart distribution, zero rows stay zero
    p = E
    it, delta = 0, np.inf
    while it < max_iter:
        # p P = ((p * inv_di) W^T * inv_du^T) W  -- two matmuls, never forms the I x I matrix
        step = tf.matmul(tf.matmul(p * inv_di, W, transpose_b=True) * tf.transpose(inv_du), W)
        p_new = (1.0 - alpha) * E + alpha * step
        delta = float(tf.reduce_max(tf.reduce_sum(tf.abs(p_new - p), axis=1)))
        p = p_new
        it += 1
        if delta < tol:
            break
    return p.numpy(), {"iterations": it, "max_l1_delta": delta}


# ---------------------------------------------------------------- implicit-feedback MF
def mf_scores(R, factors=32, conf_a=10.0, eps=1.0, reg=1e-2, steps=300, lr=0.05, seed=42, target="binary",
              verbose=False):
    """Weighted MF for implicit feedback (Hu, Koren, Volinsky 2008) trained as a TF model.

    preference p_ui = 1[R_ui > 0] (target="binary", Hu et al.) or log(1 + R_ui) (target="log",
    keeps repeat intensity);  confidence c_ui = 1 + conf_a * log(1 + R_ui / eps)
    loss = mean_ui c_ui (p_ui - x_u . y_i - b_i)^2 + reg * (mean_u ||x_u||^2 + mean_i ||y_i||^2)
    (full batch, Adam; both terms are averages so `reg` does not depend on matrix size)
    Score = x_u . y_i + b_i.  Users with an empty row keep their initial vector; they get the
    popularity fallback in rank_top5 anyway.
    """
    tf.keras.utils.set_random_seed(seed)
    Pm = tf.constant((R > 0).astype(np.float32) if target == "binary" else np.log1p(R).astype(np.float32))
    C = tf.constant(1.0 + conf_a * np.log1p(R / eps), dtype=tf.float32)
    U, I = R.shape
    init = tf.keras.initializers.RandomNormal(stddev=0.01, seed=seed)
    X = tf.Variable(init((U, factors)), name="user_factors")
    Y = tf.Variable(init((I, factors)), name="item_factors")
    b = tf.Variable(tf.zeros((1, I)), name="item_bias")
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    n = float(U * I)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            pred = tf.matmul(X, Y, transpose_b=True) + b
            loss = (tf.reduce_sum(C * tf.square(Pm - pred)) / n
                    + reg * (tf.reduce_sum(tf.square(X)) / U + tf.reduce_sum(tf.square(Y)) / I))
        grads = tape.gradient(loss, [X, Y, b])
        opt.apply_gradients(zip(grads, [X, Y, b]))
        return loss

    hist = []
    for s in range(steps):
        loss = float(train_step())
        hist.append(loss)
        if verbose and (s % 50 == 0 or s == steps - 1):
            print(f"  step {s:4d} loss {loss:.6f}")
    S = (tf.matmul(X, Y, transpose_b=True) + b).numpy()
    return S, {"final_loss": hist[-1], "loss_curve": hist[:: max(1, steps // 20)]}


# ---------------------------------------------------------------- shared top-5 rule
def rank_top5(S, items, pool, fallback, active, min_score=0.0, k=K):
    """Top-k items by score among candidate-pool items, repeats allowed.
    Only items with score > min_score count (PPR/popularity: 0 = unreachable/unseen; MF: -inf,
    every pool item is scored); remaining slots and users without history (`active` False)
    are filled from the popularity `fallback` list."""
    in_pool = np.isin(items, pool)
    Sm = np.where(in_pool[None, :], S, -np.inf).astype(np.float32)
    top_v, top_i = tf.math.top_k(tf.constant(Sm), k=k)
    top_v, top_i = top_v.numpy(), top_i.numpy()
    out = {}
    for u in range(S.shape[0]):
        rec = [] if not active[u] else [int(items[j]) for v, j in zip(top_v[u], top_i[u]) if v > min_score]
        for s in fallback:
            if len(rec) == k:
                break
            if s not in rec:
                rec.append(int(s))
        out[u] = rec
    return out


def timed(fn, *args, **kw):
    t = time.time()
    res = fn(*args, **kw)
    return res, time.time() - t

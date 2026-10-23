"""Exploratory MF probe: target type x regularisation (validation split)."""
import itertools, time
import numpy as np
from common import SPLITS, Data, ground_truth, mapk
from methods import mf_scores, rank_top5
from tune import setup
d = Data(); truth = ground_truth(d, SPLITS["val"][1]); R, items, pool, fb, active = setup(d, "val")
for target, f, reg, a in [("binary", 32, 0.0, 10.0), ("binary", 32, 1e-3, 10.0), ("binary", 32, 1e-2, 10.0),
                          ("log", 32, 0.0, 1.0), ("log", 32, 1e-3, 1.0), ("log", 32, 1e-2, 1.0),
                          ("log", 64, 1e-3, 1.0), ("log", 32, 1e-3, 10.0), ("log", 32, 1e-3, 0.0)]:
    t = time.time()
    S, info = mf_scores(R, factors=f, conf_a=a, reg=reg, steps=300, target=target)
    top = rank_top5(S, items, pool, fb, active, min_score=-np.inf)
    m = mapk(truth, {u: top[u] for u in truth})
    print(f"target={target:6s} k={f:3d} reg={reg:<6} conf_a={a:<4}: MAP@5 {m:.5f} loss {info['final_loss']:.5f} {time.time()-t:.1f}s", flush=True)

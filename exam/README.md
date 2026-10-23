# Final exam - TV programme recommendation on sweet.tv (Kaggle, MAP@5)

Final exam quiz of the LLM4Rec course (HSE University). Competition:
[sweet.tv - TV program recommender](https://www.kaggle.com/competitions/sweettv-tv-program-recommender)
(community prediction competition, late submissions are still scored). For each of 1,260 test users we
predict the five programmes (`tv_show_id`) they watch most in weeks 31-42, using the view log of weeks 11-30
and the broadcast schedules.

**Designated final submission:** [`submissions/final.csv`](submissions/final.csv), Kaggle ref 57567364,
public 0.43730 / **private 0.43229** (submitted 2026-10-23).

## Results

Validation = history weeks 11-20 -> target weeks 21-30 (4,030 users with at least 5 qualifying programmes in
the target window). Kaggle = refit on weeks 11-30 with the same settings, scored on weeks 31-42.

| Method | Validation MAP@5 | Kaggle public | Kaggle private | ref |
|---|---:|---:|---:|---|
| Popularity baseline | 0.2793 | 0.31942 | 0.30620 | 57567215 |
| Personalized PageRank (PPR) | 0.3664 | 0.41681 | 0.41137 | 57567219 |
| Matrix factorization (TF, log-count target) | 0.3714 | 0.43007 | 0.42526 | 57567227 |
| Hybrid: 0.8 MF + 0.2 PPR (score fusion) | 0.3733 | 0.43359 | 0.43008 | 57567229 |
| **Final: hybrid x schedule re-ranking (beta 0.3)** | **0.3903** | **0.43730** | **0.43229** | 57567364 |

Robustness split (history 11-24 -> target 25-30, 3,758 users): 0.2358 / 0.3469 / 0.3529 / 0.3549 / 0.3611 in
the same row order ([`results/val_improve4_val2.json`](results/val_improve4_val2.json)).

## Pipeline

1. `prep.py` - parse the four CSV files into numpy caches (`cache/`); times stay naive wall-clock seconds.
2. `join.py` - drop schedule rows with `tv_show_id = 0`, join every view interval to the labelled airings on
   the same channel whose interval overlaps it, sum the overlap per (user, airing), `screen = overlap / airing
   duration`. A viewing **qualifies** when `screen >= 0.8` - the rule the competition used to build its answer
   (data description page).
3. `common.py` - splits, ground truth (top-5 shows by number of qualifying airings, users with at least 5),
   Kaggle-style AP@5 / MAP@5.
4. `methods.py` - popularity, PPR on the user-show bipartite graph (TensorFlow ops, batched over all users),
   weighted MF for implicit feedback trained as a TensorFlow model (Adam, full batch), and the shared
   top-5 rule. `rerank.py` - schedule-aware re-ranking. `hybrid.py` - score / rank fusion.
5. `tune.py`, `hybrid.py`, `improve*.py` - validation sweeps (results in `results/val_*.json`).
6. `predict.py` - refit on weeks 11-30 and write `submissions/*.csv` in the template order.
7. `examples.py`, `diagnose.py`, `verify.py` - example users, headroom analysis, independent checks.

Shared decisions (identical for every method): candidate pool = labelled shows with at least one airing in
the target weeks (schedule only, never test viewing); already-watched shows **may** be recommended; users
without any qualifying history (4 of the 1,260 test users) get the popularity list; any slot a method cannot
fill with a positive score is filled from the same popularity list.

Selected settings: PPR continuation alpha 0.4, raw count weights, recency lambda 0.05/week; MF 64 factors,
target log(1 + count), confidence 1 + 3 log(1 + count), L2 0.02, 300 Adam steps, lr 0.05, seed 42; hybrid
weight 0.8 on min-max-scaled MF; re-ranking exponent 0.3 on the per-user channel airing ratio.

## Reproduce

```
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
# accept the competition rules on Kaggle, put your API token into ~/.kaggle/
.venv/bin/kaggle competitions download sweettv-tv-program-recommender -p data && (cd data && unzip *.zip)
.venv/bin/python prep.py && .venv/bin/python join.py
.venv/bin/python tune.py baselines && .venv/bin/python tune.py ppr && .venv/bin/python tune.py mf1 && .venv/bin/python tune.py mf2 64
.venv/bin/python hybrid.py && .venv/bin/python improve3.py && .venv/bin/python improve4.py val val2
.venv/bin/python predict.py      # writes submissions/*.csv (about 20 s)
.venv/bin/python examples.py && .venv/bin/python diagnose.py && .venv/bin/python verify.py
```

The Kaggle data is not redistributed here. Environment of our run: macOS 26.4.1, Apple M3 Pro (36 GB), CPU
only; Python 3.13.2, TensorFlow 2.21.0 (Keras 3.15.1), numpy 2.5.4, pandas 3.1.0, Kaggle CLI 2.2.5. Two
`predict.py` runs produced byte-identical submission files. `results/superseded/` keeps the first MF sweep
(binary target, regulariser divided by the matrix size), run with an earlier version of `mf_scores`.

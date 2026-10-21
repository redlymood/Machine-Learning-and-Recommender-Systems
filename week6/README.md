# Week 6 - Graph-based recommendations with Personalized PageRank (TensorFlow.js)

HW6 of the LLM4Rec course (HSE University). The assignment text is the course readme, kept unchanged as
[`assignment.md`](assignment.md).

Live demo: https://redlymood.github.io/Machine-Learning-and-Recommender-Systems/week6/

## What the app does

The page loads MovieLens 100K (`u.data`, `u.item`) and splits every user's ratings by time into 80% train,
10% validation and 10% test (same split as week 5). It builds a user-movie bipartite graph from the positive
training ratings (rating >= 4) and runs global PageRank and Personalized PageRank (PPR) with TensorFlow.js
tensor operations. It then compares four recommenders on one shared evaluation cohort with Recall@10:
popularity, global PageRank, PPR and the provided matrix-factorization (MF) reference. Results can be exported
as `verification.json`, `results.csv` and `recommendations.csv`.

## Run it

```
cd week6
python3 -m http.server 8000
```

Open http://localhost:8000/index.html (the data is loaded with `fetch`, so `file://` does not work). Then press,
in this order:

1. **Build graph**
2. **Run tests** (self-checks; expected 12 pass / 0 fail / 0 pending)
3. **Run alpha sweep (validation)** - the selected alpha is the best validation Recall@10, exact ties to the
   smaller alpha
4. set the **alpha slider** to the selected alpha (0.50 in our run)
5. **Evaluate Recall@10 - all four methods** (test; this first trains the provided MF reference)
6. **Recommend Top-10** for a method and a target user
7. the three **Export** buttons

Press Build graph before Run tests (otherwise check #11 stays pending). The results appear in the status line at the top and in the Self-checks panel at the bottom of the page.

## Versions used

From [`results/versions.txt`](results/versions.txt):

- TensorFlow.js 4.22.0 from jsDelivr (`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js`)
- d3 v7.9.0 (header of `vendor/d3.min.js`; used only for the graph view)
- Chrome 155.0.8059.82 (headless, `--headless=new`, webgl backend) for the checks of 2026-10-20; Chrome updated
  itself to 155.0.8059.85 at 2026-10-20 23:55 local time, which ran the live check
  ([`results/live_check.txt`](results/live_check.txt))
- Python 3.14.7 and numpy 2.5.3 for the offline checks in `tools/`

## Results (webgl)

| Method | Test Recall@10 | Mean list popularity |
|---|---|---|
| popularity | 0.0760 | 272.0 |
| global PageRank (alpha 0.50) | 0.0797 | 257.8 |
| PPR (alpha 0.50) | 0.1122 | 244.2 |
| MF, provided reference | 0.0211 | 96.8 |

Source: [`results/results.csv`](results/results.csv). Shared test cohort 861 users; 13 relevant test ratings are on
movies outside the training pool; 82 users have no eligible relevant movie. Validation Recall@10 of PPR for
alpha 0.50 / 0.70 / 0.85 / 0.95: 0.1145 / 0.1048 / 0.0900 / 0.0738 (846 users), so alpha = 0.50 is selected
([`results/selected_settings.json`](results/selected_settings.json)). Global PageRank uses the same alpha as a
disclosed control. The self-checks pass 12 / 0 / 0 ([`results/verification.json`](results/verification.json)).
An independent float64 numpy recomputation gives the same Recall@10 and the same Top-10 lists for every cohort
user ([`results/recall_reference.txt`](results/recall_reference.txt)).

## Design decisions

- **Seed contract.** On the MovieLens path, `personalizedPageRank(userId, alpha, opts)` takes a raw MovieLens user
  id and maps it with the graph's own id-to-index map, because the provided adapter in `script.js` passes raw ids.
  With `opts.transition` (the 4-node reference fixture) the first argument is a node index. Unknown ids throw.
- **Batched PPR.** The first PPR call for an alpha runs one power iteration for all 943 users at once (restart
  matrix [2554 x 943], two block matMuls with R and R^T per iteration, the square matrix is never built). The stop
  rule (L1 change < 1e-6, at most 500 iterations) is applied per column; a finished column is frozen. Results are
  cached on the graph object (9.65 MB per alpha, [`results/graph-check-gpu.txt`](results/graph-check-gpu.txt)).
- **Zero-degree nodes** (1 user, 253 movies): `p_next = alpha * (Q^T p + d v) + (1 - alpha) v`, the same rule for
  PageRank and PPR; p is never renormalized.
- **Synchronous read-back.** Results are read with `dataSync()`, because the provided harness rejects Promises;
  per iteration only the residual vector is read.
- **float32.** At alpha = 0.95, 40 of 943 PPR columns stop at the 500-iteration cap with residual 1.0e-6 to
  1.8e-6 and are reported as not converged. float64 converges for all of them, and their Top-10 lists are
  identical (max |delta p| 1.1e-8, [`results/recall_reference.txt`](results/recall_reference.txt)). The selected
  alpha 0.50 converges for all users.
- **The provided MF reference collapsed to the global mean.** Its checkpoint (epoch 1) has validation RMSE
  1.182994 and test RMSE 1.228900, the same as predicting mu, and all its scores are equal
  ([`results/mf-probe-gpu.txt`](results/mf-probe-gpu.txt)). Its Top-10 is therefore the lowest-id eligible
  movies for every user ([`results/experiments-page.txt`](results/experiments-page.txt)), and its Recall@10 is an
  id-order artifact, not a model result.
- **HW5 MF as an extra row (offline).** The week 5 model (epoch 2, validation RMSE 1.008837, test RMSE 1.045873,
  [`results/hw5-retrain.txt`](results/hw5-retrain.txt)) is scored in Python on the same candidates, exclusions
  and cohort: Recall@10 0.0175 ([`results/hw5_mf_recall.txt`](results/hw5_mf_recall.txt)). It is not part of the
  web app.

## Known issues

- The provided MF code never disposes its Adam optimizer, so 10 tensors stay in memory after training
  ([`results/experiments-page.txt`](results/experiments-page.txt)); the graph code itself leaves 0 tensors
  ([`results/graph-check-gpu.txt`](results/graph-check-gpu.txt)).
- The provided two-user demo always uses the two smallest cohort ids (users 1 and 2), not a chosen pair.

## Files

- `index.html`, `script.js`, `graph.js`, `data.js`, `style.css`, `vendor/` - the app (TODO stubs implemented in
  `graph.js` and `script.js`)
- `results/` - the page exports and the outputs of the checks
- `tools/` - Python checks (`independent_stats.py`, `cohort_precheck.py`, `ppr_reference.py`,
  `recall_reference.py`, `user_histories.py`, `hw5_mf_recall.py`, `extra_checks.py`, `fixture_tolerance.py`) and
  the headless-Chrome drivers (`graph_check.py`, `experiments_page.py`, `mf_probe.py`, `hw5_retrain.py`,
  `bench_matmul.py`, `live_check.py`; they need Google Chrome and perl). Run them from `week6/tools/`, e.g.
  `python3 extra_checks.py`; they write into `results/`.

Dataset: MovieLens 100K, GroupLens Research, University of Minnesota.

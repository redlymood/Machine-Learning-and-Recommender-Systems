"""Part C: retrain the HW5 MF (ML-English/week5, unchanged) once on webgl and export it.

Serves week5/ read-only, clicks "Train" with the page defaults (one synchronous
click, ~97 s on webgl in HW5), checks that the selected checkpoint reproduces
the HW5 result (epoch 2, val RMSE 1.0088, test RMSE 1.0459), and exports mu,
P and Q to results/hw5_mf_factors.json. Also compares the factors with the HW5
export (hw5-tools/results/factors-default-seed42.json).

Usage: python3 hw5_retrain.py   (writes results/hw5-retrain.txt, results/hw5_mf_factors.json)
"""
import json
import time

from paths import FACTORS, RESULTS, TOOLS, WEEK5
from week6_session import page_errors, week6_page

HW5_EXPORT = TOOLS.parent / "hw5-tools" / "results" / "factors-default-seed42.json"
EXPECTED = {"epoch": 2, "valRMSE": 1.0088, "testRMSE": 1.0459}

TRAIN_JS = r"""(() => {
  const inputs = Object.fromEntries(["epochs", "lr", "reg", "batch-size"].map((id) => [id, document.getElementById(id).value]));
  const t0 = performance.now();
  document.getElementById("train").click();
  const seconds = (performance.now() - t0) / 1000;
  return JSON.stringify({ inputs, backend: tf.getBackend(), seconds, status: document.getElementById("status").textContent,
    history: trainingHistory.map((r) => ({ ...r })), best: bestCheckpoint ? { ...bestCheckpoint } : null,
    valRMSE: computeRMSEFor(valSet), testRMSE: computeRMSEFor(testSet), tensors: tf.memory().numTensors });
})()"""

EXPORT_JS = r"""(() => JSON.stringify({
  note: "HW5 MF retrained in the HW6 session with the week5 page defaults; selected checkpoint, float32 via arraySync.",
  backend: tf.getBackend(), seed: 42, epoch: bestCheckpoint.epoch, valRMSE: bestCheckpoint.valRMSE,
  testRMSE: computeRMSEFor(testSet), globalMean: model.globalMean,
  userFactors: model.userFactors.arraySync(), itemFactors: model.itemFactors.arraySync(),
}))()"""


def main():
    t0 = time.time()
    with week6_page(use_gpu=True, alarm=178, tag="hw5-retrain", folder=WEEK5) as (chrome, session):
        train = chrome.evaluate(TRAIN_JS, session, timeout=170)
        factors = chrome.evaluate(EXPORT_JS, session, timeout=60)
        errors = page_errors(chrome, session)
    wall = time.time() - t0

    best = train["best"]
    reproduced = (best["epoch"] == EXPECTED["epoch"] and round(best["valRMSE"], 4) == EXPECTED["valRMSE"]
                  and round(train["testRMSE"], 4) == EXPECTED["testRMSE"])
    old = json.loads(HW5_EXPORT.read_text()) if HW5_EXPORT.exists() else None  # local HW5 export only
    diff = None if old is None else max(max(abs(a - b) for a, b in zip(ra, rb))
                                        for key in ("userFactors", "itemFactors") for ra, rb in zip(factors[key], old[key]))
    FACTORS.write_text(json.dumps(factors))
    lines = [f"week5 Train on {train['backend']}: {train['seconds']:.1f} s (wall {wall:.1f} s), inputs {train['inputs']}",
             f"status: {train['status']}",
             f"selected checkpoint: epoch {best['epoch']}, val RMSE {best['valRMSE']:.6f}, test RMSE {train['testRMSE']:.6f} "
             f"(in-memory val {train['valRMSE']:.6f})",
             f"reproduces HW5 (epoch 2, val 1.0088, test 1.0459): {reproduced}",
             (f"max |factor difference| vs the HW5 export: {diff:.3e}; mu {factors['globalMean']!r} vs {old['globalMean']!r}"
              if old else "HW5 export not available here; factor comparison skipped"),
             "epoch | train RMSE | val RMSE"]
    lines += [f"  {h['epoch']:>2} | {h['trainRMSE']:.6f} | {h['valRMSE']:.6f}" for h in train["history"]]
    lines.append(f"tensors after: {train['tensors']}; page errors: {errors}")
    text = "\n".join(lines) + "\n"
    (RESULTS / "hw5-retrain.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

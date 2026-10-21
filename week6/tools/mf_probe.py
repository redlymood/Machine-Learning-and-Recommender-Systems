"""One training run of the PROVIDED week6 MF reference on webgl, then a collapse probe.

Calls only the provided entry point `hw6EnsureMfForNeed()` (script.js); no MF
code is changed. Records the training time, MF_INFO and the per-epoch history,
the RMSE of the restored checkpoint (provided `hw6MfRmse`, unclamped), the
mu-only baseline RMSE, the mean row norms of P and Q, and the spread of
`mfScores` over the training candidates for users 1 and 98.

Usage: python3 mf_probe.py   (writes results/mf-probe-gpu.{json,txt})
"""
import json
import time

from page_check import RESULTS
from week6_session import page_errors, week6_page

PROBE_JS = r"""(async () => {
  const t0 = performance.now();
  const model = await hw6EnsureMfForNeed();
  const seconds = (performance.now() - t0) / 1000;
  const status = document.getElementById("status").textContent;
  const info = { ...MF_INFO, history: undefined };
  if (!model) return JSON.stringify({ seconds, trained: false, info, status });
  const mu = HW6_MF_MODEL.globalMean;
  const rmseMu = (rows) => Math.sqrt(rows.reduce((a, r) => a + (r.rating - mu) ** 2, 0) / rows.length);
  const P = HW6_MF_MODEL.userFactors.arraySync();
  const Q = HW6_MF_MODEL.itemFactors.arraySync();
  const norm = (row) => Math.sqrt(row.reduce((a, x) => a + x * x, 0));
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const pTrain = HW6_STATE.trainUserIds.map((u) => norm(P[u]));
  const qTrain = HW6_STATE.candidateIds.map((m) => norm(Q[m]));
  const scoreStats = (u) => {
    const s = Object.values(mfScores(u));
    const m = mean(s);
    return { n: s.length, mean: m, std: Math.sqrt(mean(s.map((x) => (x - m) ** 2))),
             min: Math.min(...s), max: Math.max(...s) };
  };
  return JSON.stringify({
    seconds, trained: true, backend: tf.getBackend(), status, info, history: MF_INFO.history,
    rmse: { train: hw6MfRmse(HW6_SPLIT.train), val: hw6MfRmse(HW6_SPLIT.val), test: hw6MfRmse(HW6_SPLIT.test) },
    baseline: { mu, train: rmseMu(HW6_SPLIT.train), val: rmseMu(HW6_SPLIT.val), test: rmseMu(HW6_SPLIT.test) },
    norms: { pMeanTrainUsers: mean(pTrain), pMaxTrainUsers: Math.max(...pTrain),
             qMeanTrainMovies: mean(qTrain), qMaxTrainMovies: Math.max(...qTrain),
             pMeanAllRows: mean(P.map(norm)), qMeanAllRows: mean(Q.map(norm)), rowsP: P.length, rowsQ: Q.length },
    scores: { "1": scoreStats(1), "98": scoreStats(98) },
    tensors: tf.memory().numTensors,
  });
})()"""


def main():
    t_start = time.time()
    with week6_page(use_gpu=True, alarm=170, tag="mf-probe") as (chrome, session):
        report = chrome.evaluate(PROBE_JS, session, timeout=160, await_promise=True)
        report["errors"] = page_errors(chrome, session)
    report["wallSeconds"] = round(time.time() - t_start, 1)
    (RESULTS / "mf-probe-gpu.json").write_text(json.dumps(report, indent=1))

    r = report
    lines = [f"MF probe ({r.get('backend')}): trained {r['trained']} in {r['seconds']:.1f} s (wall {r['wallSeconds']} s)",
             f"status: {r['status']}", f"MF_INFO: {r['info']}"]
    if r["trained"]:
        lines.append("epoch | train RMSE | val RMSE")
        lines += [f"  {h['epoch']:>2} | {h['trainRMSE']:.6f} | {h['valRMSE']:.6f}" for h in r["history"]]
        lines.append(f"checkpoint (restored) RMSE: train {r['rmse']['train']:.6f}, val {r['rmse']['val']:.6f}, "
                     f"test {r['rmse']['test']:.6f}")
        lines.append(f"mu-only baseline (mu {r['baseline']['mu']:.6f}): train {r['baseline']['train']:.6f}, "
                     f"val {r['baseline']['val']:.6f}, test {r['baseline']['test']:.6f}")
        lines.append(f"row norms: {r['norms']}")
        lines.append(f"mfScores over candidates: {r['scores']}")
        lines.append(f"tensors after: {r['tensors']}")
    lines.append(f"page errors: {r['errors']}")
    text = "\n".join(lines) + "\n"
    (RESULTS / "mf-probe-gpu.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

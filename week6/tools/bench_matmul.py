"""Micro-benchmark of the batched-PPR matMul pair at the real HW6 shapes (webgl).

Not a PPR implementation: it times only the two products one propagation step
needs, R @ X_movies ([nU x nM] @ [nM x B]) and R^T @ X_users (via transposeA,
[nM x nU] @ [nU x B]), with R built from the page's own training positives
(HW6_SPLIT.train, rating >= 4) and random X. B = 1 is "one user at a time",
B = nU is "all restart vectors as columns". Readback modes: dataSync every
iteration (what a synchronous personalizedPageRank would do for its residual;
at B = 1 the gap to "none" is the readback latency), await data() every
iteration, or none until the end (pure GPU throughput).

Usage: python3 bench_matmul.py [--gpu]   (writes results/bench-<mode>.json)
"""
import json
import subprocess
import sys
import time
import urllib.request

from cdp import DevToolsPipe
from page_check import READY_JS, RESULTS, WEEK6, free_port

BENCH_JS = r"""(async () => {
  const users = HW6_STATE.trainUserIds, ids = HW6_STATE.candidateIds;
  const nU = users.length, nM = ids.length;
  const uIdx = new Map(users.map((u, i) => [u, i])), mIdx = new Map(ids.map((m, i) => [m, i]));
  const buf = new Float32Array(nU * nM);
  for (const r of HW6_SPLIT.train) if (r.rating >= 4) buf[uIdx.get(r.userId) * nM + mIdx.get(r.movieId)] = 1;
  const tensorsBefore = tf.memory().numTensors;
  const R = tf.tensor2d(buf, [nU, nM]);
  const pair = (Xu, Xm) => tf.tidy(() => tf.matMul(R, Xm).sum().add(tf.matMul(R, Xu, true, false).sum()));
  const out = { backend: tf.getBackend(), nU, nM, edges: buf.reduce((a, b) => a + b, 0), runs: [] };

  async function run(B, iters, mode) {
    const Xu = tf.randomUniform([nU, B]), Xm = tf.randomUniform([nM, B]);
    for (let k = 0; k < 3; k++) { const w = pair(Xu, Xm); await w.data(); w.dispose(); }  // shader warm-up
    const t0 = performance.now();
    let last = null;
    for (let k = 0; k < iters; k++) {
      const s = pair(Xu, Xm);
      if (mode === "dataSync") s.dataSync();
      else if (mode === "data") await s.data();
      if (last) last.dispose();
      last = s;
    }
    await last.data(); last.dispose();
    const ms = (performance.now() - t0) / iters;
    Xu.dispose(); Xm.dispose();
    const gflop = 2 * 2 * nU * nM * B / 1e9;
    out.runs.push({ B, iters, readback: mode, msPerIter: +ms.toFixed(3), gflopPerIter: +gflop.toFixed(4),
                    gflops: +(gflop / (ms / 1000)).toFixed(1) });
  }

  const t0 = performance.now();
  await run(1, 300, "dataSync");
  await run(1, 300, "data");
  await run(1, 300, "none");
  await run(nU, 40, "dataSync");
  await run(nU, 40, "none");
  R.dispose();
  out.totalSeconds = +((performance.now() - t0) / 1000).toFixed(2);
  out.tensorsLeaked = tf.memory().numTensors - tensorsBefore;
  return JSON.stringify(out);
})()"""


def main():
    use_gpu = "--gpu" in sys.argv
    mode = "gpu" if use_gpu else "nogpu"
    port = free_port()
    log = open(RESULTS / f"http-server-bench-{mode}.log", "w")
    server = subprocess.Popen(["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1",
                               "--directory", str(WEEK6)], stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}/index.html"
    chrome = None
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen(url, timeout=1).read()
                break
            except OSError:
                time.sleep(0.2)
        chrome = DevToolsPipe(alarm_seconds=120, disable_gpu=not use_gpu)
        target = chrome.call("Target.createTarget", {"url": "about:blank"})["targetId"]
        session = chrome.call("Target.attachToTarget", {"targetId": target, "flatten": True})["sessionId"]
        chrome.call("Runtime.enable", session=session)
        chrome.call("Page.enable", session=session)
        t0 = time.time()
        chrome.call("Page.navigate", {"url": url}, session=session)
        chrome.wait_for("Page.loadEventFired", timeout=60)
        while not chrome.evaluate(READY_JS, session)["status"].startswith(("Dataset ready", "Error")):
            if time.time() - t0 > 60:
                raise TimeoutError("page not ready")
            time.sleep(0.5)
        report = chrome.evaluate(BENCH_JS, session, timeout=90, await_promise=True)
    finally:
        if chrome:
            chrome.close()
        server.terminate()
        server.wait(timeout=10)
        log.close()
    (RESULTS / f"bench-{mode}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}))
    for row in report["runs"]:
        print(row)


if __name__ == "__main__":
    main()

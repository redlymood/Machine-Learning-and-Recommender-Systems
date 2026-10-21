"""Check the published GitHub Pages copy of week6 in headless Chrome (webgl).

Opens the live URL (no local server), then runs the same flow as the local
experiments: "Build graph", "Run tests", the validation alpha sweep, slider :=
selected alpha, and the four-method TEST comparison (hw6HandleRecall, which
trains the provided MF). Compares the selected alpha and the test Recall@10 /
cohort / exclusion numbers with the local export (results.csv).

Usage: python3 live_check.py [URL]   (writes results/live_check.txt)
"""
import csv
import sys
import time

from cdp import DevToolsPipe
from page_check import READY_JS
from paths import EXPORTS, RESULTS

URL = "https://redlymood.github.io/Machine-Learning-and-Recommender-Systems/week6/"

FLOW_JS = r"""(async () => {
  const t0 = performance.now();
  document.getElementById("build-graph").click();
  document.getElementById("run-tests").click();
  const v = HW6_STATE.lastVerification;
  document.getElementById("alpha-sweep-btn").click();
  let best = null, bestRecall = -Infinity;
  for (const r of HW6_STATE.alphaSweepRows) if (r.error === null && r.recall > bestRecall) { bestRecall = r.recall; best = r.alpha; }
  const slider = document.getElementById("alpha");
  slider.value = String(best);
  slider.dispatchEvent(new Event("input"));
  await hw6HandleRecall();
  return JSON.stringify({ seconds: (performance.now() - t0) / 1000, backend: tf.getBackend(),
    href: location.href, implemented: typeof graphPowerIterate === "function",
    checks: { passed: v.passed, failed: v.failed, pending: v.pending }, best,
    sweep: HW6_STATE.alphaSweepRows.map((r) => ({ alpha: r.alpha, recall: r.recall })),
    rows: HW6_STATE.comparisonRows.map((r) => ({ method: r.method, alpha: r.alpha, recall: r.recall,
      cohortSize: r.cohortSize, excludedUnseen: r.excludedUnseen, excludedNoEligible: r.excludedNoEligible })) });
})()"""


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else URL
    chrome = DevToolsPipe(alarm_seconds=170, disable_gpu=False)
    try:
        browser = chrome.call("Browser.getVersion")["product"]
        target = chrome.call("Target.createTarget", {"url": "about:blank"})["targetId"]
        session = chrome.call("Target.attachToTarget", {"targetId": target, "flatten": True})["sessionId"]
        for domain in ("Page", "Runtime", "Network"):
            chrome.call(f"{domain}.enable", session=session)
        chrome.call("Network.setCacheDisabled", {"cacheDisabled": True}, session=session)
        t0 = time.time()
        chrome.call("Page.navigate", {"url": url}, session=session)
        chrome.wait_for("Page.loadEventFired", timeout=60)
        while not chrome.evaluate(READY_JS, session)["status"].startswith(("Dataset ready", "Error")):
            if time.time() - t0 > 60:
                raise TimeoutError("live page did not report 'Dataset ready'")
            time.sleep(0.5)
        live = chrome.evaluate(FLOW_JS, session, timeout=150, await_promise=True)
    finally:
        chrome.close()

    local = {}
    with open(EXPORTS / "results.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == "test" and row["recall_at_10"]:
                local[row["method"]] = row
    lines = [f"live {live['href']} | {browser} | backend {live['backend']} | implemented graph.js {live['implemented']} | "
             f"flow {live['seconds']:.1f} s",
             f"self-checks: pass {live['checks']['passed']}, fail {live['checks']['failed']}, "
             f"pending {live['checks']['pending']}",
             f"validation sweep: {[(x['alpha'], round(x['recall'], 6)) for x in live['sweep']]}; selected alpha {live['best']}",
             "method | live Recall@10 | local Recall@10 | identical | cohort | excluded_unseen | excluded_no_eligible"]
    same_all = True
    for r in live["rows"]:
        loc = local[r["method"]]
        same = repr(r["recall"]) == loc["recall_at_10"] or r["recall"] == float(loc["recall_at_10"])
        same_all &= same and str(r["cohortSize"]) == loc["cohort_size"]
        lines.append(f"{r['method']} | {r['recall']!r} | {loc['recall_at_10']} | {same} | {r['cohortSize']} | "
                     f"{r['excludedUnseen']} | {r['excludedNoEligible']}")
    lines.append(f"selected alpha equals local ({local['ppr']['alpha']}): {f'{live['best']:.2f}' == local['ppr']['alpha']}; "
                 f"all test rows identical: {same_all}")
    text = "\n".join(lines) + "\n"
    (RESULTS / "live_check.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

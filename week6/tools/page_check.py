"""Serve ML-English/week6 over HTTP and inspect it in headless Chrome (read-only).

Adapted from hw5-tools/page_check.py. Starts `python3 -m http.server` on a free
localhost port (data.js needs fetch), opens the page through the DevTools pipe
(cdp.py), waits for the bootstrap to finish, and records:
  - the TensorFlow.js version / backend and the WebGL renderer string;
  - which ops HW6 needs exist in the loaded `tf`, and any PageRank / graph /
    sparse-looking exports;
  - the rendered status, dataset summary and graph summary;
  - MF_INFO (to confirm the MF reference was NOT trained on load);
  - the "Run tests" self-check log and its pass / fail / pending summary;
  - console messages, uncaught exceptions and browser log entries.
It never clicks anything that trains MF. Nothing in week6/ is modified.

Usage: python3 page_check.py [--gpu]   (default adds --disable-gpu)
Writes results/page-<nogpu|gpu>.json.
"""
import json
import socket
import subprocess
import sys
import time
import urllib.request

from cdp import DevToolsPipe

from paths import RESULTS, TOOLS, WEEK6  # noqa: F401 (re-exported to the other drivers)

READY_JS = r"""(() => JSON.stringify({
  status: document.getElementById("status").textContent,
  tfStatus: document.getElementById("tf-status").textContent,
}))()"""

PAGE_JS = r"""(() => {
  const text = (id) => document.getElementById(id).innerText.replace(/\s+/g, " ").trim();
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
  let renderer = null;
  if (gl) {
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    renderer = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  }
  return JSON.stringify({
    tf: { version: tf.version.tfjs, backend: tf.getBackend(), tfStatusText: text("tf-status"),
          numTensors: tf.memory().numTensors },
    webgl: { contextAvailable: !!gl, renderer },
    rendered: { status: text("status"), dataset: text("dataset-summary-body"),
                graph: text("graph-summary-body"),
                userOptions: document.querySelectorAll("#user-select option").length },
    mfInfo: { ...MF_INFO, history: MF_INFO.history.length },
    mfModelPresent: HW6_MF_MODEL !== null,
    graphActive: getGraph() !== null,
  });
})()"""

# Which of the ops HW6 needs exist in the loaded tf namespace, and whether any
# export looks like a ready-made PageRank / graph / random-walk / sparse op.
TF_OPS_JS = r"""(() => {
  const ops = ["matMul", "tidy", "dispose", "keep", "sum", "abs", "sub", "add", "mul", "div",
               "tensor2d", "transpose", "oneHot", "ready", "getBackend", "memory"];
  const t = tf.tensor1d([1]);
  const methods = ["data", "dataSync", "array", "arraySync", "dispose"];
  const out = {
    version: tf.version.tfjs,
    ops: Object.fromEntries(ops.map((name) => [name, typeof tf[name] === "function"])),
    tensorMethods: Object.fromEntries(methods.map((name) => [name, typeof t[name] === "function"])),
    exportCount: Object.keys(tf).length,
    graphLike: Object.keys(tf).filter((k) => /page.?rank|graph|walk|sparse/i.test(k)).sort(),
  };
  t.dispose();
  return JSON.stringify(out);
})()"""

TESTS_JS = r"""(() => {
  document.getElementById("run-tests").click();
  const v = HW6_STATE.lastVerification;
  return JSON.stringify({
    log: document.getElementById("self-checks").textContent,
    passed: v.passed, failed: v.failed, pending: v.pending,
    checks: v.checks.map((c) => ({ name: c.name, status: c.status, detail: c.detail })),
    numTensorsAfter: tf.memory().numTensors,
    mfTrainedAfter: MF_INFO.trainedAt,
  });
})()"""


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    use_gpu = "--gpu" in sys.argv
    mode = "gpu" if use_gpu else "nogpu"
    RESULTS.mkdir(exist_ok=True)
    port = free_port()
    log = open(RESULTS / f"http-server-{mode}.log", "w")
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
        chrome = DevToolsPipe(alarm_seconds=150, disable_gpu=not use_gpu)
        report = {"mode": mode, "url": url, "browser": chrome.call("Browser.getVersion")["product"]}
        target = chrome.call("Target.createTarget", {"url": "about:blank"})["targetId"]
        session = chrome.call("Target.attachToTarget", {"targetId": target, "flatten": True})["sessionId"]
        for domain in ("Page", "Runtime", "Network", "Log"):
            chrome.call(f"{domain}.enable", session=session)
        chrome.call("Network.setCacheDisabled", {"cacheDisabled": True}, session=session)
        t0 = time.time()
        chrome.call("Page.navigate", {"url": url}, session=session)
        chrome.wait_for("Page.loadEventFired", timeout=90)
        while True:
            state = chrome.evaluate(READY_JS, session)
            if (state["status"].startswith(("Dataset ready", "Error"))
                    or "failed" in state["tfStatus"] or time.time() - t0 > 90):
                break
            time.sleep(0.5)
        report["secondsToReady"] = round(time.time() - t0, 1)
        report["page"] = chrome.evaluate(PAGE_JS, session)
        report["tfOps"] = chrome.evaluate(TF_OPS_JS, session)
        report["selfChecks"] = chrome.evaluate(TESTS_JS, session)
        chrome.call("Runtime.evaluate", {"expression": "0"}, session=session)  # flush pending events
        report["network"] = [
            {"url": e["params"]["response"]["url"], "status": e["params"]["response"]["status"]}
            for e in chrome.events if e.get("method") == "Network.responseReceived"]
        report["console"] = [
            {"type": e["params"]["type"],
             "text": " ".join(str(a.get("value", a.get("description", ""))) for a in e["params"]["args"])[:300]}
            for e in chrome.events if e.get("method") == "Runtime.consoleAPICalled"]
        report["exceptions"] = [e["params"]["exceptionDetails"].get("text")
                                for e in chrome.events if e.get("method") == "Runtime.exceptionThrown"]
        report["logEntries"] = [f'{e["params"]["entry"]["level"]}: {e["params"]["entry"]["text"][:200]}'
                                for e in chrome.events if e.get("method") == "Log.entryAdded"]
    finally:
        if chrome:
            chrome.close()
        server.terminate()
        server.wait(timeout=10)
        log.close()

    (RESULTS / f"page-{mode}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    page, checks = report["page"], report["selfChecks"]
    print(f"[{mode}] {report['browser']} | ready in {report['secondsToReady']} s")
    for row in report["network"]:
        if "tfjs" in row["url"] or row["status"] >= 400:
            print(f"net: {row['status']} {row['url']}")
    print("tf:", page["tf"])
    print("webgl:", page["webgl"])
    print("tf ops:", report["tfOps"])
    print("status:", page["rendered"]["status"])
    print("MF on load: model present", page["mfModelPresent"], "| trainedAt", page["mfInfo"]["trainedAt"],
          "| checkpoint", page["mfInfo"]["checkpoint"], "| after tests", checks["mfTrainedAfter"])
    print(f"self-checks: pass {checks['passed']}, fail {checks['failed']}, pending {checks['pending']}")
    for c in checks["checks"]:
        print(f"  {c['status']:<8}{c['name']}  |  {c['detail'][:110]}")
    print("console:", report["console"])
    print("exceptions:", report["exceptions"], "| log:", report["logEntries"])


if __name__ == "__main__":
    main()

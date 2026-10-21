"""Open ML-English/week6 (or another week folder) in headless Chrome over the DevTools pipe.

Shared by the HW6 check scripts: serves week6/ with `python3 -m http.server`,
starts Chrome under the perl alarm from cdp.py, waits until the page bootstrap
reports "Dataset ready", and yields (chrome, session). `page_errors` collects
console messages, uncaught exceptions and browser log entries.
"""
import contextlib
import subprocess
import time
import urllib.request

from cdp import DevToolsPipe
from page_check import READY_JS, RESULTS, WEEK6, free_port


@contextlib.contextmanager
def week6_page(use_gpu=True, alarm=150, tag="session", folder=WEEK6, download_dir=None):
    RESULTS.mkdir(exist_ok=True)
    port = free_port()
    log = open(RESULTS / f"http-server-{tag}.log", "w")
    server = subprocess.Popen(["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1",
                               "--directory", str(folder)], stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}/index.html"
    chrome = None
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen(url, timeout=1).read()
                break
            except OSError:
                time.sleep(0.2)
        chrome = DevToolsPipe(alarm_seconds=alarm, disable_gpu=not use_gpu)
        target = chrome.call("Target.createTarget", {"url": "about:blank"})["targetId"]
        session = chrome.call("Target.attachToTarget", {"targetId": target, "flatten": True})["sessionId"]
        for domain in ("Page", "Runtime", "Log"):
            chrome.call(f"{domain}.enable", session=session)
        if download_dir is not None:
            # Let the page's own <a download> exporters write into download_dir.
            chrome.call("Browser.setDownloadBehavior", {"behavior": "allow", "downloadPath": str(download_dir),
                                                        "eventsEnabled": True})
        t0 = time.time()
        chrome.call("Page.navigate", {"url": url}, session=session)
        chrome.wait_for("Page.loadEventFired", timeout=60)
        while not chrome.evaluate(READY_JS, session)["status"].startswith(("Dataset ready", "Error")):
            if time.time() - t0 > 60:
                raise TimeoutError(f"{folder.name} page did not report 'Dataset ready'")
            time.sleep(0.3)
        yield chrome, session
    finally:
        if chrome:
            chrome.close()
        server.terminate()
        server.wait(timeout=10)
        log.close()


def page_errors(chrome, session):
    chrome.call("Runtime.evaluate", {"expression": "0"}, session=session)  # flush pending events
    console = [{"type": e["params"]["type"],
                "text": " ".join(str(a.get("value", a.get("description", ""))) for a in e["params"]["args"])[:300]}
               for e in chrome.events if e.get("method") == "Runtime.consoleAPICalled"]
    exceptions = [e["params"]["exceptionDetails"].get("text")
                  for e in chrome.events if e.get("method") == "Runtime.exceptionThrown"]
    log = [f'{e["params"]["entry"]["level"]}: {e["params"]["entry"]["text"][:200]}'
           for e in chrome.events if e.get("method") == "Log.entryAdded"]
    return {"console": console, "exceptions": exceptions, "log": log}

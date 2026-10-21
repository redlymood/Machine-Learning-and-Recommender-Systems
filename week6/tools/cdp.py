"""Minimal Chrome DevTools Protocol client over --remote-debugging-pipe.

Reused from hw4-tools/live-demo-check.py (fd 3 = commands to Chrome, fd 4 =
replies; messages are JSON terminated by NUL), so no extra Python packages are
needed. Changes for HW5: Chrome is started under a perl alarm (headless Chrome
does not always exit on its own), always with --use-mock-keychain, and the GPU
flag is optional so the TF.js backend can be checked with and without
--disable-gpu.
"""
import json
import os
import select
import shutil
import subprocess
import tempfile
import time

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


class DevToolsPipe:
    def __init__(self, alarm_seconds=180, disable_gpu=True):
        to_chrome_r, self.to_chrome_w = os.pipe()
        self.from_chrome_r, from_chrome_w = os.pipe()
        self.profile = tempfile.mkdtemp(prefix="hw5-cdp.")
        flags = ["--headless=new", "--remote-debugging-pipe", "--no-first-run",
                 "--no-default-browser-check", "--use-mock-keychain", "--password-store=basic",
                 "--disable-extensions", "--disable-sync", f"--user-data-dir={self.profile}"]
        if disable_gpu:
            flags.append("--disable-gpu")
        # The shell maps our pipe ends to fds 3 and 4, which Chrome expects; the
        # perl alarm survives exec and kills Chrome if it hangs.
        shell = (f'exec perl -e "alarm {int(alarm_seconds)}; exec @ARGV" "$0" "$@" '
                 f"3<&{to_chrome_r} 4>&{from_chrome_w}")
        self.proc = subprocess.Popen(["/bin/sh", "-c", shell, CHROME, *flags, "about:blank"],
                                     pass_fds=(to_chrome_r, from_chrome_w),
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.close(to_chrome_r)
        os.close(from_chrome_w)
        self.buffer, self.next_id, self.events = b"", 0, []

    def _receive(self, timeout):
        deadline = time.time() + timeout
        while b"\0" not in self.buffer:
            ready, _, _ = select.select([self.from_chrome_r], [], [], max(0.0, deadline - time.time()))
            if not ready:
                raise TimeoutError("no reply from Chrome")
            chunk = os.read(self.from_chrome_r, 1 << 20)
            if not chunk:
                raise EOFError("Chrome closed the pipe")
            self.buffer += chunk
        raw, self.buffer = self.buffer.split(b"\0", 1)
        return json.loads(raw)

    def call(self, method, params=None, session=None, timeout=60):
        self.next_id += 1
        message = {"id": self.next_id, "method": method, "params": params or {}}
        if session:
            message["sessionId"] = session
        os.write(self.to_chrome_w, json.dumps(message).encode() + b"\0")
        while True:
            reply = self._receive(timeout)
            if reply.get("id") == self.next_id:
                if "error" in reply:
                    raise RuntimeError(f"{method}: {reply['error']}")
                return reply.get("result", {})
            self.events.append(reply)

    def wait_for(self, event, timeout=60):
        for i, message in enumerate(self.events):
            if message.get("method") == event:
                return self.events.pop(i)
        while True:
            message = self._receive(timeout)
            if message.get("method") == event:
                return message
            self.events.append(message)

    def evaluate(self, expression, session, timeout=120, await_promise=False):
        """Evaluate JS in the page and return its JSON-decoded string result."""
        result = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True,
                                                "awaitPromise": await_promise},
                           session=session, timeout=timeout)
        if "exceptionDetails" in result:
            raise RuntimeError(json.dumps(result["exceptionDetails"])[:2000])
        value = result["result"].get("value")
        return json.loads(value) if isinstance(value, str) else value

    def close(self):
        try:
            self.call("Browser.close", timeout=10)
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        shutil.rmtree(self.profile, ignore_errors=True)

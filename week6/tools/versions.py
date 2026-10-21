"""Record the dependency versions actually used by the HW6 app and tools.

TF.js: the script tag in index.html; d3: the header of vendor/d3.min.js;
Chrome: the Browser.getVersion of the recorded headless run (results/page-gpu.json),
the binary installed now (cdp.CHROME --version) and the install time of every
version folder of the Chrome framework; Python and numpy: the interpreter running
this script.

Usage: python3 versions.py   (writes results/versions.txt)
"""
import datetime
import json
import platform
import re
import subprocess
from pathlib import Path

import numpy

from cdp import CHROME
from paths import RESULTS, WEEK6


def main():
    html = (WEEK6 / "index.html").read_text()
    tfjs = re.search(r'src="(https://cdn\.jsdelivr\.net/npm/@tensorflow/tfjs@[^"]+)"', html).group(1)
    d3_header = (WEEK6 / "vendor" / "d3.min.js").read_text()[:200].splitlines()[0]
    chrome = subprocess.run([CHROME, "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
    recorded = RESULTS / "page-gpu.json"
    recorded = json.loads(recorded.read_text())["browser"] if recorded.exists() else "not recorded"
    framework = Path(CHROME).parents[1] / "Frameworks" / "Google Chrome Framework.framework" / "Versions"
    installed = sorted((d.name, datetime.datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M"))
                       for d in framework.iterdir() if d.is_dir() and not d.is_symlink() and d.name[0].isdigit())
    lines = [f"TensorFlow.js: {tfjs}",
             f"d3 (vendor/d3.min.js header): {d3_header}",
             f"Chrome reported by the recorded headless run (results/page-gpu.json, Browser.getVersion): {recorded}",
             f"Chrome installed now (--version): {chrome}",
             f"Chrome version folders and install times (local time): {installed}",
             f"Python: {platform.python_version()}",
             f"numpy: {numpy.__version__}"]
    text = "\n".join(lines) + "\n"
    (RESULTS / "versions.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

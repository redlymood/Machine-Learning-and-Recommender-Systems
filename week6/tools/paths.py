"""Where the HW6 tools read data and write results, in both layouts.

local      ML-English/hw6-tools/: data in ../week6, results in ./results,
           page exports in ./results/exports, HW5 factors in ./results
published  week6/tools/ of the GitHub repo: data in .., results in ../results,
           page exports in ../results, HW5 factors in ./ (next to the scripts)
The layout is detected by whether u.data sits next to the tools folder.
"""
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
PUBLISHED = (TOOLS.parent / "u.data").exists()

if PUBLISHED:
    WEEK6 = TOOLS.parent
    RESULTS = TOOLS.parent / "results"
    EXPORTS = RESULTS
    FACTORS = TOOLS / "hw5_mf_factors.json"
    WEEK5 = TOOLS.parent.parent / "week5"
else:
    WEEK6 = TOOLS.parent / "week6"
    RESULTS = TOOLS / "results"
    EXPORTS = RESULTS / "exports"
    FACTORS = RESULTS / "hw5_mf_factors.json"
    WEEK5 = TOOLS.parent / "week5"

"""HW6 experiments on the week6 page, webgl (part B), in the required order.

One headless Chrome session (GPU), driving the page's own controls and runners:
  B1 "Build graph" + "Run tests";
  B2 "Run alpha sweep (validation)"; selected alpha = best validation Recall@10,
     exact ties to the smaller alpha (the provided rule, script.js:1806);
  B3 slider := selected alpha, then the "Evaluate Recall@10 - all four methods"
     handler (hw6HandleRecall: four-method TEST comparison, which also re-runs
     the sweep);
  B4 Top-10 panel for users 1 and 98 x four methods (rows reach recommendations.csv);
  B5 one-factor PPR settings for users 1 / 98 through the Top-10 panel;
  B7 exclusion / ordering audit over both cohorts and all methods;
  B8 the page's three export buttons (downloads into results/exports/);
then the provided two-user demo (kept separately), plus data for part D
(test lists and per-user hits, PPR rankings of users 1 and 98, the
non-converged alpha 0.95 vectors). No code in week6/ is changed.

Usage: python3 experiments_page.py   (writes results/experiments-page.{json,txt},
results/page_lists.json, results/alpha095_page.json, results/exports/*)
"""
import csv
import io
import json
import time
from collections import Counter

from paths import EXPORTS, RESULTS
from week6_session import page_errors, week6_page

EXPORT_FILES = ["verification.json", "results.csv", "recommendations.csv"]
MY_USERS = [1, 98]
METHODS = ["popularity", "pagerank", "ppr", "mf"]
ONE_FACTOR = [("baseline", 1, 0.85), ("seed only", 98, 0.85), ("alpha only", 1, 0.50)]

SETUP_JS = r"""(() => {
  window.__hw6x = {
    relevant(split, u) {
      const name = split === "val" ? "val" : "test";
      const out = new Set();
      const grades = HW6_STATE.gradesBySplit[name].get(u);
      if (!grades) return out;
      const excluded = hw6ExclusionSet(name, u);
      grades.forEach((r, m) => { if (r >= 4 && HW6_STATE.candidateSet.has(m) && !excluded.has(m)) out.add(m); });
      return out;
    },
    setAlpha(a) {
      const slider = document.getElementById("alpha");
      slider.value = String(a);
      slider.dispatchEvent(new Event("input"));
      return hw6ReadAlpha();
    },
    setSelect(id, v) {
      const el = document.getElementById(id);
      el.value = String(v);
      el.dispatchEvent(new Event("change"));
      return el.value;
    },
  };
  return JSON.stringify({ backend: tf.getBackend(), version: tf.version.tfjs, tensors: tf.memory().numTensors,
                          alpha: hw6ReadAlpha() });
})()"""

BUILD_TESTS_JS = r"""(() => {
  document.getElementById("build-graph").click();
  document.getElementById("run-tests").click();
  const v = HW6_STATE.lastVerification;
  return JSON.stringify({ graphStatus: getGraph() ? getGraph().nodeCount : null, passed: v.passed, failed: v.failed,
    pending: v.pending, checks: v.checks.map((c) => ({ name: c.name, status: c.status, detail: c.detail })) });
})()"""

SWEEP_JS = r"""(() => {
  const t0 = performance.now();
  document.getElementById("alpha-sweep-btn").click();
  const seconds = (performance.now() - t0) / 1000;
  const rows = HW6_STATE.alphaSweepRows.map((r) => ({ ...r }));
  let best = null, bestRecall = -Infinity;
  for (const r of rows) if (r.error === null && r.recall > bestRecall) { bestRecall = r.recall; best = r.alpha; }
  const conv = [];
  getGraph().pprCache.forEach((e) => {
    let notConverged = 0;
    for (const c of e.converged) if (!c) notConverged++;
    conv.push({ alpha: e.alpha, notConverged, batchSeconds: e.seconds });
  });
  return JSON.stringify({ seconds, rows, best, bestRecall, conv, sliderAfter: hw6ReadAlpha(),
    status: document.getElementById("status").textContent,
    panel: document.getElementById("alpha-sweep-results").innerText.replace(/\s+/g, " ").trim() });
})()"""

COMPARE_JS = r"""(async () => {
  const t0 = performance.now();
  await hw6HandleRecall();
  const seconds = (performance.now() - t0) / 1000;
  return JSON.stringify({ seconds, sliderAfter: hw6ReadAlpha(), rows: HW6_STATE.comparisonRows,
    convergence: HW6_STATE.convergence, sweepRowsAfter: HW6_STATE.alphaSweepRows,
    mf: { ...MF_INFO, history: MF_INFO.history.length }, status: document.getElementById("status").textContent,
    panel: document.getElementById("comparison-results").innerText.replace(/\s+/g, " ").trim() });
})()"""

PANEL_JS = r"""(async (u, method, alpha) => {
  if (alpha !== null) __hw6x.setAlpha(alpha);
  __hw6x.setSelect("method", method);
  if (method === "mf") await hw6EnsureMfForNeed();
  const chosen = __hw6x.setSelect("user-select", u);
  await hw6RenderRecommendations();
  const a = hw6ReadAlpha();
  const alphaKey = method === "pagerank" || method === "ppr" ? a.toFixed(2) : "";
  const rows = HW6_STATE.recommendationRows
    .filter((r) => r.userId === u && r.method === method && r.alpha === alphaKey)
    .sort((x, y) => x.rank - y.rank);
  const rel = __hw6x.relevant("test", u);
  const panel = document.getElementById("recommendations");
  const direct = recommendTopK(method, u, 10, "test");
  return JSON.stringify({ user: u, method, alpha: a, userSelect: chosen,
    rows: rows.map((r) => ({ ...r, hit: rel.has(r.movieId) })),
    relevant: [...rel].sort((x, y) => x - y), domRows: panel.querySelectorAll("tbody tr").length,
    hints: Array.from(panel.querySelectorAll("p.hint")).map((p) => p.textContent), info: direct.info,
    panelMatchesDirect: direct.map((r) => r.movieId).join(",") === rows.map((r) => r.movieId).join(",") });
})(%s, %s, %s)"""

AUDIT_JS = r"""((split) => {
  const t0 = performance.now();
  const cohort = hw6SharedCohort(split);
  const out = { split, cohort, audit: [], lists: {}, perUser: {} };
  for (const method of ["popularity", "pagerank", "ppr", "mf"]) {
    let outside = 0, excludedHit = 0, short = 0, fewer = 0, zero = 0, unsorted = 0, lowestId = 0;
    const ids = {};
    for (const u of cohort) {
      const recs = recommendTopK(method, u, 10, split);
      const excluded = hw6ExclusionSet(split, u);
      for (let i = 0; i < recs.length; i++) {
        const r = recs[i];
        if (!HW6_STATE.candidateSet.has(r.movieId)) outside++;
        if (excluded.has(r.movieId)) excludedHit++;
        if (i > 0) {
          const p = recs[i - 1];
          if (p.score < r.score || (p.score === r.score && p.movieId >= r.movieId)) unsorted++;
        }
      }
      if (recs.length < 10) short++;
      if (recs.info.fewerThanK) fewer++;
      if (recs.info.zeroScoreIncluded) zero++;
      const lowest = HW6_STATE.candidateIds.filter((m) => !excluded.has(m)).slice(0, 10);
      if (lowest.join(",") === recs.map((r) => r.movieId).join(",")) lowestId++;
      ids[u] = recs.map((r) => r.movieId);
    }
    const rep = recallAtK(method, 10, split);
    out.audit.push({ method, lists: cohort.length, outside, excludedHit, short, fewer, zero, unsorted,
      equalsLowestEligibleIds: lowestId, recall: rep.recall, cohortSize: rep.cohort.length,
      sameCohort: rep.cohort.join(",") === cohort.join(","), excludedUnseen: rep.excludedUnseen,
      excludedNoEligible: rep.excludedNoEligible });
    out.lists[method] = ids;
    out.perUser[method] = rep.perUser;
  }
  out.alpha = hw6ReadAlpha();
  out.seconds = (performance.now() - t0) / 1000;
  return JSON.stringify(out);
})(%s)"""

HAND_JS = r"""(() => {
  const out = {};
  for (const u of [1, 98]) {
    const full = recommendTopK("ppr", u, 100000, "test");
    const rel = __hw6x.relevant("test", u);
    const rankOf = new Map(full.map((r, i) => [r.movieId, i + 1]));
    const scoreOf = new Map(full.map((r) => [r.movieId, r.score]));
    const top = full.slice(0, 10).map((r) => r.movieId);
    const hits = top.filter((m) => rel.has(m));
    out[u] = { alpha: hw6ReadAlpha(), eligibleCandidates: full.length, top10: top,
      relevant: [...rel].sort((a, b) => a - b).map((m) => ({ movieId: m, title: hw6Title(m),
        rank: rankOf.get(m), score: scoreOf.get(m), popularity: hw6PopularityOf(m) })),
      hits, recall: hits.length / rel.size, score10: full[9].score, score11: full[10].score };
  }
  return JSON.stringify(out);
})()"""

ALPHA095_JS = r"""(() => {
  const g = getGraph();
  let e = null;
  g.pprCache.forEach((x) => { if (x.alpha === 0.95 && x.tol === 1e-6 && x.maxIter === 500) e = x; });
  const users = [];
  for (let c = 0; c < e.columns; c++) {
    if (!e.converged[c]) users.push({ user: g.userIds[c], iterations: e.iterations[c], residual: e.residual[c],
      p: Array.from(e.values.subarray(c * e.n, (c + 1) * e.n)) });
  }
  return JSON.stringify({ alpha: 0.95, userIds: g.userIds, movieIds: g.movieIds, users });
})()"""

EXPORT_JS = r"""(() => {
  const strings = { verification: serializeVerification(), results: serializeResults(),
                    recommendations: serializeRecommendations() };
  for (const id of ["export-verification", "export-results", "export-recommendations"]) {
    document.getElementById(id).click();
  }
  return JSON.stringify(strings);
})()"""

DEMO_JS = r"""(() => {
  document.getElementById("two-user-demo-btn").click();
  const d = HW6_STATE.demo;
  return JSON.stringify({ error: d ? d.error : "no demo", users: d ? d.users : [], common: d ? d.common : [],
    settings: d ? d.settings.map((s) => ({ label: s.label, user: s.user, alpha: s.alpha,
      movieIds: s.recs.map((r) => r.movieId), meanPopularity: hw6MeanPopularity(s.recs) })) : [],
    panel: document.getElementById("two-user-demo-results").innerText.replace(/\s+/g, " ").trim().slice(0, 600) });
})()"""

END_JS = r"""JSON.stringify({ tensors: tf.memory().numTensors, backend: tf.getBackend(), alpha: hw6ReadAlpha(),
  recommendationRows: HW6_STATE.recommendationRows.length })"""


def wait_for_downloads(timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        names = {p.name for p in EXPORTS.iterdir()}
        if all(f in names for f in EXPORT_FILES) and not any(n.endswith(".crdownload") for n in names):
            return True
        time.sleep(0.3)
    return False


def main():
    EXPORTS.mkdir(parents=True, exist_ok=True)
    for name in EXPORT_FILES:  # only the three files this run downloads
        (EXPORTS / name).unlink(missing_ok=True)
    t_start = time.time()
    r = {}
    with week6_page(use_gpu=True, alarm=175, tag="experiments", download_dir=EXPORTS) as (chrome, session):
        ev = lambda js, timeout=120, aw=False: chrome.evaluate(js, session, timeout=timeout, await_promise=aw)
        r["setup"] = ev(SETUP_JS)
        r["b1"] = ev(BUILD_TESTS_JS)
        r["b2"] = ev(SWEEP_JS)
        selected = r["b2"]["best"]
        r["sliderSet"] = ev(f"__hw6x.setAlpha({selected})")
        r["b3"] = ev(COMPARE_JS, aw=True)
        r["b4"] = [ev(PANEL_JS % (u, json.dumps(m), "null"), aw=True) for u in MY_USERS for m in METHODS]
        r["b5"] = [dict(ev(PANEL_JS % (u, '"ppr"', a), aw=True), label=label) for label, u, a in ONE_FACTOR]
        r["sliderRestored"] = ev(f"__hw6x.setAlpha({selected})")
        r["b7"] = {split: ev(AUDIT_JS % json.dumps(split)) for split in ("test", "val")}
        r["hand"] = ev(HAND_JS)
        alpha095 = ev(ALPHA095_JS)
        r["exportStrings"] = ev(EXPORT_JS)
        r["downloadsComplete"] = wait_for_downloads()
        r["demo"] = ev(DEMO_JS)
        r["end"] = ev(END_JS)
        r["errors"] = page_errors(chrome, session)
    r["wallSeconds"] = round(time.time() - t_start, 1)

    # Part D inputs.
    (RESULTS / "alpha095_page.json").write_text(json.dumps(alpha095))
    lists = {"alpha": selected, "test": r["b7"]["test"], "val": {k: v for k, v in r["b7"]["val"].items() if k != "lists"},
             "hand": r["hand"]}
    (RESULTS / "page_lists.json").write_text(json.dumps(lists))

    # Downloaded files vs the exporter strings (generated_at differs by construction).
    checks = {}
    for name, key in zip(EXPORT_FILES, ("verification", "results", "recommendations")):
        path = EXPORTS / name
        got = path.read_bytes().decode("utf-8") if path.exists() else None  # keep the CSV CRLF line ends
        want = r["exportStrings"][key]
        if key == "verification" and got is not None:
            strip = lambda s: {k: v for k, v in json.loads(s).items() if k != "generated_at"}
            checks[name] = strip(got) == strip(want)
        else:
            checks[name] = got == want
    r["downloadMatchesExporter"] = checks
    settings = {
        "selected_alpha": selected,
        "selection": "validation Recall@10 over alpha in {0.50, 0.70, 0.85, 0.95}; exact ties go to the smaller "
                     "alpha (provided hw6RunAlphaSweep, strict > over the ascending grid, script.js:1793-1820)",
        "pagerank_alpha": selected,
        "pagerank_note": "global PageRank uses the selected PPR alpha as a disclosed control (readme 6.1); "
                         "it was not optimised independently",
        "sweep": [{"alpha": x["alpha"], "val_recall_at_10": x["recall"], "cohort_size": x["cohortSize"]}
                  for x in r["b2"]["rows"]],
        "ppr_not_converged_columns": r["b2"]["conv"],
        "mf_reference": {k: r["b3"]["mf"][k] for k in ("factors", "epochs", "learningRate", "regularization",
                                                       "batchSize", "seed", "backend", "checkpoint",
                                                       "validationRMSE", "trainedAt")},
        "mf_note": "provided TF.js reference; its checkpoint equals the mu-only baseline (collapsed, all scores tie)",
        "tol": 1e-6, "max_iter": 500, "K": 10, "backend": r["setup"]["backend"],
        "tfjs": r["setup"]["version"],
    }
    (EXPORTS / "selected_settings.json").write_text(json.dumps(settings, indent=2))
    (RESULTS / "experiments-page.json").write_text(json.dumps({k: v for k, v in r.items() if k != "b7"}, indent=1))
    lines = summarize(r, selected)
    (RESULTS / "experiments-page.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(r, selected):
    out = [f"backend {r['setup']['backend']} (tfjs {r['setup']['version']}); wall {r['wallSeconds']} s; "
           f"numTensors before {r['setup']['tensors']}, after {r['end']['tensors']}"]
    b1 = r["b1"]
    out.append(f"B1 self-checks: pass {b1['passed']}, fail {b1['failed']}, pending {b1['pending']}")
    out += [f"  #{i} {c['status']} {c['name']}" for i, c in enumerate(b1["checks"], 1) if c["status"] != "PASS"]
    b2 = r["b2"]
    out.append(f"B2 alpha sweep (validation, {b2['seconds']:.1f} s): selected alpha {selected}")
    conv = {c["alpha"]: c["notConverged"] for c in b2["conv"]}
    for x in b2["rows"]:
        out.append(f"  alpha {x['alpha']:.2f} | val Recall@10 {x['recall']!r} | cohort {x['cohortSize']} | "
                   f"PPR columns not converged {conv.get(x['alpha'])} | error {x['error']}")
    out.append(f"  slider after sweep {b2['sliderAfter']}; set to {r['sliderSet']}")
    b3 = r["b3"]
    out.append(f"B3 four-method TEST comparison ({b3['seconds']:.1f} s, slider {b3['sliderAfter']}):")
    out.append("  method | alpha | Recall@10 | cohort | excluded_unseen | excluded_no_eligible | mean pop | "
               "iterations | residual | converged | error")
    for x in b3["rows"]:
        out.append(f"  {x['method']} | {x['alpha']} | {x['recall']} | {x['cohortSize']} | {x['excludedUnseen']} | "
                   f"{x['excludedNoEligible']} | {x['meanPopularity']} | {x['iterations']} | {x['residual']} | "
                   f"{x['converged']} | {x['error']}")
    out.append(f"  convergence records: {b3['convergence']}")
    out.append(f"  MF: checkpoint {b3['mf']['checkpoint']}, val RMSE {b3['mf']['validationRMSE']}")
    res = list(csv.DictReader(io.StringIO(r["exportStrings"]["results"])))
    alphas = {(x["method"], x["split"]): x["alpha"] for x in res if x["split"] == "test" and x["recall_at_10"]}
    out.append(f"  results.csv test alpha column: {alphas}; pagerank/ppr == selected: "
               f"{alphas.get(('pagerank', 'test')) == f'{selected:.2f}' and alphas.get(('ppr', 'test')) == f'{selected:.2f}'}")

    out.append("B4 test Top-10 (Top-10 panel):")
    lists = {}
    for x in r["b4"]:
        ids = [row["movieId"] for row in x["rows"]]
        lists[(x["user"], x["method"])] = ids
        pops = [row["trainingPopularity"] for row in x["rows"]]
        out.append(f"  user {x['user']} | {x['method']} | alpha {x['alpha']:.2f} | {len(ids)} rows (DOM {x['domRows']}) | "
                   f"hits {sum(row['hit'] for row in x['rows'])}/{len(x['relevant'])} | mean pop {mean(pops):.1f} | "
                   f"panel == recommendTopK {x['panelMatchesDirect']} | info {x['info']}")
        for row in x["rows"]:
            out.append(f"     {row['rank']:>2} | {row['movieId']:>4} | {row['title']} | {row['genres']} | "
                       f"{row['score']:.6g} | pop {row['trainingPopularity']} | hit {row['hit']}")
    for m in METHODS:
        common = set(lists[(1, m)]) & set(lists[(98, m)])
        out.append(f"  common movies users 1 vs 98, {m}: {len(common)} {sorted(common)}")

    out.append("B5 one-factor PPR (users A = 1, B = 98):")
    base = r["b5"][0]
    base_ids = [row["movieId"] for row in base["rows"]]
    for x in r["b5"]:
        ids = [row["movieId"] for row in x["rows"]]
        genres = Counter(g for row in x["rows"] for g in row["genres"].split(";") if g)
        out.append(f"  {x['label']}: user {x['user']}, alpha {x['alpha']:.2f} | common with baseline "
                   f"{len(set(ids) & set(base_ids))} | mean pop {mean([row['trainingPopularity'] for row in x['rows']]):.1f} | "
                   f"top genres {genres.most_common(3)} | hits {sum(row['hit'] for row in x['rows'])} | ids {ids}")

    out.append("B7 exclusion audit (all cohort users):")
    out.append("  split | method | lists | outside pool | excluded rated | unsorted | <10 | fewerThanK | zero-score | "
               "== lowest eligible ids | Recall@10 | same cohort")
    for split in ("test", "val"):
        for a in r["b7"][split]["audit"]:
            out.append(f"  {split} | {a['method']} | {a['lists']} | {a['outside']} | {a['excludedHit']} | {a['unsorted']} | "
                       f"{a['short']} | {a['fewer']} | {a['zero']} | {a['equalsLowestEligibleIds']} | {a['recall']:.6f} | "
                       f"{a['sameCohort']}")
    out.append(f"  audit time test {r['b7']['test']['seconds']:.1f} s, val {r['b7']['val']['seconds']:.1f} s")

    rec = list(csv.DictReader(io.StringIO(r["exportStrings"]["recommendations"])))
    groups = Counter((x["user_id"], x["method"], x["alpha"]) for x in rec)
    out.append(f"B8 exports: downloads complete {r['downloadsComplete']}; download == exporter {r['downloadMatchesExporter']}")
    out.append(f"  recommendations.csv: {len(rec)} rows in {len(groups)} lists: {sorted(groups.items())}")
    out.append(f"  results.csv rows: {len(res)}; verification passed/failed/pending from export: "
               f"{[json.loads(r['exportStrings']['verification'])[k] for k in ('passed', 'failed', 'pending')]}")
    d = r["demo"]
    out.append(f"Provided two-user demo (separate; NOT my users): users {d['users']}, error {d['error']}")
    for s in d["settings"]:
        out.append(f"  {s['label']}: user {s['user']}, alpha {s['alpha']}, mean pop {s['meanPopularity']:.1f}, ids {s['movieIds']}")
    out.append(f"  common: {d['common']}")
    out.append(f"page errors: {r['errors']}")
    return out


if __name__ == "__main__":
    main()

"""TEST for the HW6 graph layer (week6/graph.js) in headless Chrome on webgl.

Drives the real page: clicks "Build graph" and "Run tests", then calls the
graph.js functions directly and records
  a) the 12 self-checks;
  b) the readme 4.1 fixture (both vectors, and p after the first update);
  c) the MovieLens graph counts and rowNormalize row sums;
  d) global PageRank at alpha in {0.50, 0.70, 0.85, 0.95};
  e) batched PPR for all users per alpha, cache size, and the seed guard
     p[seed node] >= 1 - alpha for users 1, 98, 943;
  f) batched vs one-user (opts.batched = false) runs;
  h) tf.memory().numTensors around repeated runs;
and exports the global PageRank and PPR (users 1, 98) vectors at alpha 0.85
for ppr_reference.py. Nothing in week6/ is modified; MF is not touched.

Usage: python3 graph_check.py   (writes results/graph-check-gpu.{json,txt},
results/page_vectors.json)
"""
import json
import time

from page_check import RESULTS
from week6_session import page_errors, week6_page

ALPHAS = [0.50, 0.70, 0.85, 0.95]
GUARD_USERS = [1, 98, 943]

ENV_JS = r"""JSON.stringify({ backend: tf.getBackend(), version: tf.version.tfjs,
  float32: tf.env().getBool("WEBGL_RENDER_FLOAT32_ENABLED"), webglVersion: tf.env().getNumber("WEBGL_VERSION"),
  tensors: tf.memory().numTensors })"""

BUILD_JS = r"""(() => {
  const t0 = performance.now();
  document.getElementById("build-graph").click();
  const ms = performance.now() - t0;
  const g = getGraph();
  return JSON.stringify({ ms, status: document.getElementById("status").textContent,
    nodeCount: g.nodeCount, nUsers: g.nUsers, nMovies: g.nMovies, positiveEdges: g.positiveEdges,
    zeroDegreeUsers: g.zeroDegreeUsers, zeroDegreeMovies: g.zeroDegreeMovies.length,
    paddingCheck: { firstUser: g.userIds[0], lastUser: g.userIds[g.nUsers - 1],
                    node943IsUser: g.isUserNode(943), movieAtNode943: g.movieOfNode(943) },
    summary: document.getElementById("graph-summary-body").innerText.replace(/\s+/g, " ").trim() });
})()"""

TESTS_JS = r"""(() => {
  document.getElementById("run-tests").click();
  const v = HW6_STATE.lastVerification;
  return JSON.stringify({ passed: v.passed, failed: v.failed, pending: v.pending,
    checks: v.checks.map((c) => ({ name: c.name, status: c.status, detail: c.detail })) });
})()"""

REFERENCE_JS = r"""(() => {
  const T = REFERENCE_GRAPH.transition;
  const pick = (r) => ({ p: Array.from(r.p), iterations: r.iterations, residual: r.residual, converged: r.converged });
  const errorOf = (fn) => { try { fn(); return null; } catch (e) { return String(e.message); } };
  return JSON.stringify({
    uniform: pick(pageRank(0.85, { transition: T })),
    restartA: pick(personalizedPageRank(0, 0.85, { transition: T })),
    firstUniform: pick(pageRank(0.85, { transition: T, maxIter: 1 })),
    firstRestartA: pick(personalizedPageRank(0, 0.85, { transition: T, maxIter: 1 })),
    readme: REFERENCE_GRAPH.reference,
    errors: {
      outOfRangeIndex: errorOf(() => personalizedPageRank(4, 0.85, { transition: T })),
      unknownUser: errorOf(() => personalizedPageRank(99999, 0.85)),
      badAlpha: errorOf(() => pageRank(1.5)),
    },
  });
})()"""

ROWNORM_JS = r"""(() => {
  const g = getGraph();
  const t0 = performance.now();
  const rn = rowNormalize(g);
  const ms = performance.now() - t0;
  const n = g.nodeCount;
  let maxErr = 0, positive = 0, dangling = 0, danglingNonZero = 0, maskMismatch = 0;
  for (let i = 0; i < n; i++) {
    let s = 0;
    const row = rn.Q[i];
    for (let j = 0; j < n; j++) s += row[j];
    if (rn.danglingMask[i]) { dangling++; if (s !== 0) danglingNonZero++; }
    else { positive++; maxErr = Math.max(maxErr, Math.abs(s - 1)); }
    if ((g.degree[i] === 0) !== !!rn.danglingMask[i]) maskMismatch++;
  }
  return JSON.stringify({ ms, qIsArray: Array.isArray(rn.Q), qRows: rn.Q.length, qCols: rn.Q[0].length,
    maskIsArray: Array.isArray(rn.danglingMask), positiveRows: positive, danglingRows: dangling,
    danglingRowsNonZero: danglingNonZero, maskVsDegreeMismatch: maskMismatch, maxRowSumError: maxErr });
})()"""

PAGERANK_JS = r"""((alpha) => {
  const before = tf.memory().numTensors;
  const t0 = performance.now();
  const r = pageRank(alpha);
  const ms = performance.now() - t0;
  let sum = 0, min = Infinity;
  for (const x of r.p) { sum += x; if (x < min) min = x; }
  return JSON.stringify({ alpha, iterations: r.iterations, residual: r.residual, converged: r.converged,
    sumError: Math.abs(sum - 1), minP: min, ms, n: r.p.length, tensorsDelta: tf.memory().numTensors - before });
})(%s)"""

PPR_JS = r"""((alpha, guardUsers) => {
  const g = getGraph();
  const before = tf.memory().numTensors;
  const t0 = performance.now();
  personalizedPageRank(1, alpha);
  const firstCallMs = performance.now() - t0;
  let entry = null;
  g.pprCache.forEach((e) => { if (e.alpha === alpha && e.tol === 1e-6 && e.maxIter === 500) entry = e; });
  const n = entry.n, B = entry.columns;
  const its = Array.from(entry.iterations).sort((a, b) => a - b);
  let notConverged = 0, maxSumError = 0, minP = Infinity, maxResidual = 0;
  for (let c = 0; c < B; c++) {
    if (!entry.converged[c]) notConverged++;
    maxResidual = Math.max(maxResidual, entry.residual[c]);
    let s = 0;
    for (let j = c * n; j < (c + 1) * n; j++) { const v = entry.values[j]; s += v; if (v < minP) minP = v; }
    maxSumError = Math.max(maxSumError, Math.abs(s - 1));
  }
  const positiveIts = [];
  const stuck = [];
  const degrees = [];
  for (let c = 0; c < B; c++) {
    degrees.push(g.degree[c]);
    if (g.degree[c] > 0) positiveIts.push(entry.iterations[c]);
    if (!entry.converged[c]) stuck.push({ user: g.userIds[c], degree: g.degree[c], residual: entry.residual[c] });
  }
  positiveIts.sort((a, b) => a - b);
  degrees.sort((a, b) => a - b);
  stuck.sort((a, b) => b.residual - a.residual);
  const t1 = performance.now();
  personalizedPageRank(98, alpha);
  const lookupMs = performance.now() - t1;
  const guard = guardUsers.map((u) => {
    const r = personalizedPageRank(u, alpha);
    const node = g.nodeOfUser(u);
    return { user: u, node, pSeed: r.p[node], bound: 1 - alpha, margin: r.p[node] - (1 - alpha),
             holds: r.p[node] >= 1 - alpha, iterations: r.iterations, converged: r.converged,
             pAtNodeIndexEqualToId: r.p[u], nodeIndexEqualToIdIs: g.isUserNode(u) ? "user " + g.userIds[u] : "movie " + g.movieOfNode(u) };
  });
  const zeroUser = g.zeroDegreeUsers.map((u) => { const r = personalizedPageRank(u, alpha);
    return { user: u, iterations: r.iterations, pSeed: r.p[g.nodeOfUser(u)] }; });
  return JSON.stringify({ alpha, columns: B, n, batchSeconds: entry.seconds, firstCallMs, lookupMs,
    itMin: its[0], itMedian: its[Math.floor(B / 2)], itMax: its[B - 1], notConverged, maxResidual,
    maxSumError, minP, megabytes: entry.megabytes, guard, itMinPositive: positiveIts[0],
    userDegreeMedian: degrees[Math.floor(B / 2)], stuck, zeroUser, tensorsDelta: tf.memory().numTensors - before });
})(%s, %s)"""

SINGLE_JS = r"""((u, alpha) => {
  const b = personalizedPageRank(u, alpha);
  const t0 = performance.now();
  const s = personalizedPageRank(u, alpha, { batched: false });
  const ms = performance.now() - t0;
  let maxDelta = 0;
  for (let j = 0; j < b.p.length; j++) maxDelta = Math.max(maxDelta, Math.abs(b.p[j] - s.p[j]));
  return JSON.stringify({ user: u, alpha, itBatched: b.iterations, itSingle: s.iterations,
    resBatched: b.residual, resSingle: s.residual, convBatched: b.converged, convSingle: s.converged,
    maxDelta, singleMs: ms });
})(%s, %s)"""

EXPORT_JS = r"""(() => {
  const g = getGraph();
  const pack = (r) => ({ p: Array.from(r.p), iterations: r.iterations, residual: r.residual, converged: r.converged });
  return JSON.stringify({ alpha: 0.85, userIds: g.userIds, movieIds: g.movieIds, backend: tf.getBackend(),
    pagerank: pack(pageRank(0.85)),
    ppr: { "1": pack(personalizedPageRank(1, 0.85)), "98": pack(personalizedPageRank(98, 0.85)) } });
})()"""

LEAK_JS = r"""(() => {
  const g = getGraph();
  const cacheBefore = graphPprCacheInfo(g);
  const before = tf.memory().numTensors;
  const afterEach = [];
  for (let k = 0; k < 2; k++) {
    pageRank(0.85);
    g.pprCache.forEach((e, key) => { if (e.alpha === 0.85) g.pprCache.delete(key); });
    personalizedPageRank(1, 0.85);
    afterEach.push(tf.memory().numTensors);
  }
  return JSON.stringify({ before, afterEach, after: tf.memory().numTensors, cacheBefore,
    cacheAfter: graphPprCacheInfo(g) });
})()"""


def main():
    t_start = time.time()
    report = {}
    with week6_page(use_gpu=True, alarm=150, tag="graph-check") as (chrome, session):
        ev = lambda js, timeout=60: chrome.evaluate(js, session, timeout=timeout)
        report["env"] = ev(ENV_JS)
        report["build"] = ev(BUILD_JS)
        report["selfChecks"] = ev(TESTS_JS)
        report["reference"] = ev(REFERENCE_JS)
        report["rowNormalize"] = ev(ROWNORM_JS)
        report["pagerank"] = [ev(PAGERANK_JS % a) for a in ALPHAS]
        report["ppr"] = [ev(PPR_JS % (a, json.dumps(GUARD_USERS)), timeout=90) for a in ALPHAS]
        report["single"] = [ev(SINGLE_JS % (u, a)) for a in (0.85, 0.95) for u in GUARD_USERS]
        vectors = ev(EXPORT_JS)
        report["leak"] = ev(LEAK_JS, timeout=90)
        report["errors"] = page_errors(chrome, session)
    report["wallSeconds"] = round(time.time() - t_start, 1)
    (RESULTS / "page_vectors.json").write_text(json.dumps(vectors))
    (RESULTS / "graph-check-gpu.json").write_text(json.dumps(report, indent=1))

    lines = summarize(report)
    (RESULTS / "graph-check-gpu.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def summarize(r):
    out = []
    env, b = r["env"], r["build"]
    out.append(f"env: {env} | wall {r['wallSeconds']} s")
    out.append(f"build: {b['ms']:.0f} ms | {b['status']}")
    out.append(f"  nodes {b['nodeCount']} = users {b['nUsers']} + movies {b['nMovies']} | positive edges "
               f"{b['positiveEdges']} | zero-degree users {b['zeroDegreeUsers']} | zero-degree movies "
               f"{b['zeroDegreeMovies']} | {b['paddingCheck']}")
    out.append(f"  summary panel: {b['summary']}")
    s = r["selfChecks"]
    out.append(f"self-checks: pass {s['passed']}, fail {s['failed']}, pending {s['pending']}")
    for i, c in enumerate(s["checks"], 1):
        out.append(f"  #{i:<2} {c['status']:<8}{c['name']}  |  {c['detail'][:120]}")
    ref = r["reference"]
    out.append("reference graph (alpha 0.85):")
    for key, readme_key in (("uniform", "uniform"), ("restartA", "restartAtA")):
        got, want = ref[key], ref["readme"][readme_key]["p"]
        out.append(f"  {key}: iterations {got['iterations']}, residual {got['residual']:.3e}, "
                   f"converged {got['converged']}")
        for label, x, y in zip("ABCD", got["p"], want):
            out.append(f"    {label}  page {x:.10f}  readme {y:.10f}  |diff| {abs(x - y):.2e}")
    for key in ("firstUniform", "firstRestartA"):
        f = ref[key]
        out.append(f"  {key} (p after the FIRST update): " + ", ".join(f"{x:.10f}" for x in f["p"]) +
                   f"  | residual {f['residual']:.10f}, converged {f['converged']}")
    out.append(f"  error paths: {ref['errors']}")
    out.append(f"rowNormalize(MovieLens): {r['rowNormalize']}")
    out.append("global PageRank: alpha | iterations | residual | converged | |sum-1| | min p | ms | tensorsDelta")
    for p in r["pagerank"]:
        out.append(f"  {p['alpha']:.2f} | {p['iterations']} | {p['residual']:.3e} | {p['converged']} | "
                   f"{p['sumError']:.2e} | {p['minP']:.3e} | {p['ms']:.0f} | {p['tensorsDelta']}")
    out.append("batched PPR: alpha | it min/median/max | not converged | max residual | max |sum-1| | min p | "
               "batch s | first-call ms | lookup ms | MB | tensorsDelta")
    for p in r["ppr"]:
        out.append(f"  {p['alpha']:.2f} | {p['itMin']}/{p['itMedian']}/{p['itMax']} | {p['notConverged']} | "
                   f"{p['maxResidual']:.3e} | {p['maxSumError']:.2e} | {p['minP']:.3e} | {p['batchSeconds']:.2f} | "
                   f"{p['firstCallMs']:.0f} | {p['lookupMs']:.1f} | {p['megabytes']:.2f} | {p['tensorsDelta']}")
        for g in p["guard"]:
            out.append(f"    guard user {g['user']} node {g['node']}: p_seed {g['pSeed']:.6f} >= {g['bound']:.2f} "
                       f"-> {g['holds']} (margin {g['margin']:.6f}); node index {g['user']} is "
                       f"{g['nodeIndexEqualToIdIs']} with p {g['pAtNodeIndexEqualToId']:.6f}")
        out.append(f"    zero-degree users: {p['zeroUser']}; min iterations over positive-degree users "
                   f"{p['itMinPositive']}")
        if p["stuck"]:
            st = p["stuck"]
            degs = sorted(x["degree"] for x in st)
            out.append(f"    not converged: {len(st)} users, residual {st[-1]['residual']:.3e}..{st[0]['residual']:.3e}, "
                       f"degree {degs[0]:.0f}..{degs[-1]:.0f} (median {degs[len(degs) // 2]:.0f}; all users median "
                       f"{p['userDegreeMedian']:.0f}); worst {[(x['user'], int(x['degree'])) for x in st[:5]]}")
    out.append("batched vs one-user: user | alpha | it batched/single | residual batched/single | max |delta p| | single ms")
    for x in r["single"]:
        out.append(f"  {x['user']} | {x['alpha']:.2f} | {x['itBatched']}/{x['itSingle']} | "
                   f"{x['resBatched']:.3e}/{x['resSingle']:.3e} | {x['maxDelta']:.3e} | {x['singleMs']:.0f}")
    lk = r["leak"]
    out.append(f"leak: numTensors before {lk['before']}, after each round {lk['afterEach']}, after {lk['after']}; "
               f"cache before {lk['cacheBefore']['entries']} entries {lk['cacheBefore']['megabytes']:.2f} MB")
    out.append(f"page errors: {r['errors']}")
    return out


if __name__ == "__main__":
    main()

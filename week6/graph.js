/**
 * week6/graph.js - HW6 graph / Personalized PageRank layer.
 *
 * This is a plain (classic) script, NOT an ES module. `index.html` loads the
 * TensorFlow.js runtime from a CDN, then the vendored d3 v7 build, then
 * `data.js`, then this file, then `script.js`:
 *
 *   <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
 *   <script src="vendor/d3.min.js"></script>
 *   <script src="data.js"></script>
 *   <script src="graph.js"></script>
 *   <script src="script.js"></script>
 *
 * `week6/data.js` (loaded before this file) exposes the MovieLens 100k data
 * layer as globals: `movies`, `ratings`, `numUsers`, `numMovies`,
 * `ratingMatrix`, `genreNames`, `SPLIT_EXCLUSIONS` and `splitByTimestamp()`.
 * This file does not mutate them.
 *
 * WHAT YOU MUST IMPLEMENT (`TODO(hw6)` - each stub throws until you write it):
 *   1. `buildGraph(train)`            - bipartite graph over training positives.
 *   2. `rowNormalize(graph)`          - row-stochastic transition Q + dangling mask.
 *   3. `pageRank(alpha, opts)`        - uniform-restart PageRank.
 *   4. `personalizedPageRank(userId, alpha, opts)` - one-hot restart at one user.
 *
 * PROVIDED for you (scaffolding): the `REFERENCE_GRAPH` constant with the two
 * reference vectors and both rankings from readme.md section 4.1 (plus the
 * derived `REFERENCE_GRAPH.transition`, see below), the graph
 * accessors (`GRAPH`, `setGraph`, `getGraph`), the d3 graph view
 * (`renderGraphView`), the 4-node fixture teaching demo (`fixtureStep` /
 * `fixtureReset`), and the `#graph-summary-body` renderer.
 *
 * GRACEFUL DEGRADATION: the fixture demo and the graph view work before any
 * stub is implemented, so the page loads and renders with no console errors.
 *
 * SCOPE NOTE: d3 is used ONLY by the graph view. The propagation stubs must use
 * TensorFlow.js tensor operations on `R` and its transpose for the MovieLens
 * graph (readme.md section 3.5); the fixture demo below is the only
 * plain-JavaScript propagation loop in this file.
 *
 * @module week6/graph
 */

// ---------------------------------------------------------------------------
// Typedefs (documentation only - no runtime effect)
// ---------------------------------------------------------------------------

/**
 * A single rating record as produced by `data.js` / `splitByTimestamp()`.
 *
 * @typedef {Object} Rating
 * @property {number} userId
 * @property {number} movieId
 * @property {number} rating    1..5 (MovieLens scale).
 * @property {number} timestamp Unix seconds.
 */

/**
 * The bipartite user-movie graph returned by `buildGraph`.
 *
 * @typedef {Object} Graph
 * @property {number} nUsers        Number of distinct user ids observed in TRAIN.
 * @property {number} nMovies       Number of distinct movie ids observed in TRAIN.
 * @property {number[]} userIds     User ids, ascending; index = local user index.
 * @property {number[]} movieIds    Movie ids, ascending; index = local movie index.
 * @property {Map<number,number>} userIndexOf  userId -> local user index [0, nUsers).
 * @property {Map<number,number>} movieIndexOf movieId -> local movie index [0, nMovies).
 * @property {number} nodeCount     nUsers + nMovies (no padding node).
 * @property {function(number):number} nodeOfUser    userId -> global node index.
 * @property {function(number):number} movieOfNode   global node index -> movieId.
 * @property {function(number):boolean} isUserNode   global node index -> user?
 * @property {Float64Array} degree  Bipartite degree over node index.
 * @property {number[]} zeroDegreeUsers  User ids with degree 0.
 * @property {number[]} zeroDegreeMovies Movie ids with degree 0.
 * @property {number} positiveEdges Number of positive training edges (1 entries in R).
 * @property {number[][]} R         R[u][i] = 1 iff user u rated movie i >= 4.0
 *   in TRAINING, else 0. Shape [nUsers][nMovies].
 */

/**
 * The row-normalized transition returned by `rowNormalize`.
 *
 * @typedef {Object} RowNormalized
 * @property {number[][]} Q             Row-stochastic transition; positive-degree
 *   rows sum to 1, zero-degree rows are all zero.
 * @property {boolean[]} danglingMask   true for zero-degree node indices.
 */

/**
 * A PageRank / PPR result.
 *
 * @typedef {Object} PageRankResult
 * @property {Float64Array} p     Node probabilities, node order, sums to ~1.
 * @property {number} iterations  Number of updates performed.
 * @property {number} residual    Final L1 difference between consecutive distributions.
 * @property {boolean} converged  true when residual < tol before the cap.
 */

/**
 * An explicit transition supplied to `pageRank` / `personalizedPageRank`
 * through `opts.transition`.
 *
 * Use this to propagate on a caller-provided graph instead of the global
 * `GRAPH` (the HW6 harness uses it to run the implementation on the 4-node
 * reference fixture, which has no MovieLens graph). When present it replaces
 * the global-graph lookup for that call.
 *
 * @typedef {Object} TransitionSpec
 * @property {number[][]} Q          Row-normalized transition; `Q[i][j]` is the
 *   probability of moving from node `i` to node `j`. Square (n x n). A
 *   positive-degree row sums to 1; a zero-degree row is all zero. Already
 *   normalized, so the implementation must NOT re-normalize it.
 * @property {Uint8Array} danglingMask Nonzero (1) exactly at the zero-degree
 *   node indices, indexed by position in the same node order as `Q`.
 * @property {string[]} [nodeLabels] Optional labels for display and tests only;
 *   the propagation never reads them. `REFERENCE_GRAPH.transition.nodeLabels`
 *   is `['A', 'B', 'C', 'D']`.
 */

/**
 * Options for `renderGraphView`.
 *
 * @typedef {Object} GraphViewOptions
 * @property {("fixture"|"ego")} [mode]   Which graph to draw.
 * @property {number|null} [seedUserId]   Seed user for the "ego" subgraph.
 * @property {("degree"|"score")} [colorBy] Node fill encoding.
 * @property {number} [maxNodes]          Cap on drawn nodes (default 120).
 * @property {function(number):number} [getScore] Node index -> score, used when
 *   colorBy === "score".
 */

// ---------------------------------------------------------------------------
// Reference graph (provided, read-only by convention)
//
// Exactly the readme.md section 4.1 fixture: node order A, B, C, D; directed
// edges A -> B, A -> C, B -> C, C -> A, D -> C; no dangling nodes. The two
// reference vectors and both rankings are reproduced verbatim from readme.md
// section 4.1.
// ---------------------------------------------------------------------------

/**
 * The 4-node reference graph from readme.md section 4.1, plus its reference
 * results. `nodes` assigns A, B to the left (user) column and C, D to the right
 * (movie) column so the fixture can reuse the same deterministic two-column
 * layout as the data view. Treat this object as read-only.
 */
const REFERENCE_GRAPH = {
  nodeOrder: ["A", "B", "C", "D"],
  nodes: [
    { index: 0, label: "A", kind: "user" },
    { index: 1, label: "B", kind: "user" },
    { index: 2, label: "C", kind: "movie" },
    { index: 3, label: "D", kind: "movie" }
  ],
  // [from, to] pairs in node index space (A = 0, B = 1, C = 2, D = 3).
  edges: [
    [0, 1], // A -> B
    [0, 2], // A -> C
    [1, 2], // B -> C
    [2, 0], // C -> A
    [3, 2]  // D -> C
  ],
  edgeLabels: ["A -> B", "A -> C", "B -> C", "C -> A", "D -> C"],
  hasDanglingNodes: false,
  reference: {
    // alpha = 0.85, uniform restart over all four nodes.
    uniform: {
      alpha: 0.85,
      restart: "uniform",
      p: [0.3725268513, 0.1958239118, 0.3941492369, 0.0375000000],
      ranking: [2, 0, 1, 3],
      rankingLabel: "C > A > B > D"
    },
    // alpha = 0.85, restart only at A.
    restartAtA: {
      alpha: 0.85,
      restart: "A",
      restartIndex: 0,
      p: [0.4522328999, 0.1921989825, 0.3555681176, 0.0000000000],
      ranking: [0, 2, 1, 3],
      rankingLabel: "A > C > B > D"
    }
  }
};

// PROVIDED: the fixture's row-normalized transition, DERIVED from
// `REFERENCE_GRAPH.edges` (never a hand-typed matrix). `Q` is what
// `pageRank` / `personalizedPageRank` must consume when the harness passes
// `opts.transition`; `danglingMask` is a Uint8Array, 1 at zero-degree node
// indices (all zeros here, since A/B/C/D all have an outgoing link).
//
// The load-time self-test below is deliberately NON-FATAL: it records state in
// `REFERENCE_GRAPH.transitionSelfTest` instead of throwing, so a malformed
// fixture can never break page load. It adds no global name.
(function () {
  const fixture = graphFixtureTransition(); // derives P from `edges`
  const n = REFERENCE_GRAPH.nodeOrder.length;
  const Q = fixture.P.map(function (row) { return row.slice(); });
  const danglingMask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (fixture.out[i] === 0) danglingMask[i] = 1;
  }

  REFERENCE_GRAPH.transition = {
    Q: Q,
    danglingMask: danglingMask,
    nodeLabels: REFERENCE_GRAPH.nodeOrder.slice()
  };

  let maxRowSumError = 0;
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) sum += Q[i][j];
    maxRowSumError = Math.max(maxRowSumError, Math.abs(sum - 1));
  }
  let danglingCount = 0;
  for (let i = 0; i < n; i++) {
    if (danglingMask[i]) danglingCount += 1;
  }
  REFERENCE_GRAPH.transitionSelfTest = {
    rows: n,
    cols: Q.length ? Q[0].length : 0,
    maxRowSumError: maxRowSumError,
    rowsSumToOne: maxRowSumError < 1e-12,
    danglingCount: danglingCount,
    danglingMaskAllZero: danglingCount === 0
  };
})();

// ---------------------------------------------------------------------------
// Graph-view theme (private)
//
// Canvas is #FFFFFF; the categorical pair is Okabe-Ito blue/orange (red-green
// colour-blind safe), and scores use a single-hue sequential blue ramp.
// ---------------------------------------------------------------------------

/**
 * Fixed visual parameters for the graph view. Private to graph.js.
 */
const graphTheme = {
  width: 1000,
  height: 420,
  margin: { top: 34, right: 110, bottom: 26, left: 110 },
  userColor: "#0072B2",
  movieColor: "#E69F00",
  edgeColor: "#b9c0ca",
  nodeStroke: "#ffffff",
  zeroStroke: "#1c2128",
  labelColor: "#1c2128",
  noteColor: "#5b6472",
  minRadius: 6,
  maxRadius: 22,
  rampLight: [232, 243, 252],
  rampDark: [8, 48, 96],
  tooltipId: "graph-tooltip"
};

// ---------------------------------------------------------------------------
// Module state (public: GRAPH_VIEW_STATE)
// ---------------------------------------------------------------------------

/**
 * Small module-level state for the graph view. Mutated only by the rendering
 * and fixture-demo code in this file.
 */
const GRAPH_VIEW_STATE = {
  mode: "fixture",
  seedUserId: null,
  colorBy: "degree",
  maxNodes: 120,
  lastRender: null,
  fixture: {
    iteration: 0,
    p: null,
    residual: null,
    restart: "uniform",
    alpha: 0.85
  }
};

// ---------------------------------------------------------------------------
// Graph accessors (public)
// ---------------------------------------------------------------------------

/**
 * The current graph, or null before `buildGraph` has been wired in.
 *
 * @type {Graph|null}
 */
let GRAPH = null;

/**
 * Replace the current graph and refresh the dependent UI.
 *
 * `script.js` should call this (rather than assigning `GRAPH` directly) so the
 * graph summary and the graph view re-render whenever the graph changes. A
 * `null` argument clears the graph and restores the empty states.
 *
 * @param {Graph|null} g
 * @returns {Graph|null} the stored graph.
 */
function setGraph(g) {
  GRAPH = g || null;
  try {
    graphRenderSummary(GRAPH);
  } catch (error) {
    graphSetLegendMessage("Graph summary failed: " + graphErrorMessage(error));
  }
  try {
    renderGraphView({ mode: GRAPH_VIEW_STATE.mode });
  } catch (error) {
    graphSetLegendMessage("Graph view failed: " + graphErrorMessage(error));
  }
  return GRAPH;
}

/**
 * Read the current graph without changing any state.
 *
 * @returns {Graph|null}
 */
function getGraph() {
  return GRAPH;
}

// ---------------------------------------------------------------------------
// Student stubs (TODO(hw6))
//
// Each stub throws until you implement it. Keep the signature and the return
// shape; update the JSDoc contract notes if your design differs. The graph view
// and the fixture demo keep working while these throw.
// ---------------------------------------------------------------------------

/**
 * Build the bipartite user-movie graph from the TRAINING split.
 *
 * Rules (readme.md section 3.1):
 *  - The node universe is the set of user ids and movie ids observed in
 *    `train` (any rating value). Ids are mapped explicitly to contiguous
 *    indices in ascending id order; no padding node is added.
 *  - Node order is users first, then movies:
 *      user u  -> global node userIndexOf.get(u) in [0, nUsers)
 *      movie m -> global node nUsers + movieIndexOf.get(m) in [nUsers, nUsers + nMovies)
 *    The graph view and the propagation rely on this exact order.
 *  - R[u][i] = 1 iff user u rated movie i with rating >= 4.0 in TRAINING, else
 *    0. A zero is "no positive edge", not a negative opinion. A repeated
 *    (user, movie) record is counted once.
 *  - degree is the bipartite degree (number of positive edges) over BOTH node
 *    kinds. Users and movies observed in training without a positive edge stay
 *    in the graph as zero-degree nodes and are listed in `zeroDegreeUsers` /
 *    `zeroDegreeMovies`.
 *
 * Besides the typedef fields the graph carries `pprCache`, a Map that
 * `personalizedPageRank` fills with batched results keyed by alpha, tol and
 * maxIter. The cache lives on this object, so a rebuilt graph never reuses a
 * stale walk.
 *
 * Plain JavaScript on purpose: this is parsing and id mapping (readme.md
 * section 3.5). The propagation runs on TensorFlow.js tensors.
 *
 * Failure cases: a non-array `train` is treated as empty and yields an empty
 * graph (0 nodes) instead of throwing; records whose userId or movieId is not a
 * finite number are skipped.
 *
 * @param {Rating[]} train
 * @returns {Graph} the typedef fields plus `pprCache: Map<string, Object>`.
 */
function buildGraph(train) {
  const rows = Array.isArray(train) ? train : [];
  const userSet = new Set();
  const movieSet = new Set();
  for (const row of rows) {
    if (!row) continue;
    const userId = Number(row.userId);
    const movieId = Number(row.movieId);
    if (!Number.isFinite(userId) || !Number.isFinite(movieId)) continue;
    userSet.add(userId);
    movieSet.add(movieId);
  }

  const userIds = Array.from(userSet).sort((a, b) => a - b);
  const movieIds = Array.from(movieSet).sort((a, b) => a - b);
  const nUsers = userIds.length;
  const nMovies = movieIds.length;
  const nodeCount = nUsers + nMovies;
  const userIndexOf = new Map();
  userIds.forEach((id, u) => userIndexOf.set(id, u));
  const movieIndexOf = new Map();
  movieIds.forEach((id, i) => movieIndexOf.set(id, i));

  const R = [];
  for (let u = 0; u < nUsers; u++) R.push(new Array(nMovies).fill(0));
  const degree = new Float64Array(nodeCount);
  let positiveEdges = 0;
  for (const row of rows) {
    if (!row || !(Number(row.rating) >= 4.0)) continue;
    const u = userIndexOf.get(Number(row.userId));
    const i = movieIndexOf.get(Number(row.movieId));
    if (u === undefined || i === undefined || R[u][i] === 1) continue;
    R[u][i] = 1;
    degree[u] += 1;
    degree[nUsers + i] += 1;
    positiveEdges += 1;
  }

  return {
    nUsers: nUsers,
    nMovies: nMovies,
    userIds: userIds,
    movieIds: movieIds,
    userIndexOf: userIndexOf,
    movieIndexOf: movieIndexOf,
    nodeCount: nodeCount,
    nodeOfUser: function (userId) {
      const u = userIndexOf.get(Number(userId));
      return u === undefined ? -1 : u;
    },
    movieOfNode: function (node) {
      const i = Number(node) - nUsers;
      return Number.isInteger(i) && i >= 0 && i < nMovies ? movieIds[i] : -1;
    },
    isUserNode: function (node) {
      const k = Number(node);
      return k >= 0 && k < nUsers;
    },
    degree: degree,
    zeroDegreeUsers: userIds.filter((id, u) => degree[u] === 0),
    zeroDegreeMovies: movieIds.filter((id, i) => degree[nUsers + i] === 0),
    positiveEdges: positiveEdges,
    R: R,
    pprCache: new Map()
  };
}

/**
 * Row-normalize the transition matrix and record the zero-degree nodes.
 *
 * Returns the dense square transition for inspection and for the self-checks
 * (script.js:2460 and :2951 require a square `Q` and an Array mask). `Q[i][j]`
 * is the probability of moving from node i to node j, in the users-then-movies
 * order of `buildGraph`:
 *   user u  -> movie i : R[u][i] / deg(user u)
 *   movie i -> user u  : R[u][i] / deg(movie i)
 * A positive-degree row sums to 1. A zero-degree row stays all zero and is
 * `true` in `danglingMask` (a plain Array of booleans). Degrees are recomputed
 * from R, so Q and the mask always agree with the matrix the propagation uses.
 *
 * The dangling mass is handled in the update, not here (readme.md section 3.3):
 *
 *   p_next = alpha * (Q^T * p + d * v) + (1 - alpha) * v
 *
 * where d is the probability mass currently on zero-degree nodes.
 *
 * Size: Q has nodeCount^2 plain numbers (about 6.5 million for MovieLens 100K),
 * so this helper is meant for tests and small graphs. `pageRank` and
 * `personalizedPageRank` never call it and never build the square matrix as a
 * tensor: they multiply by the two off-diagonal blocks, R / deg_user and
 * R^T / deg_movie (R^T through `transposeA`), using the degree vector
 * (readme.md section 3.5).
 *
 * Failure cases: a null/missing graph (or one without R) returns
 * { Q: [], danglingMask: [] }; an R whose shape disagrees with nUsers /
 * nMovies throws.
 *
 * @param {Graph} graph
 * @returns {RowNormalized}
 */
function rowNormalize(graph) {
  if (!graph || !Array.isArray(graph.R)) return { Q: [], danglingMask: [] };
  const shape = graphBlockShape(graph, "rowNormalize");
  const nUsers = shape.nUsers;
  const n = shape.n;
  const degree = new Float64Array(n);
  for (let u = 0; u < nUsers; u++) {
    for (let i = 0; i < shape.nMovies; i++) {
      const w = Number(graph.R[u][i]) || 0;
      degree[u] += w;
      degree[nUsers + i] += w;
    }
  }

  const Q = [];
  for (let k = 0; k < n; k++) Q.push(new Array(n).fill(0));
  for (let u = 0; u < nUsers; u++) {
    for (let i = 0; i < shape.nMovies; i++) {
      const w = Number(graph.R[u][i]) || 0;
      if (w === 0) continue;
      Q[u][nUsers + i] = w / degree[u];
      Q[nUsers + i][u] = w / degree[nUsers + i];
    }
  }
  return { Q: Q, danglingMask: Array.from(degree, (d) => d === 0) };
}

/**
 * Global PageRank with a uniform restart distribution, on TensorFlow.js.
 *
 * Power iteration (readme.md sections 3.2-3.4) with the zero-degree policy of
 * section 3.3, shared with `personalizedPageRank`:
 *
 *   p_0 = v = 1/n ;  p_next = alpha * (Q^T * p + d * v) + (1 - alpha) * v
 *
 * d is the mass currently on zero-degree nodes. Q^T * p carries exactly the
 * mass of the positive-degree nodes, so the update keeps sum(p) = 1 by itself;
 * p is never renormalized. The run stops after the first update whose L1
 * change sum_j |p_next[j] - p[j]| is below `tol`, or after `maxIter` updates;
 * hitting the cap first gives `converged = false`. No node's score is forced to
 * the restart probability.
 *
 * TRANSITION SOURCE
 *  - `opts.transition` present (a {@link TransitionSpec}, e.g.
 *    `REFERENCE_GRAPH.transition`): propagate on that caller-supplied square Q
 *    as given (no re-normalization). `danglingMask` is read by position and a
 *    marked row must be all zero. The global `GRAPH` is not needed.
 *  - `opts.transition` absent: use the global `GRAPH` from `buildGraph`. This
 *    MovieLens path never builds the square matrix. It computes Q^T * p as two
 *    block products: R x (p_movies / deg_movie) for the user nodes and
 *    R^T x (p_users / deg_user) for the movie nodes (R^T through
 *    `transposeA`), with 1 / deg = 0 on zero-degree nodes.
 *  - Neither present: throws; never falls back to a uniform guess.
 *
 * NUMERICS AND I/O: TensorFlow.js computes in float32; the result is widened to
 * a Float64Array. Each iteration runs inside `tf.tidy` and reads back only the
 * L1 residual with `dataSync()`; the vector is read once at the end. The
 * read-back is synchronous because the provided adapter rejects a Promise
 * (script.js:1333-1343), unlike readme.md 3.5 ("asynchronously where
 * practical"). Every tensor is disposed before returning.
 *
 * NODE-INDEX CONVENTION: p[i] is the probability of global node i (users
 * [0, nUsers), then movies [nUsers, nUsers + nMovies); no padding node), or of
 * Q's own row order for `opts.transition`.
 *
 * @param {number} alpha Link-following probability in [0, 1]; 1 - alpha restarts.
 * @param {{tol?: number, maxIter?: number, transition?: TransitionSpec}} [opts]
 *   Defaults tol = 1e-6, maxIter = 500, transition absent (use the global GRAPH).
 * @returns {PageRankResult} `{p, iterations, residual, converged}`: `p` sums to
 *   ~1 (float32 precision); `iterations` is the number of updates performed;
 *   `residual` is the L1 change of the last update; `converged` is true only
 *   when `residual < tol` was reached within `maxIter` updates.
 * @throws {Error} on an invalid alpha / tol / maxIter / transition, when neither
 *   `opts.transition` nor the global `GRAPH` exists, or for an empty graph.
 */
function pageRank(alpha, opts) {
  const o = graphPropagationOptions("pageRank", alpha, opts);
  if (o.transition) return graphResultColumn(graphRunTransition(o, -1), 0);
  const graph = graphRequireGraph("pageRank");
  return graphResultColumn(graphRunBlocks(graph, null, o, "pageRank"), 0);
}

/**
 * Personalized PageRank with a one-hot restart at one user's node.
 *
 * Same update, stopping rule, zero-degree policy and return shape as `pageRank`;
 * only the restart vector v is one-hot at the seed node.
 *
 * FIRST ARGUMENT - the contract depends on `opts.transition`:
 *  - MovieLens path (no `opts.transition`): a RAW MovieLens user id, mapped
 *    with the graph's own `userIndexOf` (users come first, so the local user
 *    index is the global node index). The provided adapter passes the raw user
 *    id (hw6AdapterPPR, script.js:1451-1456) and script.js is not changed.
 *    Reading the id as a node index would restart every user one position too
 *    far, and user 943 would restart at movie 1. An id that is not a user node
 *    of the graph throws an Error naming the id; there is no uniform fallback.
 *  - `opts.transition` path: a node index in [0, Q.length) of the supplied Q
 *    (`REFERENCE_GRAPH.transition.nodeLabels` names the fixture's nodes). An
 *    out-of-range or non-integer index throws.
 *
 * BATCHING (MovieLens path, unless `opts.batched === false`): the first call
 * for a given (alpha, tol, maxIter) runs ONE batched power iteration for every
 * user of the graph. The restart matrix V is [nodeCount x nUsers] with one-hot
 * columns, and each iteration does the two block matMuls on all columns at
 * once. The stopping rule is applied per column: a column whose L1 residual
 * drops below tol is frozen (no further updates), so its p, iterations and
 * residual are those of a one-user run. The loop ends when every column is
 * frozen or at maxIter. The float32 results (about 9.6 MB per alpha for
 * MovieLens 100K) are cached on the graph object (`graph.pprCache`), keyed by
 * alpha, tol and maxIter; later calls are look-ups, and a rebuilt graph never
 * reuses a stale walk. `opts.batched = false` forces a single-column run and is
 * meant for verification. The `opts.transition` path is always one column.
 *
 * Invariant (checked by the tests): the restart adds (1 - alpha) * v at every
 * step and every term is nonnegative, so p[seed node] >= 1 - alpha.
 *
 * @param {number} userId Raw user id (MovieLens path) or node index (`opts.transition`).
 * @param {number} alpha  Link-following probability in [0, 1]; 1 - alpha restarts.
 * @param {{tol?: number, maxIter?: number, transition?: TransitionSpec, batched?: boolean}} [opts]
 *   Defaults tol = 1e-6, maxIter = 500, transition absent, batched true.
 * @returns {PageRankResult} `{p, iterations, residual, converged}` for this seed.
 * @throws {Error} for an unknown user id or an out-of-range node index, invalid
 *   options, or when neither `opts.transition` nor the global `GRAPH` exists.
 */
function personalizedPageRank(userId, alpha, opts) {
  const o = graphPropagationOptions("personalizedPageRank", alpha, opts);
  if (o.transition) {
    const seed = Number(userId);
    if (!Number.isInteger(seed) || seed < 0 || seed >= o.transition.n) {
      throw new Error(
        "personalizedPageRank: restart node index " + userId + " is outside [0, " + o.transition.n + ")"
      );
    }
    return graphResultColumn(graphRunTransition(o, seed), 0);
  }

  const graph = graphRequireGraph("personalizedPageRank");
  const local = graph.userIndexOf instanceof Map ? graph.userIndexOf.get(Number(userId)) : undefined;
  if (local === undefined) {
    throw new Error(
      "personalizedPageRank: user id " + userId +
      " is not a user node of the current graph (only users observed in the training split are nodes)"
    );
  }
  if (!o.batched) return graphResultColumn(graphRunBlocks(graph, [local], o, "personalizedPageRank"), 0);
  return graphResultColumn(graphPprBatch(graph, o), local);
}

/**
 * Validate the shared alpha / opts arguments of `pageRank` and
 * `personalizedPageRank` and fill in the defaults.
 *
 * @param {string} caller Function name used in error messages.
 * @param {*} alpha
 * @param {*} opts
 * @returns {{alpha:number, tol:number, maxIter:number, batched:boolean,
 *            transition:(Object|null)}} `transition` is the checked fixture
 *   from `graphCheckTransition`, or null.
 */
function graphPropagationOptions(caller, alpha, opts) {
  if (typeof tf === "undefined") throw new Error(caller + ": TensorFlow.js is not loaded");
  if (typeof alpha !== "number" || !(alpha >= 0 && alpha <= 1)) {
    throw new Error(caller + ": alpha must be a number in [0, 1], got " + alpha);
  }
  if (opts !== undefined && opts !== null && typeof opts !== "object") {
    throw new Error(caller + ": opts must be an object, got " + typeof opts);
  }
  const o = opts || {};
  const tol = o.tol === undefined ? 1e-6 : o.tol;
  const maxIter = o.maxIter === undefined ? 500 : o.maxIter;
  const batched = o.batched === undefined ? true : o.batched;
  if (typeof tol !== "number" || !(tol > 0) || !Number.isFinite(tol)) {
    throw new Error(caller + ": opts.tol must be a positive finite number, got " + tol);
  }
  if (!Number.isInteger(maxIter) || maxIter < 1) {
    throw new Error(caller + ": opts.maxIter must be an integer >= 1, got " + maxIter);
  }
  if (typeof batched !== "boolean") throw new Error(caller + ": opts.batched must be a boolean");
  const transition = o.transition === undefined || o.transition === null
    ? null
    : graphCheckTransition(caller, o.transition);
  return { alpha: alpha, tol: tol, maxIter: maxIter, batched: batched, transition: transition };
}

/**
 * Check a caller-supplied {@link TransitionSpec} and flatten it for a tensor.
 *
 * Q must be square with finite entries, and `danglingMask` must have one entry
 * per row (Uint8Array or Array; read by position). A row marked as dangling
 * must be all zero, as the TransitionSpec contract says; Q is never
 * re-normalized here.
 *
 * @param {string} caller
 * @param {TransitionSpec} transition
 * @returns {{n:number, flatQ:Float32Array, dangling:Float32Array}}
 */
function graphCheckTransition(caller, transition) {
  const Q = transition && transition.Q;
  if (!Array.isArray(Q) || Q.length === 0) {
    throw new Error(caller + ": opts.transition.Q must be a non-empty square array");
  }
  const n = Q.length;
  const mask = transition.danglingMask;
  if (!mask || mask.length !== n) {
    throw new Error(caller + ": opts.transition.danglingMask must have " + n + " entries");
  }
  const flatQ = new Float32Array(n * n);
  const dangling = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const row = Q[i];
    if (!row || row.length !== n) throw new Error(caller + ": opts.transition.Q row " + i + " must have " + n + " entries");
    dangling[i] = mask[i] ? 1 : 0;
    for (let j = 0; j < n; j++) {
      const value = Number(row[j]);
      if (!Number.isFinite(value)) throw new Error(caller + ": opts.transition.Q[" + i + "][" + j + "] is not finite");
      if (dangling[i] && value !== 0) {
        throw new Error(caller + ": danglingMask marks node " + i + " but Q row " + i + " is not all zero");
      }
      flatQ[i * n + j] = value;
    }
  }
  return { n: n, flatQ: flatQ, dangling: dangling };
}

/**
 * The global graph, or a clear Error when it is missing or has no R.
 *
 * @param {string} caller
 * @returns {Graph}
 */
function graphRequireGraph(caller) {
  if (!GRAPH) {
    throw new Error(
      caller + ": no graph - build one with buildGraph(train) and setGraph(graph), or pass opts.transition"
    );
  }
  if (!Array.isArray(GRAPH.R)) throw new Error(caller + ": the current graph has no R matrix");
  return GRAPH;
}

/**
 * Block sizes of a graph, checked against R.
 *
 * @param {Graph} graph
 * @param {string} caller
 * @returns {{nUsers:number, nMovies:number, n:number}}
 */
function graphBlockShape(graph, caller) {
  const R = graph.R;
  const nUsers = Number.isInteger(graph.nUsers) ? graph.nUsers : R.length;
  const nMovies = Number.isInteger(graph.nMovies) ? graph.nMovies : (R.length ? R[0].length : 0);
  if (R.length !== nUsers || R.some((row) => !row || row.length !== nMovies)) {
    throw new Error(caller + ": graph.R must have shape [nUsers][nMovies] = [" + nUsers + "][" + nMovies + "]");
  }
  return { nUsers: nUsers, nMovies: nMovies, n: nUsers + nMovies };
}

/**
 * Lift R into the tensors the block propagation needs.
 *
 * Returns long-lived tensors that the caller must dispose: `R` [nUsers x
 * nMovies], the inverse degrees `invDegUsers` [nUsers x 1] and `invDegMovies`
 * [nMovies x 1] (0 on zero-degree nodes, so their rows contribute nothing to
 * Q^T * p), and `dangling` [n x 1], 1 exactly at the zero-degree nodes. A
 * graph without users or without movies has no edges: `R` is then null and
 * every node is dangling.
 *
 * @param {Array<ArrayLike<number>>} R
 * @param {number} nUsers
 * @param {number} nMovies
 * @returns {{R:(tf.Tensor2D|null), invDegUsers:(tf.Tensor2D|null),
 *            invDegMovies:(tf.Tensor2D|null), dangling:tf.Tensor2D}}
 */
function graphBlockTensors(R, nUsers, nMovies) {
  if (nUsers === 0 || nMovies === 0) {
    return { R: null, invDegUsers: null, invDegMovies: null, dangling: tf.ones([nUsers + nMovies, 1]) };
  }
  const flat = new Float32Array(nUsers * nMovies);
  for (let u = 0; u < nUsers; u++) {
    const row = R[u];
    for (let i = 0; i < nMovies; i++) {
      const w = Number(row[i]) || 0;
      if (w !== 0) flat[u * nMovies + i] = w;
    }
  }
  return tf.tidy(() => {
    const matrix = tf.tensor2d(flat, [nUsers, nMovies]);
    const degUsers = tf.sum(matrix, 1, true);
    const degMovies = tf.reshape(tf.sum(matrix, 0), [nMovies, 1]);
    // mask / max(deg, eps) is 1 / deg on positive-degree nodes and 0 otherwise.
    const inverse = (deg) => tf.div(tf.cast(tf.greater(deg, 0), "float32"), tf.maximum(deg, 1e-12));
    const isZero = (deg) => tf.cast(tf.equal(deg, 0), "float32");
    return {
      R: matrix,
      invDegUsers: inverse(degUsers),
      invDegMovies: inverse(degMovies),
      dangling: tf.concat([isZero(degUsers), isZero(degMovies)], 0)
    };
  });
}

/**
 * Link term Q^T * p for the bipartite graph, from the two off-diagonal blocks.
 *
 * For p of shape [n x B] (users first, then movies):
 *   user rows  <- R   x (p_movies * invDegMovies)   movie -> user links
 *   movie rows <- R^T x (p_users  * invDegUsers)    user -> movie links (transposeA)
 * The square matrix is never built. Must be called inside `tf.tidy`.
 *
 * @param {ReturnType<typeof graphBlockTensors>} blocks
 * @param {number} nUsers
 * @param {number} nMovies
 * @returns {function(tf.Tensor2D): tf.Tensor2D}
 */
function graphBlockLink(blocks, nUsers, nMovies) {
  return function (p) {
    if (!blocks.R) return tf.zerosLike(p);
    const columns = p.shape[1];
    const pUsers = tf.slice(p, [0, 0], [nUsers, columns]);
    const pMovies = tf.slice(p, [nUsers, 0], [nMovies, columns]);
    const toUsers = tf.matMul(blocks.R, tf.mul(pMovies, blocks.invDegMovies));
    const toMovies = tf.matMul(blocks.R, tf.mul(pUsers, blocks.invDegUsers), true, false);
    return tf.concat([toUsers, toMovies], 0);
  };
}

/**
 * One-hot restart columns: an [n x B] float32 tensor with a 1 at row
 * `indices[c]` of column c.
 *
 * @param {Int32Array|number[]} indices Node indices, one per column.
 * @param {number} n Node count.
 * @returns {tf.Tensor2D}
 */
function graphOneHotColumns(indices, n) {
  return tf.tidy(() => tf.transpose(tf.oneHot(tf.tensor1d(Int32Array.from(indices), "int32"), n, 1, 0, "float32")));
}

/**
 * Batched power iteration with a per-column stopping rule (readme.md 3.3-3.4).
 *
 * Every column c of `restart` is one restart distribution v_c; p starts at V.
 * One update for all columns:
 *
 *   d    = sum over zero-degree nodes of p          (one value per column)
 *   next = alpha * link(p) + (alpha * d + 1 - alpha) * V
 *
 * which is alpha * (Q^T p + d v) + (1 - alpha) v column by column. The L1
 * change of every column is read back with `dataSync()` (B floats). A column
 * whose change is below tol is frozen: its p, iteration count and residual stay
 * as they were after that update, exactly as a one-column run would stop.
 * Frozen columns are carried over by a 0/1 blend, which is exact. The loop ends
 * when every column is frozen or after maxIter updates.
 *
 * Each update runs inside `tf.tidy` (no async callback); the previous p is
 * disposed as soon as the new one exists.
 *
 * @param {function(tf.Tensor2D): tf.Tensor2D} link Q^T * p, called inside tidy.
 * @param {tf.Tensor2D} dangling [n x 1] 0/1 mask of zero-degree nodes.
 * @param {tf.Tensor2D} restart  [n x B] restart columns, each summing to 1.
 * @param {{alpha:number, tol:number, maxIter:number}} o
 * @returns {{p:tf.Tensor2D, iterations:Int32Array, residual:Float64Array,
 *            converged:Uint8Array}} `p` [n x B] must be disposed by the caller.
 */
function graphPowerIterate(link, dangling, restart, o) {
  const columns = restart.shape[1];
  const active = new Float32Array(columns).fill(1);
  const iterations = new Int32Array(columns);
  const residual = new Float64Array(columns);
  const converged = new Uint8Array(columns);
  let open = columns;
  let p = tf.clone(restart);
  let activeMask = null;
  try {
    for (let k = 1; k <= o.maxIter && open > 0; k++) {
      const step = tf.tidy(() => {
        const d = tf.sum(tf.mul(p, dangling), 0, true);
        const restartWeight = tf.add(tf.mul(d, o.alpha), 1 - o.alpha);
        const next = tf.add(tf.mul(link(p), o.alpha), tf.mul(restart, restartWeight));
        const change = tf.sum(tf.abs(tf.sub(next, p)), 0);
        const kept = activeMask
          ? tf.add(tf.mul(next, activeMask), tf.mul(p, tf.sub(1, activeMask)))
          : next;
        return { kept: kept, change: change };
      });
      p.dispose();
      p = step.kept;
      let change;
      try {
        change = step.change.dataSync();
      } finally {
        step.change.dispose();
      }

      let frozen = false;
      for (let c = 0; c < columns; c++) {
        if (!active[c]) continue;
        iterations[c] = k;
        residual[c] = change[c];
        if (change[c] < o.tol) {
          converged[c] = 1;
          active[c] = 0;
          open -= 1;
          frozen = true;
        }
      }
      if (frozen && open > 0) {
        if (activeMask) activeMask.dispose();
        activeMask = tf.tensor2d(active, [1, columns]);
      }
    }
  } catch (error) {
    p.dispose();
    throw error;
  } finally {
    if (activeMask) activeMask.dispose();
  }
  return { p: p, iterations: iterations, residual: residual, converged: converged };
}

/**
 * Read the final [n x B] tensor back once, column-major (one column = one run).
 *
 * @param {{p:tf.Tensor2D, iterations:Int32Array, residual:Float64Array,
 *          converged:Uint8Array}} out Result of `graphPowerIterate`.
 * @param {number} n Node count.
 * @returns {{n:number, columns:number, values:Float32Array, iterations:Int32Array,
 *            residual:Float64Array, converged:Uint8Array}}
 */
function graphFinishRun(out, n) {
  const transposed = tf.tidy(() => tf.transpose(out.p));
  let values;
  try {
    values = transposed.dataSync().slice();
  } finally {
    transposed.dispose();
  }
  return {
    n: n,
    columns: out.iterations.length,
    values: values,
    iterations: out.iterations,
    residual: out.residual,
    converged: out.converged
  };
}

/**
 * Release the tensors of one run; null entries are skipped.
 *
 * @param {Array<tf.Tensor|null|undefined>} tensors
 * @returns {void}
 */
function graphDisposeAll(tensors) {
  for (const t of tensors) {
    if (t) t.dispose();
  }
}

/**
 * Run the power iteration on a caller-supplied transition (`opts.transition`).
 *
 * The supplied Q is square by contract (the 4-node fixture), so this path is
 * the only one that multiplies by a square matrix: Q^T * p = matMul(Q, p,
 * transposeA).
 *
 * @param {{alpha:number, tol:number, maxIter:number, transition:Object}} o
 * @param {number} seed Restart node index, or -1 for the uniform restart.
 * @returns {ReturnType<typeof graphFinishRun>}
 */
function graphRunTransition(o, seed) {
  const n = o.transition.n;
  let Q = null;
  let dangling = null;
  let restart = null;
  let out = null;
  try {
    Q = tf.tensor2d(o.transition.flatQ, [n, n]);
    dangling = tf.tensor2d(o.transition.dangling, [n, 1]);
    restart = seed < 0 ? tf.fill([n, 1], 1 / n) : graphOneHotColumns([seed], n);
    out = graphPowerIterate((p) => tf.matMul(Q, p, true, false), dangling, restart, o);
    return graphFinishRun(out, n);
  } finally {
    graphDisposeAll([Q, dangling, restart, out && out.p]);
  }
}

/**
 * Run the power iteration on a bipartite graph through the R blocks.
 *
 * @param {Graph} graph
 * @param {Int32Array|number[]|null} seeds Local user indices (one column each),
 *   or null for one uniform restart column.
 * @param {{alpha:number, tol:number, maxIter:number}} o
 * @param {string} caller
 * @returns {ReturnType<typeof graphFinishRun>}
 */
function graphRunBlocks(graph, seeds, o, caller) {
  const shape = graphBlockShape(graph, caller);
  const n = shape.n;
  if (n === 0) throw new Error(caller + ": the graph has no nodes (empty training split?)");
  let blocks = null;
  let restart = null;
  let out = null;
  try {
    blocks = graphBlockTensors(graph.R, shape.nUsers, shape.nMovies);
    restart = seeds === null ? tf.fill([n, 1], 1 / n) : graphOneHotColumns(seeds, n);
    out = graphPowerIterate(graphBlockLink(blocks, shape.nUsers, shape.nMovies), blocks.dangling, restart, o);
    return graphFinishRun(out, n);
  } finally {
    graphDisposeAll([
      blocks && blocks.R, blocks && blocks.invDegUsers, blocks && blocks.invDegMovies,
      blocks && blocks.dangling, restart, out && out.p
    ]);
  }
}

/**
 * One column of a run as a `PageRankResult`.
 *
 * @param {ReturnType<typeof graphFinishRun>} run
 * @param {number} column
 * @returns {PageRankResult}
 */
function graphResultColumn(run, column) {
  const start = column * run.n;
  return {
    p: Float64Array.from(run.values.subarray(start, start + run.n)),
    iterations: run.iterations[column],
    residual: run.residual[column],
    converged: run.converged[column] === 1
  };
}

/**
 * The PPR cache of a graph (created on first use for graphs that were not made
 * by `buildGraph`, e.g. the self-check fixtures).
 *
 * @param {Graph} graph
 * @returns {Map<string, Object>}
 */
function graphPprCacheOf(graph) {
  if (!(graph.pprCache instanceof Map)) graph.pprCache = new Map();
  return graph.pprCache;
}

/**
 * Batched PPR for every user of `graph` at (alpha, tol, maxIter), computed once
 * and cached on the graph.
 *
 * Column c of the entry is the run restarted at local user c (global node c).
 *
 * @param {Graph} graph
 * @param {{alpha:number, tol:number, maxIter:number}} o
 * @returns {Object} cache entry: the `graphFinishRun` fields plus alpha, tol,
 *   maxIter, seconds and megabytes.
 */
function graphPprBatch(graph, o) {
  const cache = graphPprCacheOf(graph);
  const key = o.alpha + "|" + o.tol + "|" + o.maxIter;
  let entry = cache.get(key);
  if (entry) return entry;

  const started = performance.now();
  const shape = graphBlockShape(graph, "personalizedPageRank");
  const seeds = new Int32Array(shape.nUsers);
  for (let u = 0; u < shape.nUsers; u++) seeds[u] = u;
  const run = graphRunBlocks(graph, seeds, o, "personalizedPageRank");
  entry = Object.assign(run, {
    alpha: o.alpha,
    tol: o.tol,
    maxIter: o.maxIter,
    seconds: (performance.now() - started) / 1000,
    megabytes: (run.values.byteLength + run.iterations.byteLength +
      run.residual.byteLength + run.converged.byteLength) / 1e6
  });
  cache.set(key, entry);
  return entry;
}

/**
 * Size of a graph's PPR cache, for the report.
 *
 * @param {Graph} graph
 * @returns {{entries:number, megabytes:number, keys:string[]}}
 */
function graphPprCacheInfo(graph) {
  const cache = graph && graph.pprCache instanceof Map ? graph.pprCache : new Map();
  let megabytes = 0;
  cache.forEach((entry) => { megabytes += entry.megabytes || 0; });
  return { entries: cache.size, megabytes: megabytes, keys: Array.from(cache.keys()) };
}

// ---------------------------------------------------------------------------
// Small helpers (private, all graph-prefixed to avoid collisions)
// ---------------------------------------------------------------------------

/**
 * Readable message from an unknown thrown value.
 *
 * @param {*} error
 * @returns {string}
 */
function graphErrorMessage(error) {
  return String(error && error.message ? error.message : error);
}

/**
 * Escape text before inserting it into HTML (the local copy, so this file does
 * not depend on any helper defined by script.js).
 *
 * @param {*} text
 * @returns {string}
 */
function graphEscape(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/**
 * Read alpha from the #alpha range input, falling back to 0.85.
 *
 * @returns {number}
 */
function graphReadAlpha() {
  const el = document.getElementById("alpha");
  const value = el ? Number(el.value) : Number.NaN;
  return Number.isFinite(value) ? value : 0.85;
}

/**
 * Map t in [0, 1] onto the single-hue sequential blue ramp.
 *
 * @param {number} t
 * @returns {string} an rgb(...) string.
 */
function graphRampColor(t) {
  const n = Number(t);
  const clamped = Math.max(0, Math.min(1, Number.isFinite(n) ? n : 0));
  const lo = graphTheme.rampLight;
  const hi = graphTheme.rampDark;
  const mix = (a, b, s) => Math.round(a + (b - a) * s);
  const r = mix(lo[0], hi[0], clamped);
  const g = mix(lo[1], hi[1], clamped);
  const b = mix(lo[2], hi[2], clamped);
  return "rgb(" + r + ", " + g + ", " + b + ")";
}

/**
 * Position lookup with a safe fallback for a missing node.
 *
 * @param {Map<number,{x:number,y:number}>} positions
 * @param {number} node
 * @returns {{x:number,y:number}}
 */
function graphPos(positions, node) {
  const pt = positions.get(node);
  return pt || { x: 0, y: 0 };
}

// ---------------------------------------------------------------------------
// Fixture demo helpers (private)
// ---------------------------------------------------------------------------

/**
 * Out-degree of a fixture node (number of outgoing reference edges).
 *
 * @param {number} index
 * @returns {number}
 */
function graphFixtureOutDegree(index) {
  let d = 0;
  for (const edge of REFERENCE_GRAPH.edges) {
    if (edge[0] === index) d += 1;
  }
  return d;
}

/**
 * Build the row-stochastic transition for the 4-node reference fixture.
 *
 * @returns {{P: number[][], out: number[]}} P[from][to] and the out-degrees.
 */
function graphFixtureTransition() {
  const n = REFERENCE_GRAPH.nodes.length;
  const out = new Array(n).fill(0);
  for (const edge of REFERENCE_GRAPH.edges) out[edge[0]] += 1;

  const P = [];
  for (let i = 0; i < n; i++) P.push(new Array(n).fill(0));
  for (const edge of REFERENCE_GRAPH.edges) {
    const from = edge[0];
    const to = edge[1];
    if (out[from] > 0) P[from][to] = 1 / out[from];
  }
  return { P, out };
}

/**
 * Restart distribution for the fixture: uniform, or one-hot at a named node.
 *
 * @param {string} restart "uniform" or one of A/B/C/D.
 * @returns {number[]}
 */
function graphFixtureRestartVector(restart) {
  const n = REFERENCE_GRAPH.nodes.length;
  const v = new Array(n).fill(0);
  if (!restart || restart === "uniform") {
    v.fill(1 / n);
    return v;
  }
  const index = REFERENCE_GRAPH.nodeOrder.indexOf(restart);
  if (index < 0) {
    v.fill(1 / n);
    return v;
  }
  v[index] = 1;
  return v;
}

/**
 * One plain-JavaScript propagation step on the 4-node fixture.
 *
 * The fixture has no dangling nodes, so the d * v term from readme.md
 * section 3.3 is not needed here. Returns the new distribution and the L1
 * change.
 *
 * @param {number} alpha
 * @param {string} restart
 * @param {number[]} pPrev
 * @returns {{p: number[], residual: number}}
 */
function graphFixturePropagate(alpha, restart, pPrev) {
  const transition = graphFixtureTransition();
  const P = transition.P;
  const n = REFERENCE_GRAPH.nodes.length;
  const v = graphFixtureRestartVector(restart);
  const pNext = new Array(n).fill(0);

  // p_next[j] = alpha * sum_i p[i] * P[i][j] + (1 - alpha) * v[j]
  for (let i = 0; i < n; i++) {
    const mass = pPrev[i];
    if (!mass) continue;
    for (let j = 0; j < n; j++) {
      if (P[i][j] !== 0) pNext[j] += alpha * mass * P[i][j];
    }
  }
  for (let j = 0; j < n; j++) pNext[j] += (1 - alpha) * v[j];

  let residual = 0;
  for (let j = 0; j < n; j++) residual += Math.abs(pNext[j] - pPrev[j]);
  return { p: pNext, residual };
}

// ---------------------------------------------------------------------------
// View builders (private)
// ---------------------------------------------------------------------------

/**
 * Build the drawable view for the 4-node fixture.
 *
 * @returns {{kind:string, nodes:Object[], edges:Object[], omitted:number,
 *            nodeCount:number, note:string}}
 */
function graphBuildFixtureView() {
  const fixture = GRAPH_VIEW_STATE.fixture;
  const p = fixture.p || graphFixtureRestartVector(fixture.restart || "uniform");
  const nodes = REFERENCE_GRAPH.nodes.map((nd) => ({
    node: nd.index,
    key: "fixture:" + nd.label,
    label: nd.label,
    kind: nd.kind,
    id: nd.label,
    degree: graphFixtureOutDegree(nd.index),
    score: p[nd.index]
  }));
  const edges = REFERENCE_GRAPH.edges.map((edge) => ({
    source: edge[0],
    target: edge[1],
    directed: true
  }));
  return {
    kind: "fixture",
    nodes: nodes,
    edges: edges,
    omitted: 0,
    nodeCount: nodes.length,
    note: "4-node reference fixture (A, B, C, D) - no dangling nodes"
  };
}

/**
 * Neighbours of a node in the bipartite graph (users <-> movies), or [] when
 * the graph is missing. Uses the users-then-movies node order from `buildGraph`.
 *
 * @param {Graph} graph
 * @param {number} node
 * @returns {number[]}
 */
function graphNeighbors(graph, node) {
  const result = [];
  if (!graph || !graph.R) return result;
  if (graph.isUserNode(node)) {
    const row = graph.R[node];
    if (!row) return result;
    for (let i = 0; i < row.length; i++) {
      if (row[i]) result.push(graph.nUsers + i);
    }
  } else {
    const movieLocal = node - graph.nUsers;
    for (let u = 0; u < graph.nUsers; u++) {
      const row = graph.R[u];
      if (row && row[movieLocal]) result.push(u);
    }
  }
  return result;
}

/**
 * Build the capped ego subgraph for a seed user.
 *
 * Selection: the seed user's node plus its neighbours and their neighbours (a
 * 2-hop walk) in the bipartite graph, then apply the maxNodes cap in
 * deterministic node-index order. `omitted` is how many reachable nodes the cap
 * left out, so the legend can report it.
 *
 * @param {number|null} seedUserId
 * @param {function(number):number} [getScore]
 * @returns {{kind:string, nodes:Object[], edges:Object[], omitted:number,
 *            nodeCount:number, note:string}}
 */
function graphBuildEgoView(seedUserId, getScore) {
  const graph = GRAPH;
  if (!graph || !graph.nodeCount) {
    return {
      kind: "empty",
      nodes: [],
      edges: [],
      omitted: 0,
      nodeCount: 0,
      note: "No graph yet - build the graph (TODO(hw6) buildGraph) to draw the ego subgraph."
    };
  }

  const seed = seedUserId === null || seedUserId === undefined ? null : Number(seedUserId);
  if (seed === null || !graph.userIndexOf || !graph.userIndexOf.has(seed)) {
    return {
      kind: "empty",
      nodes: [],
      edges: [],
      omitted: 0,
      nodeCount: graph.nodeCount,
      note: "Pick a seed user (click a user node) to draw its ego subgraph."
    };
  }

  const maxNodes = Math.max(1, Math.floor(GRAPH_VIEW_STATE.maxNodes) || 120);
  const startNode = graph.nodeOfUser(seed);

  // Unbounded 2-hop reachable set, then cap deterministically.
  const reachable = new Set([startNode]);
  let frontier = [startNode];
  for (let depth = 0; depth < 2; depth++) {
    const next = [];
    for (const node of frontier) {
      for (const nb of graphNeighbors(graph, node)) {
        if (!reachable.has(nb)) {
          reachable.add(nb);
          next.push(nb);
        }
      }
    }
    frontier = next;
    if (reachable.size > 500000) break; // hard safety valve
  }

  const ordered = [startNode].concat(
    Array.from(reachable)
      .filter((node) => node !== startNode)
      .sort((a, b) => a - b)
  );
  const selected = new Set(ordered.slice(0, maxNodes));
  const omitted = reachable.size - selected.size;

  const nodes = [];
  for (const node of ordered) {
    if (!selected.has(node)) continue;
    const isUser = graph.isUserNode(node);
    const id = isUser ? graph.userIds[node] : graph.movieOfNode(node);
    let score = null;
    if (typeof getScore === "function") {
      const raw = Number(getScore(node));
      score = Number.isFinite(raw) ? raw : null;
    }
    nodes.push({
      node: node,
      key: "node:" + node,
      label: isUser ? "U" + id : "M" + id,
      kind: isUser ? "user" : "movie",
      id: id,
      degree: graph.degree ? graph.degree[node] : 0,
      score: score
    });
  }

  const edges = [];
  for (const a of nodes) {
    if (a.kind !== "user") continue;
    const uLocal = graph.userIndexOf.get(a.id);
    const row = graph.R ? graph.R[uLocal] : null;
    if (!row) continue;
    for (const b of nodes) {
      if (b.kind !== "movie") continue;
      const mLocal = graph.movieIndexOf.get(b.id);
      if (mLocal === undefined) continue;
      if (row[mLocal]) edges.push({ source: a.node, target: b.node, directed: false });
    }
  }

  return {
    kind: "ego",
    nodes: nodes,
    edges: edges,
    omitted: omitted,
    nodeCount: graph.nodeCount,
    note: "Ego subgraph of user " + seed + " - drawn " + nodes.length + " of " + graph.nodeCount + " nodes"
  };
}

/**
 * Deterministic two-column bipartite layout: users on the left, movies on the
 * right, evenly spaced top-to-bottom in ascending node order.
 *
 * @param {Object[]} nodes
 * @param {number} width
 * @param {number} height
 * @param {{top:number,right:number,bottom:number,left:number}} margin
 * @returns {{positions: Map<number,{x:number,y:number}>, leftX:number, rightX:number}}
 */
function graphLayoutTwoColumn(nodes, width, height, margin) {
  const positions = new Map();
  const usableWidth = width - margin.left - margin.right;
  const usableHeight = height - margin.top - margin.bottom;
  const leftX = margin.left + usableWidth * 0.2;
  const rightX = margin.left + usableWidth * 0.8;

  const place = (list, x) => {
    const n = list.length;
    list.forEach((item, i) => {
      const y = n <= 1 ? margin.top + usableHeight / 2 : margin.top + (usableHeight * i) / (n - 1);
      positions.set(item.node, { x: x, y: y });
    });
  };

  const users = nodes.filter((n) => n.kind === "user").sort((a, b) => a.node - b.node);
  const movies = nodes.filter((n) => n.kind === "movie").sort((a, b) => a.node - b.node);
  const others = nodes
    .filter((n) => n.kind !== "user" && n.kind !== "movie")
    .sort((a, b) => a.node - b.node);

  place(users, leftX);
  place(movies, rightX);
  place(others, leftX);

  return { positions: positions, leftX: leftX, rightX: rightX };
}

// ---------------------------------------------------------------------------
// d3 drawing (private)
// ---------------------------------------------------------------------------

/**
 * Draw one view (fixture or ego) into the #graph-view SVG element.
 *
 * Read-only with respect to GRAPH and any score arrays: it only reads nodes,
 * edges and score values. Zero-degree nodes are drawn as hollow rings with a
 * label so the dangling case is visible.
 *
 * @param {SVGElement} svgEl
 * @param {Object} view result of `graphBuildFixtureView` / `graphBuildEgoView`.
 * @param {{scoreAccessor: (function(number):number|null)|null, colorByScore: boolean}} opts
 * @returns {void}
 */
function graphDrawView(svgEl, view, opts) {
  const scoreAccessor = opts && typeof opts.scoreAccessor === "function" ? opts.scoreAccessor : null;
  const colorByScore = !!(opts && opts.colorByScore);
  const width = graphTheme.width;
  const height = graphTheme.height;
  const margin = graphTheme.margin;

  const svg = d3.select(svgEl);
  svg.selectAll("*").remove();
  svg.attr("viewBox", "0 0 " + width + " " + height);
  svg.attr("preserveAspectRatio", "xMidYMid meet");

  const defs = svg.append("defs");
  defs
    .append("marker")
    .attr("id", "graph-arrowhead")
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 10)
    .attr("refY", 0)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,-5L10,0L0,5")
    .attr("fill", graphTheme.edgeColor);

  if (!view || view.nodes.length === 0) {
    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", height / 2)
      .attr("text-anchor", "middle")
      .attr("font-size", 15)
      .attr("fill", graphTheme.noteColor)
      .text(view && view.note ? view.note : "Nothing to draw yet.");
    return;
  }

  const layout = graphLayoutTwoColumn(view.nodes, width, height, margin);
  const positions = layout.positions;

  svg
    .append("text")
    .attr("x", layout.leftX)
    .attr("y", margin.top - 12)
    .attr("text-anchor", "middle")
    .attr("font-size", 12)
    .attr("fill", graphTheme.noteColor)
    .text("users");
  svg
    .append("text")
    .attr("x", layout.rightX)
    .attr("y", margin.top - 12)
    .attr("text-anchor", "middle")
    .attr("font-size", 12)
    .attr("fill", graphTheme.noteColor)
    .text("movies");

  // Resolve scores and the shared scales.
  const scoreByNode = new Map();
  let maxScore = 0;
  for (const n of view.nodes) {
    let score = null;
    if (scoreAccessor) {
      const raw = Number(scoreAccessor(n.node));
      score = Number.isFinite(raw) ? raw : null;
    } else if (n.score !== null && n.score !== undefined) {
      const raw = Number(n.score);
      score = Number.isFinite(raw) ? raw : null;
    }
    scoreByNode.set(n.node, score);
    if (score !== null && score > maxScore) maxScore = score;
  }
  const maxDegree = view.nodes.reduce((m, n) => Math.max(m, Number(n.degree) || 0), 1);

  const normFor = (n) => {
    if (colorByScore) {
      const score = scoreByNode.get(n.node);
      return score !== null && maxScore > 0 ? Math.min(1, score / maxScore) : 0;
    }
    const deg = Math.max(0, Number(n.degree) || 0);
    return maxDegree > 0 ? Math.min(1, deg / maxDegree) : 0;
  };
  const radiusFor = (n) =>
    graphTheme.minRadius +
    (graphTheme.maxRadius - graphTheme.minRadius) * Math.sqrt(normFor(n));

  // Edges first so nodes sit on top.
  svg
    .append("g")
    .attr("class", "graph-edges")
    .selectAll("line")
    .data(view.edges)
    .join("line")
    .attr("x1", (d) => graphPos(positions, d.source).x)
    .attr("y1", (d) => graphPos(positions, d.source).y)
    .attr("x2", (d) => graphPos(positions, d.target).x)
    .attr("y2", (d) => graphPos(positions, d.target).y)
    .attr("stroke", graphTheme.edgeColor)
    .attr("stroke-width", 1)
    .attr("marker-end", (d) => (d.directed ? "url(#graph-arrowhead)" : null));

  const nodeSel = svg
    .append("g")
    .attr("class", "graph-nodes")
    .selectAll("g.graph-node")
    .data(view.nodes, (d) => d.key)
    .join("g")
    .attr("class", "graph-node")
    .attr("transform", (d) => {
      const pt = graphPos(positions, d.node);
      return "translate(" + pt.x + "," + pt.y + ")";
    })
    .style("cursor", "pointer");

  nodeSel.each(function eachNode(d) {
    const group = d3.select(this);
    const degree = Math.max(0, Number(d.degree) || 0);
    const isZero = degree === 0;
    const radius = radiusFor(d);
    const score = scoreByNode.get(d.node);

    let fill = d.kind === "user" ? graphTheme.userColor : graphTheme.movieColor;
    if (colorByScore) fill = graphRampColor(normFor(d));

    group
      .append("circle")
      .attr("r", radius)
      .attr("fill", isZero ? "#ffffff" : fill)
      .attr("stroke", isZero ? graphTheme.zeroStroke : graphTheme.nodeStroke)
      .attr("stroke-width", isZero ? 2.5 : 1.5);

    // Every fixture node is labelled; in the ego view only zero-degree nodes and
    // the seed user are labelled to keep a dense view readable.
    const showLabel =
      view.kind === "fixture" ||
      isZero ||
      (view.kind === "ego" && d.kind === "user" && d.id === GRAPH_VIEW_STATE.seedUserId);

    if (showLabel) {
      group
        .append("text")
        .attr("x", 0)
        .attr("y", -(radius + 4))
        .attr("text-anchor", "middle")
        .attr("font-size", 11)
        .attr("fill", graphTheme.labelColor)
        .attr("pointer-events", "none")
        .text(d.label);
    }

    group.on("mouseenter", (event) => graphShowTooltip(event, d, score));
    group.on("mousemove", (event) => graphShowTooltip(event, d, score));
    group.on("mouseleave", graphHideTooltip);
    group.on("click", () => graphHandleNodeClick(d, view));
  });
}

// ---------------------------------------------------------------------------
// Tooltip (private)
// ---------------------------------------------------------------------------

/**
 * Lazily create and return the single shared tooltip element.
 *
 * @returns {HTMLElement}
 */
function graphEnsureTooltip() {
  let el = document.getElementById(graphTheme.tooltipId);
  if (el) return el;
  el = document.createElement("div");
  el.id = graphTheme.tooltipId;
  el.setAttribute("role", "tooltip");
  el.style.position = "absolute";
  el.style.pointerEvents = "none";
  el.style.background = "#1c2128";
  el.style.color = "#ffffff";
  el.style.padding = "0.35rem 0.5rem";
  el.style.borderRadius = "6px";
  el.style.fontSize = "0.78rem";
  el.style.lineHeight = "1.35";
  el.style.zIndex = "10";
  el.style.whiteSpace = "nowrap";
  el.style.display = "none";
  document.body.appendChild(el);
  return el;
}

/**
 * Show the hover tooltip for a node: id, kind, degree and current score.
 *
 * @param {MouseEvent} event
 * @param {{id:(number|string), kind:string, node:number, degree:number}} d
 * @param {number|null} score
 * @returns {void}
 */
function graphShowTooltip(event, d, score) {
  const el = graphEnsureTooltip();
  const kind = d.kind === "user" ? "user" : "movie";
  const scoreText = score === null || score === undefined || !Number.isFinite(Number(score))
    ? "-"
    : Number(score).toFixed(6);
  el.textContent =
    kind + " " + d.id + "  |  node " + d.node + "  |  degree " + d.degree + "  |  score " + scoreText;
  el.style.display = "block";
  const left = (event && event.pageX ? event.pageX : 0) + 12;
  const top = (event && event.pageY ? event.pageY : 0) + 12;
  el.style.left = left + "px";
  el.style.top = top + "px";
}

/**
 * Hide the hover tooltip.
 *
 * @returns {void}
 */
function graphHideTooltip() {
  const el = document.getElementById(graphTheme.tooltipId);
  if (el) el.style.display = "none";
}

/**
 * Handle a click on a node: in the fixture mode it moves the restart node; in
 * the ego mode it selects a seed user. Never mutates GRAPH or any score array.
 *
 * @param {{label:string, kind:string, id:(number|string)}} d
 * @param {{kind:string}} view
 * @returns {void}
 */
function graphHandleNodeClick(d, view) {
  if (view.kind === "fixture") {
    GRAPH_VIEW_STATE.fixture.restart = d.label;
    fixtureReset();
    return;
  }
  if (view.kind === "ego" && d.kind === "user") {
    GRAPH_VIEW_STATE.seedUserId = Number(d.id);
    renderGraphView({ mode: "ego", seedUserId: Number(d.id) });
  }
}

// ---------------------------------------------------------------------------
// Legend and controls (private)
// ---------------------------------------------------------------------------

/**
 * A coloured square swatch for the legend.
 *
 * @param {string} color
 * @returns {string} HTML
 */
function graphSwatch(color) {
  return '<span class="legend-swatch" style="background:' + color + '"></span>';
}

/**
 * A hollow-ring swatch for the zero-degree legend entry.
 *
 * @returns {string} HTML
 */
function graphRingSwatch() {
  return (
    '<span class="legend-swatch" style="background:#ffffff;border:2px solid ' +
    graphTheme.zeroStroke +
    ';border-radius:50%"></span>'
  );
}

/**
 * A gradient swatch showing the single-hue sequential ramp.
 *
 * @returns {string} HTML
 */
function graphRampSwatch() {
  return (
    '<span class="legend-swatch" style="background:linear-gradient(90deg, ' +
    graphRampColor(0) +
    ", " +
    graphRampColor(1) +
    ')"></span>'
  );
}

/**
 * One legend row: swatch (may be empty) + text.
 *
 * @param {string} swatchHtml
 * @param {string} text
 * @returns {string} HTML
 */
function graphLegendItem(swatchHtml, text) {
  return (
    '<span class="legend-item">' + swatchHtml + "<span>" + graphEscape(text) + "</span></span>"
  );
}

/**
 * Replace the legend with a plain message (used for errors / unavailable d3).
 *
 * @param {string} message
 * @returns {void}
 */
function graphSetLegendMessage(message) {
  const target = document.getElementById("graph-legend");
  if (!target) return;
  target.innerHTML = '<p class="empty-state">' + graphEscape(message) + "</p>";
}

/**
 * Write one line per colour/size meaning into #graph-legend.
 *
 * @param {Object} view
 * @param {{colorByScore: boolean}} opts
 * @returns {void}
 */
function graphRenderLegend(view, opts) {
  const target = document.getElementById("graph-legend");
  if (!target) return;
  const colorByScore = !!(opts && opts.colorByScore);
  const items = [];

  if (view && view.kind === "fixture") {
    items.push(graphLegendItem(graphSwatch(graphTheme.userColor), "user node (A, B)"));
    items.push(graphLegendItem(graphSwatch(graphTheme.movieColor), "movie node (C, D)"));
    items.push(graphLegendItem(graphRampSwatch(), "fill = current probability mass (single-hue ramp)"));
    items.push(graphLegendItem(graphRingSwatch(), "hollow ring = zero-degree node"));
    items.push(graphLegendItem("", "arrow = link direction; this fixture has no dangling nodes"));
    items.push(graphLegendItem("", "radius = outgoing degree; click a node to restart there"));
    target.innerHTML = items.join("");
    return;
  }

  if (!view || view.kind === "empty") {
    const note = view && view.note ? view.note : "No graph to draw yet.";
    target.innerHTML = '<p class="empty-state">' + graphEscape(note) + "</p>";
    return;
  }

  items.push(graphLegendItem(graphSwatch(graphTheme.userColor), "user node (left column)"));
  items.push(graphLegendItem(graphSwatch(graphTheme.movieColor), "movie node (right column)"));
  if (colorByScore) {
    items.push(graphLegendItem(graphRampSwatch(), "fill = score (single-hue sequential ramp)"));
  } else {
    items.push(graphLegendItem("", "fill = node kind; switch colour-by to score to use the ramp"));
  }
  items.push(graphLegendItem(graphRingSwatch(), "hollow ring with label = zero-degree node"));
  items.push(
    graphLegendItem(
      "",
      colorByScore ? "radius scales with score" : "radius scales with degree"
    )
  );
  if (view.omitted > 0) {
    items.push(
      graphLegendItem(
        "",
        view.omitted + " node(s) omitted by the " + GRAPH_VIEW_STATE.maxNodes + "-node cap"
      )
    );
  }
  target.innerHTML = items.join("");
}

/**
 * Fill #graph-controls with the view-mode controls.
 *
 * @returns {void}
 */
function graphRenderControls() {
  const target = document.getElementById("graph-controls");
  if (!target) return;
  const state = GRAPH_VIEW_STATE;

  const modeOptions = ["fixture", "ego"]
    .map((m) => '<option value="' + m + '"' + (state.mode === m ? " selected" : "") + ">" + m + "</option>")
    .join("");
  const colorOptions = ["degree", "score"]
    .map((c) => '<option value="' + c + '"' + (state.colorBy === c ? " selected" : "") + ">" + c + "</option>")
    .join("");
  const restart = state.fixture.restart || "uniform";
  const restartOptions = ["uniform"].concat(REFERENCE_GRAPH.nodeOrder)
    .map((r) => '<option value="' + r + '"' + (restart === r ? " selected" : "") + ">" + r + "</option>")
    .join("");

  const seedText = state.seedUserId === null ? "- (click a user node)" : String(state.seedUserId);

  target.innerHTML =
    '<label class="legend-item">View <select id="graph-mode-select">' + modeOptions + "</select></label>" +
    '<label class="legend-item">Colour by <select id="graph-colorby-select"' +
    (state.mode === "fixture" ? " disabled" : "") +
    ">" + colorOptions + "</select></label>" +
    '<label class="legend-item">Fixture restart <select id="graph-restart-select"' +
    (state.mode === "fixture" ? "" : " disabled") +
    ">" + restartOptions + "</select></label>" +
    '<span class="hint">Seed user: ' + graphEscape(seedText) + "</span>";

  const modeSelect = document.getElementById("graph-mode-select");
  if (modeSelect) {
    modeSelect.addEventListener("change", () => renderGraphView({ mode: modeSelect.value }));
  }
  const colorSelect = document.getElementById("graph-colorby-select");
  if (colorSelect) {
    colorSelect.addEventListener("change", () => renderGraphView({ colorBy: colorSelect.value }));
  }
  const restartSelect = document.getElementById("graph-restart-select");
  if (restartSelect) {
    restartSelect.addEventListener("change", () => {
      GRAPH_VIEW_STATE.fixture.restart = restartSelect.value;
      fixtureReset();
    });
  }
}

// ---------------------------------------------------------------------------
// Graph summary (private)
// ---------------------------------------------------------------------------

/**
 * Numeric value of a count-or-list field (arrays report their length).
 *
 * @param {*} value
 * @returns {number}
 */
function graphCount(value) {
  if (Array.isArray(value)) return value.length;
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

/**
 * Render the #graph-summary-body panel: nodes, edges and zero-degree counts.
 *
 * @param {Graph|null} graph
 * @returns {void}
 */
function graphRenderSummary(graph) {
  const target = document.getElementById("graph-summary-body");
  if (!target) return;
  if (!graph) {
    target.innerHTML =
      '<p class="empty-state">No graph yet. Press "Build graph" to report nodes, edges, ' +
      "zero-degree users, zero-degree movies and positive training interactions.</p>";
    return;
  }
  const nUsers = graphCount(graph.nUsers);
  const nMovies = graphCount(graph.nMovies);
  const nodeCount = graphCount(graph.nodeCount) || nUsers + nMovies;
  const positiveEdges = graphCount(graph.positiveEdges);
  const zeroUsers = graphCount(graph.zeroDegreeUsers);
  const zeroMovies = graphCount(graph.zeroDegreeMovies);

  target.innerHTML =
    '<dl class="summary-list">' +
    "<dt>User nodes</dt><dd>" + nUsers.toLocaleString("en-US") + "</dd>" +
    "<dt>Movie nodes</dt><dd>" + nMovies.toLocaleString("en-US") + "</dd>" +
    "<dt>Nodes (no padding node)</dt><dd>" + nodeCount.toLocaleString("en-US") + "</dd>" +
    "<dt>Positive training edges</dt><dd>" + positiveEdges.toLocaleString("en-US") + "</dd>" +
    "<dt>Zero-degree users</dt><dd>" + zeroUsers.toLocaleString("en-US") + "</dd>" +
    "<dt>Zero-degree movies</dt><dd>" + zeroMovies.toLocaleString("en-US") + "</dd>" +
    "</dl>";
}

// ---------------------------------------------------------------------------
// Graph view (public: renderGraphView)
// ---------------------------------------------------------------------------

/**
 * Render the graph view into #graph-view, #graph-legend and #graph-controls.
 *
 * Merges `opts` into the module state, builds either the deterministic 4-node
 * fixture view or the capped ego subgraph of the seed user, and draws it with
 * d3 v7 in a two-column bipartite layout. The view is read-only: it never
 * modifies GRAPH or any score array.
 *
 * Degrades gracefully: with no d3, no graph, or no seed user it writes a plain
 * message instead of throwing, and an unimplemented `buildGraph` stub simply
 * leaves GRAPH null (the fixture demo still draws).
 *
 * @param {GraphViewOptions} [opts]
 * @returns {(Object|null)} a small description of the last render, or null.
 */
function renderGraphView(opts) {
  const o = opts || {};
  if (o.mode === "fixture" || o.mode === "ego") GRAPH_VIEW_STATE.mode = o.mode;
  if (o.seedUserId !== undefined) {
    GRAPH_VIEW_STATE.seedUserId = o.seedUserId === null ? null : Number(o.seedUserId);
  }
  if (o.colorBy === "degree" || o.colorBy === "score") GRAPH_VIEW_STATE.colorBy = o.colorBy;
  if (o.maxNodes !== undefined && o.maxNodes !== null && Number.isFinite(Number(o.maxNodes))) {
    GRAPH_VIEW_STATE.maxNodes = Math.max(1, Math.floor(Number(o.maxNodes)));
  }

  const svgEl = document.getElementById("graph-view");
  if (!svgEl) return null;
  if (typeof d3 === "undefined") {
    graphSetLegendMessage("d3 is not available, so the graph view cannot be drawn. Check that vendor/d3.min.js loaded.");
    return null;
  }

  const mode = GRAPH_VIEW_STATE.mode;
  const external = typeof o.getScore === "function" ? o.getScore : null;
  let view;
  let scoreAccessor = null;
  let colorByScore = false;

  if (mode === "fixture") {
    view = graphBuildFixtureView();
    scoreAccessor = (node) => {
      const p = GRAPH_VIEW_STATE.fixture.p;
      return p ? p[node] : 0;
    };
    colorByScore = true;
  } else {
    view = graphBuildEgoView(GRAPH_VIEW_STATE.seedUserId, external);
    if (external && GRAPH_VIEW_STATE.colorBy === "score") {
      scoreAccessor = external;
      colorByScore = true;
    }
  }

  try {
    graphDrawView(svgEl, view, { scoreAccessor: scoreAccessor, colorByScore: colorByScore });
  } catch (error) {
    try {
      d3.select(svgEl).selectAll("*").remove();
    } catch (ignored) {
      /* the SVG may not be an element we can clear */
    }
    graphSetLegendMessage("Graph view could not be drawn: " + graphErrorMessage(error));
  }

  graphRenderLegend(view, { colorByScore: colorByScore });
  graphRenderControls();

  GRAPH_VIEW_STATE.lastRender = {
    mode: mode,
    seedUserId: GRAPH_VIEW_STATE.seedUserId,
    colorByScore: colorByScore,
    nodeCount: view.nodes.length,
    edgeCount: view.edges.length,
    omitted: view.omitted || 0,
    note: view.note || ""
  };
  return GRAPH_VIEW_STATE.lastRender;
}

// ---------------------------------------------------------------------------
// Fixture teaching demo (public: fixtureStep / fixtureReset)
//
// THIS DEMO EXISTS ONLY FOR THE 4-NODE FIXTURE. It is a tiny plain-JavaScript
// propagation so the animation works before you write any stub, and it is NOT a
// template for the MovieLens implementation. The MovieLens propagation must use
// TensorFlow.js tensor operations on R and R-transpose (readme.md section 3.5),
// as required by pageRank / personalizedPageRank.
// ---------------------------------------------------------------------------

/**
 * Update #convergence-results with the fixture iteration and its L1 residual.
 *
 * @returns {void}
 */
function graphRenderConvergence() {
  const target = document.getElementById("convergence-results");
  if (!target) return;
  const fixture = GRAPH_VIEW_STATE.fixture;
  const p = fixture.p || [];
  const values = p
    .map((value, i) => REFERENCE_GRAPH.nodeOrder[i] + " " + Number(value).toFixed(10))
    .join(", ");
  const residual = fixture.residual === null ? "-" : Number(fixture.residual).toExponential(3);

  target.innerHTML =
    "<p><strong>Fixture demo (4-node reference graph)</strong> - iteration " +
    fixture.iteration +
    ", alpha " +
    Number(fixture.alpha !== undefined ? fixture.alpha : graphReadAlpha()).toFixed(2) +
    ", restart " +
    graphEscape(fixture.restart || "uniform") +
    ".</p>" +
    '<dl class="summary-list">' +
    "<dt>L1 residual (last step)</dt><dd>" + graphEscape(residual) + "</dd>" +
    "<dt>Current p</dt><dd>" + graphEscape(values) + "</dd>" +
    "</dl>" +
    '<p class="hint">This plain-JavaScript demo exists only for the 4-node fixture and is not a ' +
    "template for the MovieLens implementation, which must use TensorFlow.js tensor operations " +
    "on R and R-transpose (readme.md section 3.5).</p>";
}

/**
 * Advance the 4-node fixture demo by one propagation step and redraw it.
 *
 * @returns {void}
 */
function fixtureStep() {
  const fixture = GRAPH_VIEW_STATE.fixture;
  const alpha = graphReadAlpha();
  if (!fixture.p) {
    fixture.p = graphFixtureRestartVector(fixture.restart || "uniform");
    fixture.iteration = 0;
  }
  const result = graphFixturePropagate(alpha, fixture.restart || "uniform", fixture.p);
  fixture.p = result.p;
  fixture.iteration += 1;
  fixture.residual = result.residual;
  fixture.alpha = alpha;
  GRAPH_VIEW_STATE.mode = "fixture";
  renderGraphView({ mode: "fixture" });
  graphRenderConvergence();
}

/**
 * Reset the fixture demo to iteration 0 (the restart distribution) and redraw.
 *
 * @returns {void}
 */
function fixtureReset() {
  const fixture = GRAPH_VIEW_STATE.fixture;
  fixture.alpha = graphReadAlpha();
  fixture.iteration = 0;
  fixture.p = graphFixtureRestartVector(fixture.restart || "uniform");
  fixture.residual = null;
  GRAPH_VIEW_STATE.mode = "fixture";
  renderGraphView({ mode: "fixture" });
  graphRenderConvergence();
}

// ---------------------------------------------------------------------------
// Bootstrap (private)
// ---------------------------------------------------------------------------

/**
 * Wire the fixture step/reset buttons and draw the fixture demo once.
 *
 * @returns {void}
 */
function graphInitGraphView() {
  const stepButton = document.getElementById("step-btn");
  if (stepButton) stepButton.addEventListener("click", fixtureStep);
  const resetButton = document.getElementById("reset-steps");
  if (resetButton) resetButton.addEventListener("click", fixtureReset);
  fixtureReset();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", graphInitGraphView);
} else {
  graphInitGraphView();
}

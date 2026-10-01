// ---------------------------------------------------------------------------
// HW3 — Collaborative Filtering core
//
// Missing-value strategy (see week3/readme.md section 6):
//
//   [x] weight similarity by the number of co-rated items
//
// A 0 in ratingMatrix means "not rated", not "hated" (real ratings are 1-5),
// so the raw cosine is computed over co-rated entries only — that is the
// building block of this strategy, not a second one. On its own it fails:
// with one co-rated movie the cosine is exactly 1.0, and 19 of a typical
// user's top-20 neighbours share fewer than 5 movies with them. So every
// similarity is used as sim' = sim * min(common, β) / β (β = SIGNIFICANCE_BETA),
// which discounts similarities backed by little evidence.
// ---------------------------------------------------------------------------

// Neighbourhood sizes for the two predictors
const N_NEIGHBORS = 20;   // User-Based: top-N most similar users who rated the movie
const K_NEIGHBORS = 20;   // Item-Based: top-K most similar movies the user rated

// Significance weighting: similarities need at least β co-rated items for full weight
const SIGNIFICANCE_BETA = 50;

// Reliability threshold for the Top-5 lists (not a missing-value strategy):
// a candidate is recommended only if its prediction rests on at least this
// many neighbours / similar movies. Without it the top is full of movies
// that got a single 5 from a single neighbour and therefore score exactly 5.0.
const MIN_SUPPORT = 3;

// Second reliability threshold for the Top-5 lists (also not a missing-value
// strategy): a candidate movie needs at least this many ratings. Significance
// weighting alone does not remove rare movies: all their neighbours are
// down-weighted by the same factor, and a weighted average does not change
// when every weight is multiplied by the same number.
const MIN_ITEM_RATINGS = 20;

// Initialize the application when the window loads
window.onload = async function() {
    const userBased = document.getElementById('user-based-result');
    const itemBased = document.getElementById('item-based-result');

    try {
        userBased.innerHTML = '<p>Loading movie data...</p>';
        itemBased.innerHTML = '<p>Loading movie data...</p>';

        await loadData();

        populateUserDropdown();
        populateMovieDropdown();

        userBased.innerHTML = '<p>Data loaded. Select a user.</p>';
        itemBased.innerHTML = '<p>Data loaded. Select a user.</p>';
    } catch (error) {
        console.error('Initialization error:', error);
        // The error message is already shown by data.js
    }
};

// Populate the user dropdown with one option per user id found in u.data
function populateUserDropdown() {
    const selectElement = document.getElementById('user-select');

    // Clear existing options except the first placeholder
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }

    for (let userId = 1; userId <= numUsers; userId++) {
        const option = document.createElement('option');
        option.value = userId;
        option.textContent = `User ${userId}`;
        selectElement.appendChild(option);
    }
}

// Populate the movie dropdown with all movies sorted by title.
// u.item contains 18 titles twice under different ids, so duplicates get the id appended.
function populateMovieDropdown() {
    const selectElement = document.getElementById('movie-select');

    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }

    const titleCounts = new Map();
    movies.forEach(movie => titleCounts.set(movie.title, (titleCounts.get(movie.title) || 0) + 1));

    const sortedMovies = [...movies].sort((a, b) => a.title.localeCompare(b.title) || a.id - b.id);
    sortedMovies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = titleCounts.get(movie.title) > 1 ? `${movie.title} [id ${movie.id}]` : movie.title;
        selectElement.appendChild(option);
    });
}

// Look up a movie title by id
let movieTitleById = null;
function getMovieTitle(itemId) {
    if (!movieTitleById) {
        movieTitleById = new Map(movies.map(movie => [movie.id, movie.title]));
    }
    return movieTitleById.get(itemId) || `Movie ${itemId}`;
}

// ---------------------------------------------------------------------------
// Cosine similarity over co-rated entries only.
//
// a, b    — rating vectors of equal length (a matrix row or an itemColumns entry)
// indices — optional list of positions to scan. Only positions rated by BOTH
//           sides matter, so passing the shorter of the two "rated" lists gives
//           the same result as a full scan, just faster.
//
// Returns { sim, common }: sim in [0, 1] (ratings are positive, so it is never
// negative) and the number of co-rated entries. sim is 0 when nothing is co-rated.
//
// Note: the norms are taken over the co-rated entries of each particular pair,
// so they cannot be precomputed once per user/movie; the speed-up comes from
// scanning only the rated positions instead.
// ---------------------------------------------------------------------------
function cosineSimilarity(a, b, indices) {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    let common = 0;

    const length = indices ? indices.length : Math.min(a.length, b.length);
    for (let k = 0; k < length; k++) {
        const idx = indices ? indices[k] : k;
        const x = a[idx];
        const y = b[idx];
        if (x > 0 && y > 0) {
            dot += x * y;
            normA += x * x;
            normB += y * y;
            common++;
        }
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    if (denominator === 0) {
        return { sim: 0, common: 0 };
    }
    return { sim: dot / denominator, common };
}

// Significance weighting of a raw { sim, common } pair: sim * min(common, β) / β
function weightedSimilarity({ sim, common }) {
    return sim * Math.min(common, SIGNIFICANCE_BETA) / SIGNIFICANCE_BETA;
}

// Shorter of two sorted id lists — enough to find every co-rated position
function shorterList(listA, listB) {
    return listA.length <= listB.length ? listA : listB;
}

// Sort by similarity descending, ties by id ascending (keeps results deterministic)
function bySimilarity(a, b) {
    return b.sim - a.sim || a.id - b.id;
}

// User-user similarities for the active user, computed once and reused until the user changes
let neighborCache = { userId: null, neighbors: [] };

function getUserNeighbors(userId) {
    if (neighborCache.userId === userId) {
        return neighborCache.neighbors;
    }

    const t0 = performance.now();
    const activeRow = ratingMatrix[userId];
    const neighbors = [];

    for (let otherId = 1; otherId <= numUsers; otherId++) {
        if (otherId === userId) continue;
        const indices = shorterList(userRatedItems[userId], userRatedItems[otherId]);
        const raw = cosineSimilarity(activeRow, ratingMatrix[otherId], indices);
        const sim = weightedSimilarity(raw);
        if (sim > 0) {
            neighbors.push({ id: otherId, sim, common: raw.common });
        }
    }
    neighbors.sort(bySimilarity);

    neighborCache = { userId, neighbors };
    console.log(`[CF] user-user similarities for user ${userId}: ${neighbors.length} neighbours, ${(performance.now() - t0).toFixed(1)} ms`);
    return neighbors;
}

// Item-item similarities (raw { sim, common }), computed on demand for the needed pairs only.
//
// The cache is cleared when the active user changes, like neighborCache. A size
// cap was rejected: one Item-Based Top-5 for a heavy user (e.g. user 405) needs
// ~340k pairs, so a 300k cap would be flushed in the middle of a single click and
// the pairs recomputed. Per-user reset keeps memory bounded by one user's needs
// and still makes repeated clicks for the same user fast.
const itemSimilarityCache = new Map();
let itemCacheUserId = null;

function resetItemCacheFor(userId) {
    if (itemCacheUserId !== userId) {
        itemSimilarityCache.clear();
        itemCacheUserId = userId;
    }
}

function getItemSimilarity(itemA, itemB) {
    const low = Math.min(itemA, itemB);
    const high = Math.max(itemA, itemB);
    const key = low * (numMovies + 1) + high;

    let entry = itemSimilarityCache.get(key);
    if (entry === undefined) {
        const indices = shorterList(itemRaters[low], itemRaters[high]);
        entry = cosineSimilarity(itemColumns[low], itemColumns[high], indices);
        itemSimilarityCache.set(key, entry);
    }
    return entry;
}

// Validate ids shared by both predictors; returns a reason string or null
function checkPredictionInput(userId, itemId) {
    if (!(userId >= 1 && userId <= numUsers)) return `Unknown user ${userId}.`;
    if (!(itemId >= 1 && itemId <= numMovies)) return `Unknown movie ${itemId}.`;
    if (itemRaters[itemId].length === 0) return 'Nobody has rated this movie yet.';
    return null;
}

// Mean number of co-rated items over the neighbours actually used
function averageCommon(neighbors) {
    return neighbors.reduce((sum, neighbor) => sum + neighbor.common, 0) / neighbors.length;
}

// Similarity-weighted average of ratings: sum(sim * r) / sum(sim)
function weightedAverage(neighbors, ratingOf) {
    let numerator = 0;
    let denominator = 0;
    for (const neighbor of neighbors) {
        numerator += neighbor.sim * ratingOf(neighbor.id);
        denominator += neighbor.sim;
    }
    return numerator / denominator;
}

// ---------------------------------------------------------------------------
// User-Based CF: predict userId's rating of itemId from the top-N most similar
// users (weighted sim > 0) who rated itemId.
//
// Returns { score, support, avgCommon, reason }: score is null (with a reason)
// when no similar user has rated the movie; support = number of neighbours
// used; avgCommon = their mean number of co-rated movies with userId.
// ---------------------------------------------------------------------------
function predictUserBased(userId, itemId, N = N_NEIGHBORS) {
    const inputProblem = checkPredictionInput(userId, itemId);
    if (inputProblem) {
        return { score: null, support: 0, avgCommon: 0, reason: inputProblem };
    }

    // getUserNeighbors is already sorted by similarity, so the first N raters are the top-N
    const used = [];
    for (const neighbor of getUserNeighbors(userId)) {
        if (ratingMatrix[neighbor.id][itemId] > 0) {
            used.push(neighbor);
            if (used.length === N) break;
        }
    }

    if (used.length === 0) {
        return { score: null, support: 0, avgCommon: 0, reason: `None of the users similar to User ${userId} has rated this movie.` };
    }

    const score = weightedAverage(used, neighborId => ratingMatrix[neighborId][itemId]);
    return { score, support: used.length, avgCommon: averageCommon(used), reason: null };
}

// ---------------------------------------------------------------------------
// Item-Based CF: predict userId's rating of itemId from the top-K movies the
// user has rated that are most similar (weighted sim > 0) to itemId.
//
// Returns { score, support, avgCommon, reason }: score is null (with a reason)
// when none of the user's movies is similar to itemId; support = number of
// movies used; avgCommon = their mean number of co-raters with itemId.
// ---------------------------------------------------------------------------
function predictItemBased(userId, itemId, K = K_NEIGHBORS) {
    const inputProblem = checkPredictionInput(userId, itemId);
    if (inputProblem) {
        return { score: null, support: 0, avgCommon: 0, reason: inputProblem };
    }

    resetItemCacheFor(userId);
    const similar = [];
    for (const ratedId of userRatedItems[userId]) {
        if (ratedId === itemId) continue;
        const raw = getItemSimilarity(itemId, ratedId);
        const sim = weightedSimilarity(raw);
        if (sim > 0) {
            similar.push({ id: ratedId, sim, common: raw.common });
        }
    }

    if (similar.length === 0) {
        return { score: null, support: 0, avgCommon: 0, reason: `None of the movies User ${userId} has rated shares a rater with this movie.` };
    }

    similar.sort(bySimilarity);
    const used = similar.slice(0, K);
    const score = weightedAverage(used, ratedId => ratingMatrix[userId][ratedId]);
    return { score, support: used.length, avgCommon: averageCommon(used), reason: null };
}

// Run a predictor over every sufficiently rated movie the user has not rated
// and keep the reliable top-K
function collectRecommendations(userId, topK, predict) {
    const candidates = [];
    for (let itemId = 1; itemId <= numMovies; itemId++) {
        if (ratingMatrix[userId][itemId] > 0) continue;
        if (itemRaters[itemId].length < MIN_ITEM_RATINGS) continue;

        const prediction = predict(userId, itemId);
        if (prediction.score === null || prediction.support < MIN_SUPPORT) continue;

        candidates.push({ id: itemId, title: getMovieTitle(itemId), score: prediction.score, support: prediction.support });
    }

    // Score descending, compared at 1e-6 so float noise (5.0 vs 4.999999999999999)
    // counts as a tie; ties broken by stronger support, then by id. Scores stay unrounded.
    candidates.sort((a, b) =>
        Math.round(b.score * 1e6) - Math.round(a.score * 1e6) || b.support - a.support || a.id - b.id);
    return candidates.slice(0, topK);
}

// User-Based Top-K: array of { title, score, support }
function getUserBasedRecommendations(activeUserId, topK = 5) {
    const t0 = performance.now();
    const result = collectRecommendations(activeUserId, topK, predictUserBased);
    console.log(`[CF] User-Based Top-${topK} for user ${activeUserId}: ${(performance.now() - t0).toFixed(1)} ms`);
    return result.map(({ title, score, support }) => ({ title, score, support }));
}

// Item-Based Top-K: array of { title, score, support }
function getItemBasedRecommendations(activeUserId, topK = 5) {
    const t0 = performance.now();
    const result = collectRecommendations(activeUserId, topK, predictItemBased);
    console.log(`[CF] Item-Based Top-${topK} for user ${activeUserId}: ${(performance.now() - t0).toFixed(1)} ms (item-item cache: ${itemSimilarityCache.size} pairs for user ${itemCacheUserId})`);
    return result.map(({ title, score, support }) => ({ title, score, support }));
}

// Escape text before inserting it into innerHTML (titles contain "&", quotes, etc.)
function escapeHtml(text) {
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

// Read the selected user id from the shared dropdown (NaN if none selected)
function getSelectedUserId() {
    return parseInt(document.getElementById('user-select').value, 10);
}

// Title of one of the user's highest-rated movies, for the Item-Based explanation
function getFavoriteMovieTitle(userId) {
    let bestId = null;
    for (const itemId of userRatedItems[userId]) {
        if (bestId === null || ratingMatrix[userId][itemId] > ratingMatrix[userId][bestId]) {
            bestId = itemId;
        }
    }
    return bestId === null ? null : getMovieTitle(bestId);
}

// Read the selected user and render both recommendation lists
function getRecommendations() {
    const userId = getSelectedUserId();

    if (isNaN(userId)) {
        renderList('user-based-result', [], 'Please select a user first.');
        renderList('item-based-result', [], 'Please select a user first.');
        return;
    }

    renderList('user-based-result', [], 'Calculating recommendations...');
    renderList('item-based-result', [], 'Calculating recommendations...');

    // Let the browser repaint the status message before the heavy computation
    setTimeout(() => {
        const favorite = getFavoriteMovieTitle(userId);
        renderList('user-based-result', getUserBasedRecommendations(userId),
            null, 'Because you are similar to other users, we recommend:');
        renderList('item-based-result', getItemBasedRecommendations(userId),
            null, `Because you liked ${favorite ? `"${favorite}"` : 'your movies'} and similar titles, we recommend:`);
    }, 50);
}

// Render a list of { title, score, support } into the given element
function renderList(elementId, items, message, intro) {
    const el = document.getElementById(elementId);

    if (message) {
        el.innerHTML = `<p>${escapeHtml(message)}</p>`;
        return;
    }

    if (!items || items.length === 0) {
        el.innerHTML = `<p>No reliable recommendations: no unrated movie with at least ${MIN_ITEM_RATINGS} ratings has a prediction backed by at least ${MIN_SUPPORT} neighbours.</p>`;
        return;
    }

    const entries = items
        .map(item => `<li><span class="rec-title">${escapeHtml(item.title)}</span>` +
            `<span class="rec-meta">score ${Number(item.score).toFixed(3)} &middot; support ${item.support}</span></li>`)
        .join('');
    el.innerHTML = `<p class="intro">${escapeHtml(intro || '')}</p><ol>${entries}</ol>`;
}

// Read the selected user and movie and render both predicted ratings
function predictRating() {
    const userId = getSelectedUserId();
    const itemId = parseInt(document.getElementById('movie-select').value, 10);

    if (isNaN(userId) || isNaN(itemId)) {
        const message = '<p>Please select a user and a movie first.</p>';
        document.getElementById('user-based-prediction').innerHTML = message;
        document.getElementById('item-based-prediction').innerHTML = message;
        return;
    }

    let t0 = performance.now();
    const userBased = predictUserBased(userId, itemId);
    console.log(`[CF] User-Based prediction (user ${userId}, movie ${itemId}): ${(performance.now() - t0).toFixed(1)} ms`);

    t0 = performance.now();
    const itemBased = predictItemBased(userId, itemId);
    console.log(`[CF] Item-Based prediction (user ${userId}, movie ${itemId}): ${(performance.now() - t0).toFixed(1)} ms`);

    const actual = ratingMatrix[userId][itemId];
    const movieRatings = itemRaters[itemId].length;
    renderPrediction('user-based-prediction', userBased, actual, userId, movieRatings, 'similar users', 'common movies');
    renderPrediction('item-based-prediction', itemBased, actual, userId, movieRatings, 'similar movies', 'common raters');
}

// Render one prediction card: big number, or a message when the prediction is null
function renderPrediction(elementId, prediction, actual, userId, movieRatings, supportLabel, commonLabel) {
    const el = document.getElementById(elementId);
    const actualLine = actual > 0 ? `<p class="actual">Actual rating: ${actual}</p>` : '';

    if (prediction.score === null) {
        el.innerHTML =
            `<p class="no-prediction">Cannot predict: ${escapeHtml(prediction.reason)}</p>` +
            `<p class="support">For reference, User ${userId}'s average rating is ${userMean[userId].toFixed(1)}.</p>` +
            actualLine;
        return;
    }

    const warnings = [];
    if (prediction.support < MIN_SUPPORT) warnings.push('low support');
    if (movieRatings < MIN_ITEM_RATINGS) warnings.push(`few ratings (${movieRatings})`);
    const warningText = warnings.length > 0 ? ` <span class="warning">(${warnings.join(', ')})</span>` : '';
    el.innerHTML =
        `<div class="big-score">${prediction.score.toFixed(1)}</div>` +
        `<p class="label">predicted</p>` +
        `<p class="support">based on ${prediction.support} ${supportLabel}, ` +
        `avg ${Math.round(prediction.avgCommon)} ${commonLabel}${warningText}</p>` +
        actualLine;
}

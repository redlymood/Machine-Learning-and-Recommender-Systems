// Global variables for storing movie and rating data
let movies = [];
let ratings = [];

// Collaborative filtering structures (populated by buildRatingMatrix)
let numUsers = 0;          // highest user id found in u.data
let numMovies = 0;         // number of parsed movies
let ratingMatrix = null;   // (numUsers + 1) x (numMovies + 1); 0 = "not rated"

// Helper structures for fast sparse access (also populated by buildRatingMatrix)
let itemColumns = [];      // transposed matrix: itemColumns[itemId][userId] === rating
let userRatedItems = [];   // userRatedItems[userId] = ids of movies the user rated (ascending)
let itemRaters = [];       // itemRaters[itemId] = ids of users who rated the movie (ascending)
let userMean = null;       // userMean[userId] = the user's average rating (0 if none)

// Genre names as defined in the u.item file
const genreNames = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

// Primary function to load data from files
async function loadData() {
    try {
        // Load and parse movie data
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        // u.item is Latin-1 encoded (e.g. byte 0xE9 = "é"); response.text() would
        // decode it as UTF-8 and turn such titles into "Mis�rables".
        const moviesBuffer = await moviesResponse.arrayBuffer();
        const moviesText = new TextDecoder('latin1').decode(moviesBuffer);
        parseItemData(moviesText);

        // Load and parse rating data
        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
        }
        const ratingsText = await ratingsResponse.text();
        parseRatingData(ratingsText);

        // Derive matrix dimensions, then build the rating matrix
        numUsers = ratings.reduce((max, r) => Math.max(max, r.userId), 0);
        numMovies = movies.length;
        buildRatingMatrix();
    } catch (error) {
        console.error('Error loading data:', error);
        const errorTarget = document.getElementById('user-based-result');
        if (errorTarget) {
            errorTarget.innerHTML = `<p class="error">Error: ${error.message}. Please make sure u.item and u.data are in the correct location.</p>`;
        }
        throw error; // Re-throw so script.js can handle the error
    }
}

// Parse movie data from u.item format
function parseItemData(text) {
    const lines = text.split('\n');

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('|');
        if (fields.length < 5) continue; // Skip invalid lines

        const id = parseInt(fields[0]);
        const title = fields[1];

        // Extract genres (last 19 fields)
        const genreValues = fields.slice(5, 24).map(value => parseInt(value));
        const genres = genreNames.filter((_, index) => genreValues[index] === 1);

        movies.push({ id, title, genres });
    }
}

// Parse rating data from u.data format
function parseRatingData(text) {
    const lines = text.split('\n');

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('\t');
        if (fields.length < 4) continue; // Skip invalid lines

        const userId = parseInt(fields[0]);
        const itemId = parseInt(fields[1]);
        const rating = parseFloat(fields[2]);
        const timestamp = parseInt(fields[3]);

        ratings.push({ userId, itemId, rating, timestamp });
    }
}

// Build the user-item rating matrix plus the helper structures declared above.
//
// Shape: (numUsers + 1) x (numMovies + 1), indexed by raw id, so that
//   ratingMatrix[userId][movieId] === rating
// Row 0 and column 0 are unused. A missing entry is 0: MovieLens ratings are
// 1-5, so 0 unambiguously means "not rated" and no separate mask is needed.
function buildRatingMatrix() {
    ratingMatrix = [];
    itemColumns = [];
    userRatedItems = [];
    itemRaters = [];
    userMean = new Float64Array(numUsers + 1);

    for (let u = 0; u <= numUsers; u++) {
        ratingMatrix.push(new Float32Array(numMovies + 1));
        userRatedItems.push([]);
    }
    for (let i = 0; i <= numMovies; i++) {
        itemColumns.push(new Float32Array(numUsers + 1));
        itemRaters.push([]);
    }

    for (const { userId, itemId, rating } of ratings) {
        if (userId < 1 || userId > numUsers || itemId < 1 || itemId > numMovies) continue;
        ratingMatrix[userId][itemId] = rating;
        itemColumns[itemId][userId] = rating;
        userRatedItems[userId].push(itemId);
        itemRaters[itemId].push(userId);
    }

    for (let u = 1; u <= numUsers; u++) {
        const items = userRatedItems[u].sort((a, b) => a - b);
        let sum = 0;
        for (const i of items) sum += ratingMatrix[u][i];
        userMean[u] = items.length > 0 ? sum / items.length : 0;
    }
    for (let i = 1; i <= numMovies; i++) {
        itemRaters[i].sort((a, b) => a - b);
    }
}

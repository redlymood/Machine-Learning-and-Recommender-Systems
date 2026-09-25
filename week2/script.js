// Initialize the application when the window loads
window.onload = async function() {
    try {
        // Display loading message
        const resultElement = document.getElementById('result');
        resultElement.textContent = "Loading movie data...";
        resultElement.className = 'loading';
        
        // Load data
        await loadData();
        
        // Populate dropdown and update status
        populateMoviesDropdown();
        resultElement.textContent = "Data loaded. Please select a movie.";
        resultElement.className = 'success';
    } catch (error) {
        console.error('Initialization error:', error);
        // Error message already set in data.js
    }
};

// Populate every movie dropdown with sorted movie titles
function populateMoviesDropdown() {
    // Sort movies alphabetically by title (done once, reused for all dropdowns)
    const sortedMovies = [...movies].sort((a, b) => a.title.localeCompare(b.title));

    // Single-movie dropdown + the 3 profile dropdowns
    const selectIds = ['movie-select', 'profile-select-1', 'profile-select-2', 'profile-select-3'];

    selectIds.forEach(id => {
        const selectElement = document.getElementById(id);
        if (!selectElement) return; // Skip if the block is not present

        // Clear existing options except the first placeholder
        while (selectElement.options.length > 1) {
            selectElement.remove(1);
        }

        // Add movies to this dropdown
        sortedMovies.forEach(movie => {
            const option = document.createElement('option');
            option.value = movie.id;
            option.textContent = movie.title;
            selectElement.appendChild(option);
        });
    });
}

// Cosine similarity between two equal-length numeric vectors.
// Returns dotProduct / (magnitudeA * magnitudeB).
// If either vector has magnitude 0 (all zeros), returns 0 to avoid divide-by-zero.
function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    let sumSqA = 0;
    let sumSqB = 0;

    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        sumSqA += vecA[i] * vecA[i];
        sumSqB += vecB[i] * vecB[i];
    }

    const magnitudeA = Math.sqrt(sumSqA);
    const magnitudeB = Math.sqrt(sumSqB);

    if (magnitudeA === 0 || magnitudeB === 0) {
        return 0;
    }

    return dotProduct / (magnitudeA * magnitudeB);
}

// Main recommendation function
function getRecommendations() {
    const resultElement = document.getElementById('result');
    
    try {
        // Step 1: Get user input
        const selectElement = document.getElementById('movie-select');
        const selectedMovieId = parseInt(selectElement.value);
        
        if (isNaN(selectedMovieId)) {
            resultElement.textContent = "Please select a movie first.";
            resultElement.className = 'error';
            return;
        }
        
        // Step 2: Find the liked movie
        const likedMovie = movies.find(movie => movie.id === selectedMovieId);
        if (!likedMovie) {
            resultElement.textContent = "Error: Selected movie not found in database.";
            resultElement.className = 'error';
            return;
        }
        
        // Show loading message while processing
        resultElement.textContent = "Calculating recommendations...";
        resultElement.className = 'loading';
        
        // Use setTimeout to allow the UI to update before heavy computation
        setTimeout(() => {
            try {
                // Step 3: Prepare for similarity calculation
                const likedVector = likedMovie.genreVector;
                const candidateMovies = movies.filter(movie => movie.id !== likedMovie.id);

                // Step 4: Calculate cosine similarity scores
                const scoredMovies = candidateMovies.map(candidate => {
                    const score = cosineSimilarity(likedVector, candidate.genreVector);

                    return {
                        ...candidate,
                        score: score
                    };
                });
                
                // Step 5: Sort by score in descending order
                scoredMovies.sort((a, b) => b.score - a.score);
                
                // Step 6: Select top recommendations
                const topRecommendations = scoredMovies.slice(0, 5);
                
                // Step 7: Display results
                if (topRecommendations.length > 0) {
                    const recommendationTitles = topRecommendations.map(movie => movie.title);
                    resultElement.textContent = `Because you liked "${likedMovie.title}", we recommend: ${recommendationTitles.join(', ')}`;
                    resultElement.className = 'success';
                } else {
                    resultElement.textContent = `No recommendations found for "${likedMovie.title}".`;
                    resultElement.className = 'error';
                }
            } catch (error) {
                console.error('Error in recommendation calculation:', error);
                resultElement.textContent = "An error occurred while calculating recommendations.";
                resultElement.className = 'error';
            }
        }, 100);
    } catch (error) {
        console.error('Error in getRecommendations:', error);
        resultElement.textContent = "An unexpected error occurred.";
        resultElement.className = 'error';
    }
}

// Profile recommendation function:
// averages 3 watched movies into one profile vector, then recommends top-5.
function getProfileRecommendations() {
    const resultElement = document.getElementById('result');

    try {
        // Step 1: Get the 3 selected movie ids
        const ids = [
            parseInt(document.getElementById('profile-select-1').value),
            parseInt(document.getElementById('profile-select-2').value),
            parseInt(document.getElementById('profile-select-3').value)
        ];

        if (ids.some(id => isNaN(id))) {
            resultElement.textContent = "Please select all three movies first.";
            resultElement.className = 'error';
            return;
        }

        // Require 3 different movies (averaging a duplicate would skew the profile)
        if (new Set(ids).size !== 3) {
            resultElement.textContent = "Please select three different movies.";
            resultElement.className = 'error';
            return;
        }

        // Step 2: Find the watched movie objects
        const watchedMovies = ids.map(id => movies.find(movie => movie.id === id));
        if (watchedMovies.some(movie => !movie)) {
            resultElement.textContent = "Error: One of the selected movies was not found in database.";
            resultElement.className = 'error';
            return;
        }

        // Show loading message while processing
        resultElement.textContent = "Calculating recommendations...";
        resultElement.className = 'loading';

        // Use setTimeout to allow the UI to update before heavy computation
        setTimeout(() => {
            try {
                // Step 3: Build the profile vector = element-wise average of the 3 genre vectors
                const vectorLength = watchedMovies[0].genreVector.length;
                const profileVector = [];
                for (let i = 0; i < vectorLength; i++) {
                    const sum = watchedMovies.reduce((acc, movie) => acc + movie.genreVector[i], 0);
                    profileVector.push(sum / watchedMovies.length);
                }

                // Step 4: Score every movie (excluding the 3 watched) with cosine similarity
                const watchedIds = new Set(ids);
                const candidateMovies = movies.filter(movie => !watchedIds.has(movie.id));
                const scoredMovies = candidateMovies.map(candidate => {
                    return {
                        ...candidate,
                        score: cosineSimilarity(profileVector, candidate.genreVector)
                    };
                });

                // Step 5: Sort by score in descending order
                scoredMovies.sort((a, b) => b.score - a.score);

                // Step 6: Select top 5 recommendations
                const topRecommendations = scoredMovies.slice(0, 5);

                // Step 7: Display results
                if (topRecommendations.length > 0) {
                    const watchedTitles = watchedMovies.map(movie => movie.title).join(', ');
                    const recommendationTitles = topRecommendations.map(movie => movie.title);
                    resultElement.textContent = `Because you watched ${watchedTitles}, we recommend: ${recommendationTitles.join(', ')}`;
                    resultElement.className = 'success';
                } else {
                    resultElement.textContent = "No recommendations found for your profile.";
                    resultElement.className = 'error';
                }
            } catch (error) {
                console.error('Error in profile recommendation calculation:', error);
                resultElement.textContent = "An error occurred while calculating recommendations.";
                resultElement.className = 'error';
            }
        }, 100);
    } catch (error) {
        console.error('Error in getProfileRecommendations:', error);
        resultElement.textContent = "An unexpected error occurred.";
        resultElement.className = 'error';
    }
}

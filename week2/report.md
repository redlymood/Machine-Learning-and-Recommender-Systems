# A Content-Based Movie Recommender Using Cosine Similarity on Genre Vectors

**Student:** Polina Elatontseva
**Team:** Individual
**Email:** redlymood@gmail.com
**Date:** 2026-09-25
**Assignment:** A02 Content-Based Filtering (Week 2)

---

## Abstract

This report describes a content-based movie recommender built with the MovieLens 100k dataset. Each movie is turned into a 19-dimensional genre vector, and similarity is measured with cosine similarity. The system works in two modes: item-to-item (one seed movie) and profile mode (the mean vector of three watched movies), and both return a top-5 list. I verified the math by hand: Toy Story and Aladdin give a cosine of 0.8660, which is equal to the code result and to √3/2. The experiments also show how cosine reduces the bias of popular multi-genre movies compared to a raw dot product.

**Index Terms** — content-based filtering, cosine similarity, vector space model, recommender systems, MovieLens.

---

## 1 Introduction

**Problem.** People have too many movies to choose from. A recommender system helps a user find movies that are similar to what they like.

**Motivation.** A content-based method only needs the movie features (here, the genres). It does not need data from other users. This is useful when we do not have many ratings, and it is easy to explain to the user.

**Concrete example.** If a user likes *Toy Story* (Animation, Children's, Comedy), the system should suggest other Animation/Children's/Comedy movies, such as *Aladdin*. In our results, *Aladdin and the King of Thieves* got a similarity of 1.0000 with *Toy Story*.

**Contributions.** This work makes three contributions:
- It builds a working recommender with two modes: item-to-item and a user profile made from three movies.
- It compares cosine similarity with a raw dot product and shows the effect on popular movies.
- It checks the main number by hand and by code, so the result is verified.

---

## 2 Related Work and Analysis

**Related work.** The MovieLens dataset used here is described by Harper and Konstan [1]. The idea of representing items as vectors and comparing them by angle comes from the vector space model of Salton, Wong, and Yang [2]. Content-based recommendation is surveyed by Pazzani and Billsus [3] and by Lops, de Gemmis, and Semeraro [4]. The use of cosine similarity for recommendation on MovieLens data is shown by Sarwar, Karypis, Konstan, and Riedl [5].

**Item-to-item vs profile.** I used *Toy Story* as the seed for item-to-item mode, and *Toy Story + Aladdin + Lion King* for profile mode. The two top-5 lists are shown side by side in Table I. The two lists have **0 movies in common**.

**Table I — Top-5: item-to-item vs profile**

| # | Item-to-item (Toy Story) | Score | Profile (Toy Story + Aladdin + Lion King) | Score |
|---|---|---|---|---|
| 1 | Aladdin and the King of Thieves (1996) | 1.0000 | Cats Don't Dance (1997) | 0.9058 |
| 2 | Goofy Movie, A (1995) | 0.8660 | Three Caballeros, The (1945) | 0.9058 |
| 3 | Aladdin (1992) | 0.8660 | Hunchback of Notre Dame, The (1996) | 0.9058 |
| 4 | Gumby: The Movie (1995) | 0.8165 | Beauty and the Beast (1991) | 0.9058 |
| 5 | Land Before Time III (1995) | 0.8165 | Anastasia (1997) | 0.9058 |

**Bias mitigation (blockbusters).** A raw dot product gives a high score to movies with many genres, because they have more chances to share a genre. I tested this with the query *Toy Story* and two 5-genre movies: *Space Jam* and *Hercules*. Table II shows the result.

**Table II — Blockbuster rank: dot product vs cosine (query: Toy Story)**

| Movie (5 genres) | Dot value | Dot rank | Cosine value | Cosine rank |
|---|---|---|---|---|
| Space Jam (1996) | 3 | #3 | 0.7746 | #38 |
| Hercules (1997) | 3 | #4 | 0.7746 | #39 |

With the dot product, these big movies are near the top (#3 and #4). With cosine, they drop far down (#38 and #39). The average number of genres in the dot-product top-5 is 4.2, but in the cosine top-5 it is 3.0. So cosine reduces the bias toward multi-genre movies.

**Catalog discovery (long-tail).** Using `u.data`, I counted how many ratings each movie has. The catalog has 1682 rated movies, and the median is 27 ratings per movie. I call a movie "long-tail" if it has 27 ratings or fewer. Table III shows the rating count for each recommended movie.

**Table III — Popularity of the recommendations**

| Item-to-item (Toy Story) | n_ratings | Profile (Toy+Aladdin+Lion King) | n_ratings |
|---|---|---|---|
| Aladdin and the King of Thieves | 26 (long-tail) | Cats Don't Dance | 32 |
| Goofy Movie, A | 20 (long-tail) | Three Caballeros, The | 22 (long-tail) |
| Aladdin (1992) | 219 | Hunchback of Notre Dame | 127 |
| Gumby: The Movie | 5 (long-tail) | Beauty and the Beast | 202 |
| Land Before Time III | 6 (long-tail) | Anastasia | 66 |

Item-to-item mode returns **4/5** long-tail movies, while profile mode returns **1/5**. So item-to-item mode surfaces rarer movies.

---

## 3 Method

**Genre vector.** Each movie is a 19-dimensional vector of 0/1 genre flags, read from columns 5–23 of `u.item`. For example, *Toy Story* is `[0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]`, which means Animation, Children's, Comedy.

**Off-by-one fix.** The first genre column in `u.item` is "unknown", so there are 19 genre columns, not 18. The original code used a list of only 18 genre names, so the names did not line up with the numbers. I added "unknown" to the front of the genre-name list (now 19 names). After the fix, *Toy Story* correctly shows Animation, Children's, Comedy.

**Cosine similarity.** For two vectors A and B:

cosine(A, B) = (A · B) / (||A|| × ||B||)

where A · B is the dot product, and ||A|| and ||B|| are the vector lengths (magnitudes). If a magnitude is 0, the function returns 0, so there is no division by zero.

**Profile vector.** In profile mode, I take the three genre vectors of the three watched movies and compute the mean for each of the 19 positions. This mean vector is the user profile. For example, for *Toy Story + Aladdin + Lion King* the profile has Animation = 1.0, Children's = 1.0, Comedy = 0.667, Musical = 0.667.

**Recommendation.** I score every movie with cosine similarity against the seed (or the profile), remove the input movie(s), sort by score, and return the top-5.

**Tools.** The app is written in vanilla JavaScript (HTML, CSS, JS). Node.js was not installed on the machine, so I ran the real `cosineSimilarity` function with Apple's JavaScriptCore (`osascript -l JavaScript`), and I reproduced all the batch numbers with Python.

---

## 4 Experiments and Verification

I verified the main similarity number by hand and by code.

**Vectors.**
- Toy Story (id 1): `[0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]`
- Aladdin (id 95): `[0,0,0,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0]`

**Hand calculation.**
- Dot product = (1×1) + (1×1) + (1×1) = 3 (the Musical position is 0×1 = 0).
- ||A|| = √(1+1+1) = √3 ≈ 1.7320508
- ||B|| = √(1+1+1+1) = √4 = 2
- cosine = 3 / (√3 × 2) = 3 / 3.4641016 = 0.8660254

**Code result (JavaScriptCore).** The same two vectors through the real `cosineSimilarity` function gave 0.8660254037844387.

**Result.** Hand value = 0.8660, code value = 0.8660. They are **equal**, and both equal √3/2 (the cosine of a 30° angle).

---

## 5 Discussion

**Failure case.** In the beginning my recommender showed wrong genres. For "Toy Story" it printed Children's, Comedy, Crime, but the real genres are Animation, Children's, Comedy. Every movie was shifted by one genre.

**Root cause.** The file u.item has 19 genre columns, and the first column is "unknown". But my genreNames list had only 18 names and started from "Action". The code did slice(5, 24) (19 numbers) but matched them with 18 names from index 0, so every name moved one place. This is an off-by-one bug, not a math bug.

**Fix + verification.** I added "unknown" to the front of genreNames, so now 19 names line up 1:1 with 19 numbers. After the fix I printed movie id 1 and it showed Animation, Children's, Comedy — correct. I checked it by eyes against the raw line in u.item.

**What worked.** Cosine normalization really stopped the big multi-genre movies. For the query "Toy Story", "Space Jam" and "Hercules" (5 genres each) were #3 and #4 with raw dot product, but with cosine they fell to #38 and #39. Also my hand cosine for Toy Story vs Aladdin = 0.8660 was exactly equal to the code (√3/2).

**What surprised me.** I thought the profile mode (average of 3 movies) will discover more rare films. But it was opposite: single-movie gave 4/5 long-tail films, profile gave only 1/5. The average makes a "wide middle taste" and this matches big popular animated musicals. So for catalog discovery, here the single item was better.

**Next improvement.** With more time I would give rare genres more weight (like IDF), or use the ratings in u.data to weight the profile. This maybe push more long-tail movies into the profile recommendations.

---

## 6 AI Usage Disclosure

I used an AI coding assistant (Claude Code) to help read the starter code, write the JavaScript functions, and run the batch experiments in Python and JavaScriptCore.

What I verified myself:
- I checked the five references and their DOIs on Crossref.
- I re-did the hand cosine calculation for *Toy Story* vs *Aladdin* (= 0.8660 = √3/2) and confirmed it matches the code output.

---

## References

[1] F. M. Harper and J. A. Konstan, "The MovieLens Datasets: History and Context," *ACM Trans. Interactive Intelligent Systems*, vol. 5, no. 4, 2015. doi:10.1145/2827872

[2] G. Salton, A. Wong, and C. S. Yang, "A Vector Space Model for Automatic Indexing," *Communications of the ACM*, vol. 18, no. 11, 1975. doi:10.1145/361219.361220

[3] M. J. Pazzani and D. Billsus, "Content-Based Recommendation Systems," in *The Adaptive Web*, LNCS 4321, Springer, 2007. doi:10.1007/978-3-540-72079-9_10

[4] P. Lops, M. de Gemmis, and G. Semeraro, "Content-based Recommender Systems: State of the Art and Trends," in *Recommender Systems Handbook*, Springer, 2011. doi:10.1007/978-0-387-85820-3_3

[5] B. Sarwar, G. Karypis, J. Konstan, and J. Riedl, "Item-Based Collaborative Filtering Recommendation Algorithms," in *Proc. WWW '01*, 2001. doi:10.1145/371920.372071

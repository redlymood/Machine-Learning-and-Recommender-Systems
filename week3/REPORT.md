# User-Based and Item-Based Collaborative Filtering for Movie Rating Prediction

**Student:** Polina Elatontseva **Team:** Individual **Email:** redlymood@gmail.com **Date:** 2026-10-01 **Assignment:** A03 Collaborative Filtering (Week 3)

## Abstract

This report describes a movie recommender with user-based and item-based collaborative filtering (CF) on the MovieLens 100k dataset (943 users, 1682 movies, 100,000 ratings, 93.7% sparsity). Similarity is cosine on co-rated items with significance weighting (β = 50), and a prediction is a weighted average of the top 20 neighbours. On a per-user 80/20 split, user-based CF has RMSE 1.019 and item-based CF has RMSE 1.057, while matrix factorization reaches 0.929. With plain co-rated cosine, 19 of the top-20 neighbours of a typical user have fewer than 5 common movies; with weighting this number is 0. A popularity threshold of 20 ratings raises Precision@5 of user-based CF from 0.060 to 0.104. The browser results are equal to a numpy reference script in all digits.

**Index Terms** — collaborative filtering, cosine similarity, significance weighting, matrix factorization, cold start, MovieLens.

## 1 Introduction

**Problem.** A movie site has many users and many movies, but each user rates only a few movies. We want to guess the missing ratings and recommend good movies. In this homework I built a movie recommender with collaborative filtering (CF), which uses only ratings, not genres. The page predict a rating for a user and a movie and shows two Top-5 lists.

**Motivation.** CF does not need genres or descriptions. It finds people with similar taste (user-based) or movies with similar fans (item-based). But the rating matrix is very sparse, so similarities can be wrong when two users have only one or two movies in common.

**Concrete example.** User 1 gave "Toy Story (1995)" a 5. The page predicts 4.0 with user-based CF (20 similar users, on average 93 common movies) and 4.2 with item-based CF (20 similar movies, on average 171 common raters).

**Contributions.**

- A static web app with two CF methods: a rating prediction for one user and one movie, and two Top-5 lists.
- An offline comparison of four missing-value strategies, plus experiments on cold start, sparsity and stability of similarities.
- A check that the browser code and a numpy reference give the same numbers, and one hand calculation.

## 2 Related Work and Analysis

**Related work.** The MovieLens datasets are described by Harper and Konstan [1]. Herlocker et al. [2] describe the neighbourhood framework for CF and the significance weighting that I use (a similarity with fewer than 50 common items gets a smaller weight). Item-based CF was proposed by Sarwar et al. [3], and Linden, Smith and York [4] show how Amazon uses item-to-item CF at large scale. Matrix factorization is explained by Koren, Bell and Volinsky [5]. Schein et al. [6] study the cold-start problem.

### 2.1 User-based vs item-based

The choice depend from the shape of the data. A user-user similarity uses the movies both users rated, and an item-item similarity uses the users who rated both movies. If users ≫ items, every item has many ratings. Then item-item similarities are cheap (few item pairs) and stable (a lot of data). They change slowly, so we can compute them in advance. This is why big shops like Amazon use item-based CF [4]. If items ≫ users, it is the opposite: user-based CF is cheaper and more stable.

Our case is the second one: 1682 movies and 943 users. Movies have less ratings than users: 47.6 train ratings per movie and 84.8 per user on average.

**Table I — User-based vs item-based (MovieLens 100k, train split)**

| | User-Based | Item-Based |
|---|---|---|
| Entities | 943 users | 1682 movies |
| Pairs (n²) | 889,249 | 2,829,124 |
| Mean train ratings per entity | 84.8 | 47.6 |
| Mean co-rated items per pair | 12.1 | 4.5 |
| Pairs with no co-rated items | 6.3% | 39.6% |
| Full similarity matrix (numpy) | 0.028 s | 0.055 s |
| RMSE / MAE, strategy A (co-rated) | 1.019 / 0.806 | 1.057 / 0.836 |
| RMSE / MAE, strategy C (β = 50) | 1.019 / 0.809 | 1.091 / 0.846 |
| Prediction time for test, strategy A | 0.19 s | 0.11 s |
| Similarity stability r, strategy A | 0.36 | 0.19 |
| Similarity stability r, strategy C (β = 50) | 0.76 | 0.78 |

Item-Based has 3 times more pairs, and its matrix takes 2 times longer. User-Based is more accurate: RMSE 1.019 against 1.057.

For stability, I computed similarities on two random halves of train and compared pairs with at least 5 common ratings in both. With strategy A the correlation is 0.36 for users and only 0.19 for movies. With C both are about 0.77, but this number is too high, because min(common, β)/β is almost the same in both halves.

So item similarities are less stable here. This is not against the theory. It agrees with it: the side with more ratings per entity gives better similarities, and here it is the users.

### 2.2 Missing-value strategy trade-off

A uses only co-rated items. B fills missing values with the user mean (UB) or movie mean (IB). C is A with significance weighting [2]. D is matrix factorization (MF) [5] with SGD: μ + b_u + b_i + p_u·q_i, k = 20.

**Table II — Missing-value strategies (test RMSE, preparation time)**

| Strategy | Simplicity | Bias | Computing cost | RMSE UB | RMSE IB | Prep time UB / IB |
|---|---|---|---|---|---|---|
| A co-rated only | simple | sim = 1.0 for pairs with 1 common item | low | 1.019 | 1.057 | 0.034 / 0.064 s |
| B mean imputation | simplest | pulls everything to the mean | low time, dense vectors | 1.049 | 1.080 | 0.007 / 0.011 s |
| C weighted, β = 10 | simple, one parameter | less noise when there are few common items | same as A | 1.015 | 1.046 | 0.035 / 0.067 s |
| C weighted, β = 50 | simple, one parameter | less noise when there are few common items | same as A | 1.019 | 1.091 | 0.034 / 0.066 s |
| D matrix factorization | complex | learns user and item biases | high | 0.929 | (one model) | 11.8 s |
| Baselines | | | | global mean 1.128, user mean 1.050, item mean 1.021 | | |

B is the easiest idea. But after filling, all vectors look like the mean, so almost every pair looks similar. B is worse than A for both methods.

C costs only one multiplication. With β = 10 it is the best CF result (UB 1.015, IB 1.046). With β = 50 it is not better for UB and worse for IB (1.091). I still chose β = 50 for the app, because with it the neighbours have more common movies. With A, 19 of the top-20 neighbours of a typical user have fewer than 5 common movies. With C (β = 50), none.

C has one limit: it does not change the score of a rare movie. All neighbours of a rare movie have common = 1 or 2, so C multiplies all their weights by the same small number. A weighted average Σ sim'·r / Σ sim' does not change if all weights are multiplied by one number. In this case it give the same score as before (Table III).

**Table III — Top-5 quality for 50 random users**

| Top-5 config | P@5 | Median popularity | Recs with ≤ 5 ratings |
|---|---|---|---|
| UB, A | 0.048 | 81.5 | 16.4% |
| UB, C (β = 50) | 0.060 | 54 | 31.6% |
| IB, A | 0.020 | 2 | 76.0% |
| IB, C (β = 50) | 0.024 | 2 | 71.2% |
| UB, C (β = 50), ≥ 20 ratings | 0.104 | 214 | 0% |
| IB, C (β = 50), ≥ 20 ratings | 0.064 | 45 | 0% |
| MF | 0.080 | 142 | 0.4% |
| Most popular | 0.156 | 405 | 0% |

Without a popularity threshold, 71% of Item-Based recommendations have 5 or fewer ratings, even with C. Only the threshold of 20 ratings removes them, and then UB with C has the best CF precision (0.104).

![Fig. 1. Precision@5 and the share of rare movies in the Top-5 lists](experiments/results/top5_quality.png)

MF is different with the other methods. It has the best RMSE (0.929), but training takes 11.8 s against 0.03 to 0.06 s for a similarity matrix. It needs hyperparameters and early stopping. On the validation set (10% of train) the best epoch was 20 (RMSE 0.915). At epoch 30 the RMSE was 0.927. MF is also harder to explain.

### 2.3 Cold start and sparsity

CF compares rows (users) or columns (movies). A new user or a new movie has no data, so there is nothing to compare [6]. A new user gives no informations to the system. In our split, 32 test movies had no train ratings, and every CF method used the fallback for them.

I also simulated new users. I took 5 users (82, 394, 397, 587, 710) and kept 0, 1, 3, 5 or 10 of their ratings in train. Their other 553 ratings were the test.

**Table IV — Simulated new users (553 held-out ratings)**

| Ratings kept | Coverage UB = IB | RMSE UB-A | RMSE IB-A | RMSE UB-C | RMSE fallback only |
|---|---|---|---|---|---|
| 0 | 0% | 1.145 | 1.145 | 1.145 | 1.145 |
| 1 | 99.5% | 0.924 | 1.628 | 0.924 | 1.628 |
| 3 | 99.5% | 0.923 | 1.224 | 0.903 | 1.235 |
| 5 | 99.6% | 0.941 | 1.178 | 0.891 | 1.189 |
| 10 | 99.6% | 0.950 | 1.112 | 0.864 | 1.119 |

With 0 ratings CF can do nothing: coverage is 0. With 1 rating, IB repeats this rating, like the fallback. UB looks good, but all raters of that movie get sim = 1.0, so the neighbours are almost random. With UB-C, when the user has more ratings (from 1 to 10), the error goes down from 0.924 to 0.864.

The second problem is too little evidence. We discuss about this problem with numbers. Of 444,153 user pairs, 37,739 (8.5%) have only one common movie, and all of them have cosine exactly 1.0. For 2 or more common movies the mean similarity is about 0.94, and it is almost the same for any number of common movies. So co-rated cosine on positive ratings cannot show which users are really similar. For example, user 1 gave "Air Bud (1997)" a 1 and user 88 gave it a 5. This is their only common movie, and their cosine is 1.0.

**Table V — Test RMSE by the number of train ratings of the movie**

| Train ratings of the movie | Test pairs | UB-A | IB-A | IB-C (β = 50) | MF |
|---|---|---|---|---|---|
| 0 | 36 | 0.970 | 0.970 | 0.970 | 0.970 |
| 1-5 | 278 | 1.424 | 1.294 | 1.362 | 1.094 |
| 6-20 | 1197 | 1.164 | 1.180 | 1.425 | 1.038 |
| 21-100 | 7231 | 1.032 | 1.096 | 1.185 | 0.947 |
| >100 | 11258 | 0.982 | 1.011 | 0.974 | 0.899 |

For rare movies (1 to 20 ratings) the CF error is between 1.16 and 1.43. MF is the best in every group.

![Fig. 2. Test RMSE by movie popularity](experiments/results/rmse_by_item_popularity.png)

In practice, a new user gets popular movies or a few questions at registration ("rate 10 movies", "choose your favourite genres"). For a new movie, content-based filtering helps, because genres exist from the first day. Real systems usually use a hybrid: content-based or popularity at the start, CF when there are enough ratings.

## 3 Method

**Data.** `data.js` loads `u.item` and `u.data` with `fetch()`. The file `u.item` uses Latin-1, not UTF-8, and some titles were broken, for example "Misérables, Les (1995)". So I read it with `arrayBuffer()` and `new TextDecoder('latin1')`.

**Rating matrix.** `buildRatingMatrix()` makes one `Float32Array` row per user, indexed by real ids. A value of 0 means "not rated", because real ratings are from 1 to 5. The function also builds helper lists for the speed. They keep the movies of each user, the users of each movie, and the mean rating of each user.

**Similarity and prediction.** `script.js` has the CF logic and the UI:

```
cosine(a, b) = Σ a_k·b_k / ( sqrt(Σ a_k²) · sqrt(Σ b_k²) )   k = co-rated positions
sim'(a, b)   = cosine(a, b) · min(common, β) / β             β = 50, common = count of k
prediction   = Σ sim'·r / Σ sim'                             top neighbours with sim' > 0
```

- User-Based CF: the neighbours are the N = 20 most similar users who rated the movie.
- Item-Based CF: the neighbours are the K = 20 movies of this user that are most similar to the target movie.
- If there are no neighbours, the prediction is null and the page shows the reason.

In Top-5 a movie must pass two reliability thresholds. The prediction uses at least MIN_SUPPORT = 3 neighbours. The movie has at least MIN_ITEM_RATINGS = 20 ratings. Scores are compared with 1e-6 accuracy, then by support and id, so very small float differences do not change the order.

**Cache.** User-user similarities are computed one time per user. Item-item similarities are computed only for needed pairs and saved in a cache (`Map`). The cache is cleared when user changes, because one Top-5 for a heavy user needs about 340,000 pairs. In JavaScriptCore the Item-Based Top-5 for user 1 took 63 ms on the first click and 20 ms on the second click.

![Fig. 3. The "Predict a Rating" block for user 1 and Toy Story](experiments/results/ui_predict.png)

**Tools.** The app is written in vanilla JavaScript (HTML, CSS, JS) with no libraries. Node.js was not installed, so I ran the real `script.js` functions with Apple's JavaScriptCore (`jsc`) and tested the page in headless Chrome. The offline experiments use Python with numpy (`experiments/evaluate.py`), and `check_cf.py` is a numpy reference of the same formulas.

**Code availability.** Code: https://github.com/redlymood/Machine-Learning-and-Recommender-Systems/tree/main/week3. Live demo: https://redlymood.github.io/Machine-Learning-and-Recommender-Systems/week3/.

## 4 Experiments and Verification

**Setup.** Every user have at least 20 ratings. For each user I put a random 20% of the ratings into test and 80% into train (seed 42). Train has 80,000 ratings and test has 20,000. All 943 users stay in train. 32 test movies have no ratings in train (36 test pairs).

**Metrics.**

- RMSE and MAE. For null predictions I use the user mean as fallback.
- Coverage: the share of test pairs where the method itself gave a prediction.
- Precision@5 for 50 random users: a movie is relevant if it is in the user's test with rating 4 or 5. Candidates are all movies not rated in train.
- Time: preparation (similarity matrix or training) and prediction of the whole test.

**Browser vs numpy.** For the pairs (1, 1), (1, 300), (13, 50), (405, 1500) and for the Top-5 lists of users 1 and 405, the page in headless Chrome and `check_cf.py` gave the same numbers in all digits.

I also checked one weighted similarity by hand: user 1 and user 866 (full data, as in the app).

**Vectors.** The co-rated movies are "Kolya (1996)", "Full Monty, The (1997)" and "Good Will Hunting (1997)".

- User 1: [5, 5, 3]
- User 866: [3, 3, 2]
- common = 3

**Hand calculation.**

- Dot product = (5×3) + (5×3) + (3×2) = 15 + 15 + 6 = 36
- ||a|| = √(25 + 25 + 9) = √59 ≈ 7.6811457
- ||b|| = √(9 + 9 + 4) = √22 ≈ 4.6904158
- cosine = 36 / (7.6811457 × 4.6904158) = 36 / 36.0277 ≈ 0.9992293
- sim' = 0.9992293 × min(3, 50) / 50 = 0.9992293 × 0.06 ≈ 0.0599538

**Code result (JavaScriptCore).** The real `cosineSimilarity` and `weightedSimilarity` functions gave sim = 0.9992292869760641, common = 3, and sim' = 0.059953757218563844.

**Result.** Hand value = 0.0600, code value = 0.0600. They are equal. Without weighting this neighbour has sim = 0.9992 and looks almost perfect, but with only 3 common movies its weight drops to 0.06.

## 5 Discussion

**Failure case.** In my first version the Top-5 lists looked strange. For user 1, User-Based CF recommended "Star Kid (1997)" and "Prefontaine (1997)" with score 5.0, and Item-Based CF recommended "Chairman of the Board (1998)", a movie with only one rating. Also Item-Based predicted 2.6 for user 13 and "Star Wars (1977)", but the real rating was 5.

**Root cause.** I used cosine only on co-rated movies. Ratings are always positive, so when two users have only one common movie, the cosine is exactly 1.0. In 8.5% of user pairs this happen. 19 of 20 neighbours of typical user had less than 5 common movies. So the "most similar" neighbours were just random people with very small overlap.

**Fix + verification.** I added significance weighting: sim' = sim · min(common, 50) / 50. After this the Star Wars prediction became 4.3. But rare movies stayed in Top-5, because when all weights are multiplied by one number, the weighted average does not change. So I added a second threshold: a movie needs at least 20 ratings. I checked the browser results against a numpy script, and they was equal in all digits.

**What worked.** With weighting, the neighbours with less than 5 common movies went from 19 of 20 to 0. With the popularity threshold, P@5 of User-Based CF went from 0.060 to 0.104, and there are no movies with 5 or fewer ratings in the lists.

**What surprised me.** I expected item-based is better, because I read that Amazon use it. But here user-based was more accurate (RMSE 1.019 vs 1.057), and item similarities were less stable (0.19 vs 0.36). Then I understand why: in MovieLens 100k there are more movies than users, so one movie has fewer ratings than one user. Also the simple "most popular" list had the best P@5 (0.156), better than all my CF methods.

**Next improvement.** I would use mean-centered ratings (adjusted cosine or Pearson), because now almost all similarities are near 0.94. For new movies I would mix CF with the genre-based method from week 2.

## 6 AI Usage Disclosure

I used an AI coding assistant (Claude Code) to read the starter code, write the JavaScript functions, and run the experiments in Python, JavaScriptCore and headless Chrome. It also helped me to build the tables and the PDF from my results.

What I verified myself:

- I compared the Top-5 lists for user 1 in the browser with the output of `check_cf.py`.
- I checked the six references and their DOIs on Crossref.

## References

[1] F. M. Harper and J. A. Konstan, "The MovieLens Datasets: History and Context," *ACM Trans. Interactive Intelligent Systems*, vol. 5, no. 4, 2015. doi:10.1145/2827872

[2] J. L. Herlocker, J. A. Konstan, A. Borchers, and J. Riedl, "An Algorithmic Framework for Performing Collaborative Filtering," in *Proc. SIGIR '99*, 1999, pp. 230-237. doi:10.1145/312624.312682

[3] B. Sarwar, G. Karypis, J. Konstan, and J. Riedl, "Item-Based Collaborative Filtering Recommendation Algorithms," in *Proc. WWW '01*, 2001, pp. 285-295. doi:10.1145/371920.372071

[4] G. Linden, B. Smith, and J. York, "Amazon.com Recommendations: Item-to-Item Collaborative Filtering," *IEEE Internet Computing*, vol. 7, no. 1, pp. 76-80, 2003. doi:10.1109/MIC.2003.1167344

[5] Y. Koren, R. Bell, and C. Volinsky, "Matrix Factorization Techniques for Recommender Systems," *Computer*, vol. 42, no. 8, pp. 30-37, 2009. doi:10.1109/MC.2009.263

[6] A. I. Schein, A. Popescul, L. H. Ungar, and D. M. Pennock, "Methods and Metrics for Cold-Start Recommendations," in *Proc. SIGIR '02*, 2002, pp. 253-260. doi:10.1145/564376.564421

## Introduction to Computational Thinking and Data Science - Notes

## Clustering

### Hierarchical Clustering

^86413f

1. Start by assigning each item to its own cluster.
2. Find the two most similar clusters and merge then into a single cluster.
3. Repeat until all items are clustered into a single cluster.
#### Linkage Metrics

^f2bf45

The way in which you evaluate the distance between two clusters
- **Single-Linkage** - For two clusters being compared, compare them using the shortest distance between a point from one cluster and a point from the other.
- **Complete-Linkage** - For two clusters being compared, compare them using the greatest distance between a point from one cluster and a point from the other.
- **Average-Linkage** - Take the average distance of all combination of points from one cluster to another.
> This algorithm is **deterministic**. Given a particular distance metric, you always get the same answer. It is also a **greedy** algorithm as you are making locally optimal decisions at each point (merging clusters with minimum distance) which may or may not be globally optimal.

#### Performance
This algorithm can be extremely slow on large datasets. The naive algorithm has a complexity $O(n^3)$. However for some linkage metrics like single-linkage, there exist some tricks and optimizations which allow it to run in $O(n^2)$, which is still quite slow.

### K-Means
Most commonly used clustering algorithm.
1. Randomly choose $k$ examples to be your initial centroids.
2. Create $k$ clusters by assigning each sample to the nearest centroid.
3. Recompute the centroids by taking the average of the distances within each cluster.
4. Repeat the process until the centroids stop moving.
#### Performance
At each iteration, for each of the $k$ centroids, the algorithm computes the distance from each sample to the centroid. Thus for a single iteration with $n$ examples, the computational complexity is $O(k\cdot n)$. Typically $k\cdot n \ll n^2\ or\  n^3$, and this results in less computation as compared to Hierarchical Clustering. It is also observed that the K-Means algorithm tends to reach its convergence limit relatively quickly too.
#### Downsides
- Choosing the wrong value of $k$ can lead to strange and inaccurate results.
- The results are dependent on the initial centroids which are chosen randomly. The algorithm is **not deterministic**.
#### Choosing K
- **A priori knowledge** about the problem domain. If you have existing knowledge about the domain in which the problem lies then that can be helpful in determining the optimal $k$. For example, it could be known that there are only $n$ types of bacteria which can cause diseases which you have data for. Setting $k=n$ would allow you to determine which type was responsible for a particular sample and identify correlations.
- **Elbow Method** - Run the algorithm with many different values of $k$ and plot the resulting errors (sum of distances between points in each cluster) for the corresponding number of clusters. Typically, the $k$ which occurs at the '*elbow*' of this graph results in a good clustering.
- Run **Hierarchical Clustering** on a subset of your data and get a sense of the similarities and cluster relationships to choose a value of $k$.

## Classification

**Imbalanced Datasets** - When designing and machine learning model, if your dataset contains a class imbalance, that is, many more samples of one class as compared to others, then *accuracy* is not a sufficient metric to evaluate the model.
	If a medical dataset contains 700 samples labeled 'Died' and 300 samples labeled 'Survived', then a model which predicts every sample as 'Died' would result in a 70% accuracy.
	For cases like rare diseases which occur to < 0.01% of the population, a model which simply predicts that a patient does not have a disease would be 99.99% accurate.

### Evaluation Ratios
#### Key Terms
- **True Positive** - An *actual positive* (1) value, *predicted as positive*: `(1,1)`
- **False Negative** - An *actual positive* (1) value, *predicted as negative*: `(1,0)`
- **True Negative** - An *actual negative* (0) value, *predicted as negative*: `(0,0)`
- **False Positive** - An *actual negative* (0) value, *predicted as positive*: `(0,1)`

#### Evaluation Ratios
- **Sensitivity (Recall)** - How good your model is at identifying *actual* positive cases:
	$\frac{true\ positive}{true\ positive\ +\ false\ negative}$, that is: $\frac{correctly\ classified\ positives}{all\ actual\ positives}$
	Imagine a spam filter. Sensitivity would tell you the percentage of actual spam emails the filter catches out of all real spam emails. High sensitivity is crucial for spam filters to avoid missing important emails
- **Specificity** - How good your model is at identifying *actual* negative cases:
	$\frac{true\ negative}{true\ negative\ +\ false\ positive}$, that is: $\frac{correctly\ classified\ negatives}{all\ actual\ negatives}$
	A medical test for a disease. Specificity indicates what percentage of healthy individuals (True Negatives) the test correctly identifies as negative, out of all the healthy individuals who took the test. A high specificity is important to avoid unnecessary follow-up tests on healthy people.
- **Positive Predictive Value (Precision)** - If you make a positive prediction, what's the probability that it's actually correct
	$\frac{true\ positive}{true\ positive\ +\ false\ positives}$, that is: $\frac{correct\ positive\ prediction}{all\ positive\ predictions}$
- **Negative Predictive Value** - If you make a negative prediction, what's the probability that it's actually correct
	$\frac{true\ negative}{true\ negative\ +\ false\ negatives}$, that is: $\frac{correct\ negative\ prediction}{all\ negative\ predictions}$

### Logistical Regression

> ==Statistics about the data is NOT THE SAME as the data itself==
> ==Analysis of BAD DATA is worse than no analysis at all==

Involves assigning weights to each feature. A positive weight (almost) implies that a variable is *positively correlated* with the outcome (label). A negative weight implies that a variable is *negatively correlated* with the outcome. The magnitude of the weight implies the strength of the correlation.

> *A key goal of regression analysis is to isolate the relationship between each independent variable and the dependent variable.*

A change in the p-value (threshold of classification?) parameter of LR results in a change in sensitivity/specificity, and the positive predictive value of the model. In the case of the Titanic, with classes 'Survived' and 'Died', a higher p-value (0.9) would increase the precision (+pv) but lower the sensitivity. That is, out of all the cases *that the model predicted*, most of them would be correct (high precision) **however**, the model would also *misclassify many of the cases that actually survived* (low sensitivity).
#### Types of Logistical Regression
LR is primarily used in two different ways:
1. L1 - Finding some weights associated with features and 'drive them to 0'. Particularly useful when you have a high-dimensional problem relative to the number of samples. L1 is designed to avoid overfitting by taking many of the variables and giving them 0 weight. However, if you have variables which are correlated, L1 would give one (or some) of them 0 weight and give a relatively higher weight to the others.
2. L2 - Is designed so that it spreads the weight across the variables. That is, if there are many correlated variables, each of them would get a small amount of the weight. However, this might make it seem that that subset of variables are not very important which could make a difference in inference when dealing with high dimensionality.
#### Identifying Variable Relationships
**How to tell if there is a correlation between a set of features?**
One method might be to run comparative analyses on the data with all features thought to be correlated and then with some features eliminated. A significant change in weight of the variables or the lack thereof might provide some insight into whether the variables are correlated or not.

https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/
> Multicollinearity  affects coefficients and p-values, but it does not influence the predictions, the precision of predictions, and the goodness-of-fit statistics

#### Receiver Operating Characteristic (ROC)
Changing the thresholding p-value has a fairly significant impact on the outcome of the model and the inferences or consequences one may gather from it. Therefore, it is crucial to identify the purpose for which the model is being designed and to consider the possible side-effects of the parameters, features and samples chosen when training the model.

The ROC presents a way of evaluating not just one cutoff (p-value) value but looking at all possible cutoffs and looking at its 'shape' in graphical format.
It involves training a *single* model and then applying it with varying values of `p` and tracking all calculated results. The results are then plotted such that the y-axis represents sensitivity and the x-axis represents (1-specificity). 
	**Why plot (1-specificity)?**
	In order to be able to compute the AOC. If just specificity was plotted, then the resulting curve would be convex making the AUROC analysis more complicated.
The method also involves calculating the *Area* under the plotted curve.

**Interpreting the ROC Graph**
![[Pasted image 20240610170827.png]]
- The top right corner of the graph, with **high sensitivity** and **low specificity** (as (1 - spec) increases with decrease in spec). **A Low specificity means high amount of false positives**.
- The bottom left corner is very specific and very insensitive. That means little to no samples are being declared positive - **High amount of false negatives**.
The knee of this graph would be a good place to choose a p-value for the model.
> The green line shown within the graph represents the performance of a *Random Classifier*.

The Area under the ROC (blue) curve represents how good the model is. The Area between the ROC curve and the Random classifier line shows how much better the model is compared to a random classification. *The closer the curve is to random, the worse the model is considered to perform.*

#### Convenience Sampling
Results in:
- **Survivorship Bias** - Course evaluations at the end of a semester, etc.
- **Non-Response Bias**: Typically present in data acquired from volunteer based surveys in which the participants who gave a response may not appropriately represent the entire population, the majority of which did not provide a response.
When samples are not random and independent, **we should not draw conclusions from them** using things like the empirical rule or central limit theorem.
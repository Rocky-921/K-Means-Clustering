# K-Means-Clustering

The dataset consists of N observations.
Each observation is a d-dimensional vector.
Given a set of observations x1,x2,...,xN, the dataset must be partitioned into k disjoint clusters {S_1,S_2, , S_k}.
Suppose that mi denotes the arithmetic mean of the data points in subset Si.
mi is called the centroid of the cluster.
The choice of clusters must minimize the sum of the square of Euclidean distances between each point in the set and the centroid.
Develop a parallel solution on Nvidia GPU using CUDA and compare with that of cuML.

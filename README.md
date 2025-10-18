# K-Means-Clustering

Problem Statement:-

The dataset consists of N observations.
Each observation is a d-dimensional vector.
Given a set of observations x1,x2,...,xN, the dataset must be partitioned into k disjoint clusters {S_1,S_2, , S_k}.
Suppose that mi denotes the arithmetic mean of the data points in subset Si.
mi is called the centroid of the cluster.
The choice of clusters must minimise the sum of the squares of Euclidean distances between each point in the set and the centroid.

Develop a parallel solution on an Nvidia GPU using CUDA and compare with that of cuML.

Created various implementations as given under the All_implementations folder.

Created checker.py Python script to check the performance of the implementations with that of NVIDIA's built-in.

Created benchmark.py to plot the performance of the implementations with each other using the plots given in the corresponding folder.

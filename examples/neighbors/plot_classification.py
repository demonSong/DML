"""
================================
Nearest Neighbors Classification
================================

Sample usage of Nearest Neighbors classification.
It will plot the decision boundaries for each class.
"""

from learn import datasets
from learn.neighbors.nearest_centroid import NearestCentroid
import numpy as np
from learn.metric.pairwise import euclidean_distances,paired_euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances

# import some data to play with
X,y = datasets.load_hellen_appointment(True)
res = paired_euclidean_distances(X, X)

pairwise_distances(X,Y = None,metric = "euclidean",n_jobs = 5)


# clf = NearestCentroid()
# clf.fit(X, y)






    
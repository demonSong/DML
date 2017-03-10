"""
================================
Nearest Neighbors Classification
================================

Sample usage of Nearest Neighbors classification.
It will plot the decision boundaries for each class.
"""

from learn import datasets
import numpy as np
from learn.metric.pairwise import euclidean_distances,paired_euclidean_distances,\
    pairwise_distances
from learn.neighbors.nearest_centroid import NearestCentroid


# import some data to play with
X,y = datasets.load_hellen_appointment(True)

# pairwise_distances(X,Y = None,metric = "euclidean",n_jobs = 5)
# euclidean_distances(X,X)

clf = NearestCentroid()
clf.fit(X, y)
pred = clf.predict(X)
score = clf.score(X,y)






    
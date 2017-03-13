"""
================================
Nearest Neighbors Classification
================================

Sample usage of Nearest Neighbors classification.
It will plot the decision boundaries for each class.
"""

from learn import datasets
from learn.neighbors.nearest_centroid import NearestCentroid

# import some data to play with
X,y = datasets.load_hellen_appointment(True)

clf = NearestCentroid()
clf.fit(X, y)
score = clf.score(X,y)

print("预测准确率：",score)






    
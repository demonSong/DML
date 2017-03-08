"""
================================
Nearest Neighbors Classification
================================

Sample usage of Nearest Neighbors classification.
It will plot the decision boundaries for each class.
"""

import matplotlib.pyplot as plt
from learn import datasets


# import some data to play with
returnMat,classLabelVector = datasets.load_hellen_appointment(True)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(returnMat[:,1],returnMat[:,2])
plt.show()





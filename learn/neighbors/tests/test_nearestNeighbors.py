import numpy as np
from learn.neighbors.unsupervised import NearestNeighbors


X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

# NearestNeighbors 只是在做参数的设置
nbrs = NearestNeighbors(n_neighbors=3, algorithm='kd_tree',leaf_size=30).fit(X)
result= nbrs.kneighbors(X)


#distances, indices = nbrs.kneighbors(X)


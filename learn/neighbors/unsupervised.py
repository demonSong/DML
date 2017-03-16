from .base import NeighborsBase
from .base import UnsupervisedMixin
from .base import KNeighborsMixin

class NearestNeighbors(NeighborsBase,KNeighborsMixin,UnsupervisedMixin):

    def __init__(self, n_neighbors=5, radius=1.0,
                 algorithm='auto', leaf_size=30, metric='minkowski',
                 p=2, metric_params=None, n_jobs=1, **kwargs):
        self._init__params(n_neighbors=n_neighbors,
                           radius=radius,
                           algorithm=algorithm,
                           leaf_size=leaf_size, metric=metric, p=p,
                           metric_params=metric_params, n_jobs=n_jobs, **kwargs)

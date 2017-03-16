from abc import abstractmethod
from .kd_tree import KDTree


class NeighborsBase(object):
    @abstractmethod
    def __init__(self):
        pass

    def _init__params(self, n_neighbors=None, radius=None,
                      algorithm='auto', leaf_size=30, metric='minkowski',
                      p=2, metric_params=None, n_jobs=1):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.metric_params = metric_params
        self.p = p
        self.n_jobs = n_jobs
        self._fit_X = None
        self._tree = None
        self._fit_method = None

    def _fit(self, X):

        n_samples = X.shape[0]

        if n_samples == 0:
            raise ValueError("n_samples must be greater than 0")

        self._fit_method = self.algorithm
        self._fit_X = X


        if self._fit_method == "kd_tree":
            print("kd_tree algorithm worked")
            self._tree = KDTree(X,self.leaf_size)

        elif self._fit_method == "ball_tree":
            print("ball_tree algorithm worked")

        return self


class KNeighborsMixin(object):

    def kneighbors(self,X=None,n_neighbors=None,return_distance=True):

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        if X is not None:
            print("check_array method invoke")
        else:
            X = self._fit_x

        # 判断训练数据 小于 k近邻数据时 报错

        n_samples, _ = X.shape # 样例总数 和 样例维度

        if self._fit_method in ['ball_tree','kd_tree']:
            result = self._tree.query(X,n_neighbors,return_distance)

        return result



class UnsupervisedMixin(object):

    def fit(self, X, y=None):
        return self._fit(X)

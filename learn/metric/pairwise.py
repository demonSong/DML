import numpy as np

from ..utils.extmath import row_norms,safe_sparse_dot
from sklearn.externals.joblib.parallel import cpu_count # 该模块暂且不研究
from sklearn.utils import gen_even_slices
from sklearn.externals.joblib.parallel import Parallel, delayed


def paired_euclidean_distances(X, Y):
    """
    Computes the paired euclidean distances between X and Y

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    Y : array-like, shape (n_samples, n_features)

    Returns
    -------
    distances : ndarray (n_samples, )
    """
    return row_norms(X - Y)



def euclidean_distances(X, Y=None, squared=False):
    """
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples_1, n_features)
 
    Y : {array-like, sparse matrix}, shape (n_samples_2, n_features)
     
    For efficiency reasons, the euclidean distance between a pair of row
    vector x and y is computed as::
 
        dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))
         
    """
    XX = row_norms(X,squared = True)[:,np.newaxis]        
    if X is Y:
        YY = XX.T
    YY = row_norms(Y, squared = True)[np.newaxis,:]
     
    distances = safe_sparse_dot(X,Y.T)
     
    distances *= -2
    distances += XX
    distances += YY
     
    np.maximum(distances,0,out = distances) # out 指定输出矩阵
     
    return distances if squared else np.sqrt(distances,out=distances) # 需要开根号则开根号

def _parallel_pairwise(X,Y,func,n_jobs,**kwds):
    
    if n_jobs < 0:
        n_jobs = max(cpu_count() + 1 + n_jobs,1)
    
    if Y is None:
        Y = X

    if n_jobs == 1:
        # Special case to avoid picklability checks in delayed
        return func(X, Y, **kwds)

    # TODO: in some cases, backend='threading' may be appropriate
    fd = delayed(func)
    ret = Parallel(n_jobs=n_jobs, verbose=0)(
        fd(X, Y[s], **kwds)
        for s in gen_even_slices(Y.shape[0], n_jobs))

    return np.hstack(ret)

def pairwise_distances(X,Y = None,metric = "euclidean",n_jobs = 1, **kwds):
    func = PAIRWISE_DISTANCE_FUNCTIONS[metric]
    return _parallel_pairwise(X, Y, func, n_jobs, **kwds)


# Helper functions - distance
PAIRWISE_DISTANCE_FUNCTIONS = {
    'euclidean': euclidean_distances
}

    
    
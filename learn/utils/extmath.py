import numpy as np

def row_norms(X,squared = False):
    norms = np.einsum('ij,ij->i',X,X) # 该函数的作用是什么?
    if not squared:
        np.sqrt(norms,norms)
    return norms


def safe_sparse_dot(a,b):
    return np.dot(a,b)
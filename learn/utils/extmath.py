import numpy as np

def row_norms(X,squared = False): # square 表示为 开平方
    norms = np.einsum('ij,ij->i',X,X) # 该函数的作用是什么?
    if not squared:
        np.sqrt(norms,norms)    # norms = 根号 norms
    return norms


def safe_sparse_dot(a,b):
    return np.dot(a,b)
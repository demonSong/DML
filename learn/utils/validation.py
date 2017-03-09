import numpy as np

def column_or_1d(y,warn = False):
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    
    raise ValueError("bad input shape {0}".format(shape))
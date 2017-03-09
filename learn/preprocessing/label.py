'''
Created on 2017��3��9��

@author: DemonSong
'''

import numpy as np


from ..utils import column_or_1d

class LabelEncoder():
    
    def fit_transform(self,y):
        y = column_or_1d(y,warn = True)
        self.classes_, y = np.unique(y, return_inverse=True) # return_inverse 为True时，返回y值
        return y
        
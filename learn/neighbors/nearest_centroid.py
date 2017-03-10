import numpy as np

from ..base import ClassifierMixin
from ..preprocessing import LabelEncoder
from learn.metric.pairwise import pairwise_distances


class NearestCentroid(ClassifierMixin):
    def __init__(self,metric='euclidean'):
        self.metric = metric
    
    
    def fit(self,X,y):
        n_samples, n_features = X.shape
        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        self.classes_ = classes = le.classes_
        n_classes = classes.size
        if n_classes < 2:
            raise ValueError('y has less than 2 classes')
        self.centroids_ = np.empty((n_classes,n_features),dtype = np.float64) # 对应于三个特征 三种标签集合
        nk = np.zeros(n_classes)
        
        for cur_class in range(n_classes): # 标签数
            center_mask = y_ind == cur_class
            nk[cur_class] = np.sum(center_mask)
            # 欧几里德距离计算
            self.centroids_[cur_class] = X[center_mask].mean(axis = 0) #计算每个标签对应的每个特征的均值
        return self
    
    def predict(self,X):
        return self.classes_[pairwise_distances(
            X,self.centroids_,metric=self.metric).argmin(axis = 1)] # axis = 0 针对矩阵的维度 axis = n 表示第n维
        
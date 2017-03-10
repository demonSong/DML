import numpy as np

def _weighted_sum(sample_score,sample_weight,normalize = False):
    if normalize:
        return np.average(sample_score,weights = sample_weight)
    elif sample_weight is not None:
        return np.dot(sample_score,sample_weight)
    else:
        return sample_score.sum()
    
# 计算 真实值和预测值之间的 准确率
def accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):
    score = y_true == y_pred
    return _weighted_sum(score, sample_weight, normalize)
    
    

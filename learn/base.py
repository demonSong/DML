
class ClassifierMixin(object):
    
    def score(self,X,y,sample_weight = None):
        
        from .metric import accuracy_score
        return accuracy_score(y,self.predict(X),sample_weight = sample_weight);
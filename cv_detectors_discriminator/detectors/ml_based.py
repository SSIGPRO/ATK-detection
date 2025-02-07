from detectors._base import Detector
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class SklearnDetector(Detector):
    def __init__(self, **kwargs):
        pass

    def fit(self, X_train):
        self.detector.fit(X_train)
        return
    
    def score(self, X_test):
        return self.detector.decision_function(X_test)
    
class OCSVM(SklearnDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.detector = make_pipeline(
                StandardScaler(),
                OneClassSVM(**self.kwargs)
                )
        return 
    
class LOF(SklearnDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs # number of neighbors
        self.detector = make_pipeline(
                StandardScaler(),
                LocalOutlierFactor(**self.kwargs, novelty=True)
                )
        return 
    
class IF(SklearnDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs # number of estimators
        self.detector = make_pipeline(
                StandardScaler(),
                IsolationForest(**self.kwargs)
                )
        return
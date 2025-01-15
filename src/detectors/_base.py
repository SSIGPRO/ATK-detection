import abc
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def AUC(scores, n_ori, n_atk):
    labels = np.hstack((torch.zeros(n_ori), torch.ones(n_atk)))
    return 1-roc_auc_score(labels, scores)
    
def P_D(scores, n_ori, n_atk):
        auc = AUC(scores, n_ori, n_atk)
        return 0.5 + np.abs(0.5 - auc)
    
class Detector():
    def __init__(self, **kwargs):
        pass
    
    @abc.abstractmethod
    def fit(self, dataset_train):
        pass
    
    @abc.abstractmethod
    def score(self, X_test):
        pass

    def test(self, X_ori, X_atk, metric):
        # concat original and attacked data
        X_test = torch.cat([X_ori, X_atk]).detach().numpy()
        scores = self.score(X_test)

        # test the detector in terms of AUC or P_D
        if metric == 'AUC':
            n_ori = X_ori.shape[0]
            n_atk = X_atk.shape[0]
            return AUC(scores, n_ori, n_atk)
        elif metric == 'P_D':
            n_ori = X_ori.shape[0]
            n_atk = X_atk.shape[0]
            return P_D(scores, n_ori, n_atk)

    

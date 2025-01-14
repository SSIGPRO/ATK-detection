import torch
from sklearn.metrics import roc_auc_score

def AUC(scores, n_ori, n_atk):
    labels = torch.hstack((torch.zeros(n_ori), torch.ones(n_atk)))
    return 1-roc_auc_score(labels, scores)
    
def P_D(scores):
        auc = AUC(scores)
        return 0.5 + np.abs(0.5 - auc)
    
class Detector():
    
    def __init__(self, **kwargs):
        pass
    
    def fit(self, dataset_train):
        pass
    
    def score(self, X_test):
        return scores
        
    def test(self, X_ori, X_atk, metric):
        # concat original and attacked data
        X_test = torch.cat([X_ori, X_atk])
        scores = self.score(X_test)

        # test the detector in terms of AUC or P_D
        if metric == 'AUC':
            n_ori = X_ori.shape[0]
            n_atk = X_atk.shape[0]
            return AUC(scores, n_ori, n_atk)
        elif metric == 'P_D':
            return P_D(scores)

    

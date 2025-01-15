import sys
sys.path.insert(0, '/home/leandro/repos/peepholelib')

# python stuff
import os
sys.path.append('..')
import pandas as pd
from pathlib import Path
from contextlib import ExitStack
import numpy as np
from sklearn.metrics import roc_auc_score 

# detectors stuff
from detectors.ml_based import OCSVM, LOF, IF  
from detectors.gaussian_distribution_based import MD 

# Corevectors 
from peepholelib.coreVectors.coreVectors import CoreVectors 

# torch stuff
import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['SCIPY_USE_PROPACK'] = "True"
threads = "64"
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads

def AUC(scores, n_ori, n_atk):
    labels = np.hstack((torch.zeros(n_ori), torch.ones(n_atk)))
    return 1-roc_auc_score(labels, scores)

if __name__ == '__main__':
    #--------------------------------
    # Directories definitions
    #--------------------------------
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 2
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")
    name_model = 'vgg16'
    dataset = 'CIFAR100' 
    verbose = True
    
    results_path = Path.cwd()/'../data'
    results_name = 'results_detectors.pk'

    cvs_name = 'corevectors'
    cvs_path = f'/srv/newpenny/XAI/generated_data/toy_case/corevectors/{dataset}/{name_model}'
    
    cvs_atks_home = f'/srv/newpenny/XAI/generated_data/toy_case'

    verbose = True
    attack_names = ['BIM', 'CW', 'PGD', 'DeepFool']
    # attack_names = ['BIM', 'PGD']
    metric_type = 'AUC'#, P_D

    target_layers = [
                     'classifier.0', 
                     'classifier.3', 
                     'features.14', 
                     'features.28', 
                     'features.7'
                    ]
    detector_configs = {
                        'OCSVM': {
                            'kernel': 'rbf', 
                            'nu': 0.01
                            },
                    
                        'LOF': {'n_neighbors': 20},
                           
                        'IF': {'n_estimators':100},
                            
                        'MD': {}
                        }    
    #--------------------------------
    # Detectors 
    #--------------------------------
    if verbose: print('Creating detectors')
    detectors = {}
    for _det_name in detector_configs:
        
        conf = detector_configs[_det_name].copy()
        
        detectors[_det_name] = eval(_det_name)(**conf)

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    cv = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            verbose = verbose
            )
    
    cv_atks = {} 
    for atk_name in attack_names:
        cv_atk_path = cvs_atks_home+f'/corevectors=my{atk_name}/{dataset}/{name_model}'
        cv_atks[atk_name] = CoreVectors(
                path = cv_atk_path,
                name = cvs_name,
                verbose = verbose
                ) 

    #--------------------------------
    # For saving results 
    #--------------------------------

    metric = {}
    cv_size = 128
    
    for _atk_name in attack_names:
        metric[_atk_name] = {}
        print(f'Attack: {_atk_name}')
        for _det_name in detectors:
            score_ori = {}
            score_atk = {}
            detector = detectors[_det_name]

            with ExitStack() as stack:
                # get dataloader for corevectors from the original dataset 
                stack.enter_context(cv) # enter context manager
                cv.load_only(
                        loaders = ['train', 'test'],
                        verbose = verbose 
                        ) 
                
                train_ori = []
                for _layer in target_layers:
                    train_ori.append(cv._corevds['train']['coreVectors'][_layer][:,:cv_size])
                train_ori = torch.cat(train_ori,dim=1)
                    
                if verbose: print(f'------\n fitting detector {detector}\n------')
                detector.fit(train_ori)

                # get dataloader for corevectors from atks dataset 
                if verbose: print(f'\n---------\nLoading dataset for attack: {_atk_name}')
                stack.enter_context(cv_atks[_atk_name]) # enter context manager
                cv_atks[_atk_name].load_only(
                        loaders = ['train', 'test'],
                        verbose = verbose 
                        )
                 
                if verbose: print(f'computing {metric_type} for {_atk_name} attacked test samples')
                    
                test_ori = []
                for _layer in target_layers:
                    
                    test_ori.append(cv._corevds['test']['coreVectors'][_layer][:,:cv_size])
                test_ori = torch.cat(test_ori,dim=1)

                test_atk = []
                for _layer in target_layers:
                    
                    test_atk.append(cv_atks[_atk_name]._corevds['test']['coreVectors'][_layer][:,:cv_size])
                test_atk = torch.cat(test_atk,dim=1)
                
                metric[_atk_name][_det_name] = detector.test(test_ori, test_atk, metric_type) 
                if verbose: print(f'{_det_name} Vs {_atk_name} metric {metric_type}: ', metric[_atk_name][_det_name])
                
            
    df = pd.DataFrame.from_dict(metric, orient='index')
    print(df)


    

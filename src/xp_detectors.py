import sys
sys.path.insert(0, '/home/leandro/repos/peepholelib')

# python stuff
import os
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import abc  
from pathlib import Path
import pandas as pd
from contextlib import ExitStack

# math stuff
from sklearn.metrics import roc_auc_score

# detectors stuff
from detectors.ml_based import OCSVM, LOF, IF  
from detectors.gaussian_distribution_based import MD 
#from detectors.feature_based import *

# Attcks
import torchattacks

# torch stuff
import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT
from torchvision.models import vgg16, VGG16_Weights
from cuda_selector import auto_cuda
from torch.utils.data import DataLoader

# Corevectors 
#from adv_atk.attacks_base import fds, ftd
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.classifier.classifier_base import trim_corevectors

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['SCIPY_USE_PROPACK'] = "True"
threads = "64"
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    name_model = 'vgg16'
    dataset = 'CIFAR100' 
    verbose = True

    cvs_name = 'corevectors'
    cvs_path = f'/srv/newpenny/XAI/generated_data/corevectors/{dataset}/{name_model}'
    
    cvs_atks_home = f'/srv/newpenny/XAI/generated_data/toy_case'

    verbose = True
    #attack_names = ['BIM', 'CW', 'PGD', 'DeepFool']
    attack_names = ['BIM', 'PGD']
    metric_type = 'AUC' #P_D, AUC

    detector_configs = {
            'OCSVM': {
                'classifier.0': {'cv_size':5, 'kernel': 'rbf', 'nu': 0.01},
                'classifier.3': {'cv_size':10, 'kernel': 'polynomial', 'nu': 0.02},
                },
            'LOF': {
                'classifier.0': {'cv_size':15, 'h': 5},
                'classifier.3': {'cv_size':20, 'h': 4},
                },
            'IF': {
               'classifier.0': {'cv_size':25, 'l':250},
               'classifier.3': {'cv_size':30, 'l':250},
                },
            'MD': {
               'classifier.0': {'cv_size':35, },
               'classifier.3': {'cv_size':40, },
                }
            }
    

    #--------------------------------
    # Detectors 
    #--------------------------------
    detectors = {}
    for _det_name in detector_configs:
        detectors[_det_name] = {}
        for _layer in detector_configs[_det_name]:
            detectors[_det_name][_layer] = eval(_det_name)(**detector_configs[_det_name][_layer])

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    cv = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            device = device
            )
    
    cv_atks = {} 
    for atk_name in attack_names:
        cv_atk_path = cvs_atks_home+f'/corevectors=my{atk_name}/{dataset}/{name_model}'
        cv_atks[atk_name] = CoreVectors(
                path = cv_atk_path,
                name = cvs_name,
                device = device
                ) 

    #--------------------------------
    # Dataframe for results 
    #--------------------------------
    results_df = pandas.DataFrame(columns = ['metric', 'layer', 'detector', 'attack'])
    
    # TODO: iterate over detectors
    detector = detectors[0]
    # TODO: iterate over target layers
    layer = target_layers[0]

    for  
    with ExitStack() as stack:
        # get dataloader for corevectors from the original dataset 
        stack.enter_context(cv) # enter context manager
        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = verbose 
                ) 
        
        if verbose: print(f'------\n fitting detector {detector}\n------')
        detector.fit(cv._corevds['train']['coreVectors'][layer][:,:cv_size])

        # save results 
        #results = torch.zeros(len(attack_name), len(target_layers))

        # get dataloader for corevectors from atks dataset 
        for atk_name in attack_names:
            if verbose: print(f'\n---------\nLoading dataset for attack: {atk_name}')
            stack.enter_context(cv_atks[atk_name]) # enter context manager
            cv_atks[atk_name].load_only(
                    loaders = ['train', 'test'],
                    verbose = verbose 
                    )
             
            if verbose: print(f'computing {metric_type} for {atk_name} attacked test samples')
            data_ori = cv._corevds['test']['coreVectors'][layer][:,:cv_size]
            data_atk = cv_atks[atk_name]._corevds['test']['coreVectors'][layer][:,:cv_size]

            metric = detector.test(data_ori, data_atk, metric_type) 
            if verbose: print(f'metric {metric_type}: ', metric)

    results_df.to_pickle('../data/results_detectors.pk')

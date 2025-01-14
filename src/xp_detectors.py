# python stuff
import os
import sys
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
from adv_atk.attacks_base import fds, ftd

# torch stuff
import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT
from torchvision.models import vgg16, VGG16_Weights
from cuda_selector import auto_cuda
from torch.utils.data import DataLoader

# Corevectors 
from XAI.coreVectors.coreVectors import CoreVectors 
from XAI.classifier.classifier_base import trim_corevectors


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
    bs = 256
    verbose = True

    cvs_name = 'corevectors'
    cvs_path = f'/srv/newpenny/XAI/generated_data/corevectors/{dataset}/{name_model}'

    verbose = True
    #attack_names = ['BIM', 'CW', 'PGD', 'DeepFool']
    attack_names = ['BIM', 'PGD']

    detectors_confs = {
            'OCSVM': {'kernel': 'rbf', 'nu': 0.01},
            'LOF': {'h': 5},
            'IF': {'l':250},
            'MD': {}
            }
    
    layer = 'classifier.0'
    peep_size = 100
    metric = 'AUC' #P_D, AUC

    #--------------------------------
    # Detectors 
    #--------------------------------
    detectors = [eval(k)(**v) for k, v in detectors_confs.items()]

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
        cv_atks[atk_name] = CoreVectors(
                path = f'/srv/newpenny/XAI/generated_data/corevectors_attacks={atk_name}/{dataset}/{name_model}',
                name = cvs_name,
                device = device
                ) 
    
    with ExitStack() as stack:
        # get dataloader for corevectors from the original dataset 
        stack.enter_context(cv) # enter context manager
        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = verbose 
                ) 
        cv_dl = cv.get_dataloaders(
                batch_size = bs,
                verbose = verbose,
                )
        
        # get dataloader for corevectors from atks dataset 
        cv_atk_dl = {}
        for atk_name in attack_names:
            if verbose: print(f'\n---------\nLoading dataset for attack: {atk_name}')
            stack.enter_context(cv_atks[atk_name]) # enter context manager
            cv_atks[atk_name].load_only(
                    loaders = ['train', 'test'],
                    verbose = verbose 
                    )
            cv_atk_dl[atk_name] = cv_atks[atk_name].get_dataloaders(
                    bs = bs,
                    verbose = verbose 
                    )

    quit()
            

    # TODO: no copy
    cv_train_dict = {key: corevectors for key, corevectors in cv_dl['train'].dataset['coreVectors'].items() }
    cv_val_dict = {key: corevectors for key, corevectors in cv_dl['val'].dataset['coreVectors'].items() }
    cv_test_dict = {key: corevectors for key, corevectors in cv_dl['test'].dataset['coreVectors'].items() }

    # ## Detector implementation
    Xok_train = cv_train_dict[layer][:,:dim].cpu().numpy()

    # detectors parameters
    detectors_labels = [f'OCSVM_{kernel}_{nu}', f'LOF_{h}',f'IF_{l}', 'MD']
    detectors_dict = dict(zip(detectors_labels, detectors))


    name = 'PGD'
    cvs_name = 'corevectors'
    cvs_path = f'/srv/newpenny/XAI/generated_data/corevectors_attacks={name}/{dataset}/{name_model}'


    # copy dataset to coreVect dataset
    with corevecs as cv:
        cv.load_only(
                loaders = ['test'],
                verbose = True
                ) 

        cv_dl = cv.get_dataloaders(
                batch_size = bs,
                verbose = True,
                )
        # cva_train_dict = {key: corevectors for key, corevectors in cv_dl['train'].dataset['coreVectors'].items() }
        # cva_val_dict = {key: corevectors for key, corevectors in cv_dl['val'].dataset['coreVectors'].items() }
        cva_test_dict = {key: corevectors for key, corevectors in cv_dl['test'].dataset['coreVectors'].items() }

    dim_list = [64, 128]
    attack_list = ['BIM', 'CW', 'PGD']

    for atk_name in attack_list:
        cvs_name = 'corevectors'
        cvs_path = f'/srv/newpenny/XAI/generated_data/corevectors_attacks={atk_name}/{dataset}/{name_model}'
        
        corevecs = CoreVectors(
                path = cvs_path,
                name = cvs_name,
                model = model,
                device = device
                )
        
        # copy dataset to coreVect dataset
        with corevecs as cv:
            cv.load_only(
                    loaders = ['test'],
                    verbose = True
                    ) 
        
            cv_dl = cv.get_dataloaders(
                    batch_size = bs,
                    verbose = True,
                    )
            cva_test_dict = {key: corevectors for key, corevectors in cv_dl['test'].dataset['coreVectors'].items() }
            
        for dim in dim_list:
            path = Path.cwd()/f'../data/detectors_results/dim={dim}'
            os.makedirs(path, exist_ok=True)
            file_path = os.path.join(path, f'scores_detectors_{layer}_{atk_name}_defaultConfig.png')
                
            for layer in target_layers:
                Xok_train = cv_train_dict[layer][:,:dim].cpu().numpy()
        
                Xko_test = (cva_test_dict[layer][:,:dim]-means[layer][:dim])/stds[layer][:dim]
                Xok_test = cv_test_dict[layer][:,:dim].cpu().numpy()
                
                X_test = np.concatenate([Xok_test, Xko_test])
                fig, ax = plt.subplots(len(detectors_dict.items()), sharex=True,
                               figsize=(8, 1.5*(len(detectors_dict.items()))))
                for name, detector, ax_ in zip(detectors_dict.keys(), detectors_dict.values(), ax):
                    scores = detector.fit(Xok_train).score(X_test)
                    metric_value = detector.test(X_test, metric)
                    ax_.plot(scores, label=f'{name}: {metric}={np.round(metric_value, 3)}')
                    ax_.legend()
                    ax_.grid()
                    ax_.set(ylabel='score')
                fig.tight_layout()
                fig.savefig(file_path)


    dim_list = [32, 64]
    metric_list = [
                   # 'P_D',
                   'AUC'
                  ]

    attack_list = [
                   'BIM', 
                   'DeepFool',
                   'CW', 
                   'PGD'
                    ]

    metric = 'AUC' #P_D, AUC

    target_layers = [
                'features.14',
                'features.28',
                'classifier.0',
                'classifier.3',
                # 'features.7',
                ]

    dict_score = {}

    for dim in dim_list:
        for metric in metric_list:
            results = []

            for atk_name in attack_list:
                cvs_name = 'corevectors'
                cvs_path = f'/srv/newpenny/XAI/generated_data/corevectors_attacks={atk_name}/{dataset}/{name_model}'
                
                corevecs = CoreVectors(
                        path = cvs_path,
                        name = cvs_name,
                        model = model,
                        device = device
                        )
                
                # copy dataset to coreVect dataset
                with corevecs as cv:
                    cv.load_only(
                            loaders = ['test'],
                            verbose = True
                            ) 
                
                    cv_dl = cv.get_dataloaders(
                            batch_size = bs,
                            verbose = True,
                            )
                    cva_test_dict = {key: corevectors for key, corevectors in cv_dl['test'].dataset['coreVectors'].items() }
                    
                # for dim in dim_list:
                #     path = Path.cwd()/f'../data/detectors_results/dim={dim}'
                #     os.makedirs(path, exist_ok=True)
                #     file_path = os.path.join(path, f'scores_detectors_{layer}_{atk_name}_defaultConfig.png')
                        
                for layer in target_layers:
                    Xok_train = cv_train_dict[layer][:,:dim].cpu().numpy()
            
                    Xko_test = (cva_test_dict[layer][:,:dim]-means[layer][:dim])/stds[layer][:dim]
                    Xok_test = cv_test_dict[layer][:,:dim].cpu().numpy()
                    
                    X_test = np.concatenate([Xok_test, Xko_test])
                    
                    for name, detector in zip(detectors_dict.keys(), detectors_dict.values()):
                        scores = detector.fit(Xok_train).score(X_test)
                        scores_ok = detector.fit(Xok_train).score(Xok_test)
                        scores_ko = detector.fit(Xok_train).score(Xko_test.cpu().numpy())
                        
                        dict_score[(name, dim, layer, atk_name)] = (scores_ok, scores_ko)
                        metric_value = detector.test(X_test, metric)
                        results.append({
                            "METHOD": name,
                            "LAYER": layer,
                            "ATTACK": atk_name,
                            "METRIC": np.round(metric_value, 3)  # Save the rounded metric value
                        })
            df = pd.DataFrame(results)
            df_pivot = df.pivot_table(
                index=["METHOD", "LAYER"],  # Rows
                columns="ATTACK",          # Columns
                values="METRIC",           # Values
                aggfunc='first'            # In case of duplicates, take the first value
            ).reset_index()
           
            df_pivot.to_csv(f"metric={metric}_dim={dim}.csv", index=False)


    dict_score.keys()
    df = pd.DataFrame(results)
    df_pivot = df.pivot_table(
        index=["METHOD", "LAYER"],  # Rows
        columns="ATTACK",          # Columns
        values="METRIC",           # Values
        aggfunc='first'            # In case of duplicates, take the first value
    ).reset_index()

    layer_order = ["features.14", "features.28", "classifier.0", "classifier.3"]
    df_pivot["LAYER"] = pd.Categorical(
        df_pivot["LAYER"],
        categories=layer_order,
        ordered=True
    )

    # Now sort by METHOD, then LAYER
    df_pivot = df_pivot.sort_values(["METHOD", "LAYER"])
    df_pivot.style.background_gradient(cmap ='viridis').set_properties(**{'font-size': '20px'})
    df_pivot.to_csv(f"metric={metric}_dim={dim}_n.csv", index=False)


    # ## Visualization of dataframe
    metric = 'AUC'
    dim = 16
    df = pd.read_csv(f'metric={metric}_dim={dim}.csv')
    df.style.background_gradient(cmap ='viridis').set_properties(**{'font-size': '20px'})

import sys
sys.path.insert(0, '/home/leandro/repos/peepholelib')

# python stuff
from pathlib import Path
from contextlib import ExitStack
import numpy as np
import pandas

# Our stuff
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.utils.samplers import dist_preserving as dpss 

# torch stuff
from cuda_selector import auto_cuda
import torch
from torch.utils.data import DataLoader

# tensordict stuff
from tensordict import MemoryMappedTensor as MMT 
from tensordict import TensorDict as TD 
from tensordict import PersistentTensorDict as PTD

# KDE stuff
from sklearn.neighbors import KernelDensity

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    dataset = 'CIFAR100' 
    name_model = 'vgg16'
    verbose = True 
    cv_size = 100 

    cvs_name = 'corevectors'
    out_name = 'output'
    cvs_path = Path(f'/srv/newpenny/XAI/generated_data/corevectors/{dataset}/{name_model}')
    cvs_atks_home = Path('/srv/newpenny/XAI/generated_data')

    cv_datasets_path = Path.cwd()/f'../data/cv_datasets'
    cv_datasets_path.mkdir(parents=True, exist_ok=True)

    attack_names = ['BIM', 'CW', 'PGD', 'DeepFool']
    labels = torch.linspace(0, 100-1, 100, dtype=torch.int).numpy().tolist()

    kde_params = {
            'kernel': 'gaussian',
            'bandwidth': 'silverman',
            }

    #--------------------------------
    # Layers selection 
    #--------------------------------
    target_layers = [
            'classifier.0',
            'classifier.3',
            'features.14',
            'features.28'
            ]   
    
    #--------------------------------
    # CoreVectors 
    #--------------------------------
    cv = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            device = device
            )
    
    out = CoreVectors(
            path = cvs_path,
            name = out_name,
            verbose = verbose
            )

    cv_atks = {} 
    out_atks = {}
    for atk_name in attack_names:
        cv_atk_path = cvs_atks_home/f'corevectors_attacks=my{atk_name}/{dataset}/{name_model}'
        cv_atks[atk_name] = CoreVectors(
                path = cv_atk_path,
                name = cvs_name,
                verbose = verbose
                ) 

        out_atks[atk_name] = CoreVectors(
                path = cv_atk_path,
                name = out_name,
                verbose = verbose
                ) 
    
    # store results
    _res_layer = []
    _res_auc  = []
    _res_atk = []

    # load original CVs 
    with cv, out:
        cv.load_only(
            loaders = ['train', 'test'],
            verbose = verbose 
            )

        out.load_only(
            loaders = ['train', 'test'],
            verbose = verbose 
            ) 

        # iterate for each layer
        for layer in target_layers:

            # compute labes distribution
            _ds = cv._corevds['train']
            
            # filtering cvs per label
            if verbose: print(f'------\n fitting detector for layer {layer}\n------')
            avg, std = {}, {}
            for _l in labels:
                __ds = _ds[_ds['label'] == _l]
                train_data = __ds['coreVectors'][layer][:, :cv_size].detach()
                avg[_l] =  train_data.mean(dim=0)
                std[_l] =  train_data.var(dim=0)
               
            # testing
            for _atk_name in attack_names:
                with cv_atks[_atk_name] as cv_atk, out_atks[_atk_name] as out_atk:
                    if verbose: print(f'\n---------\nLoading dataset for attack: {_atk_name}')

                    cv_atk.load_only(
                            loaders = ['test'],
                            verbose = verbose 
                            )

                    out_atk.load_only(
                            loaders = ['test'],
                            verbose = verbose 
                            )
                    
                    if verbose: print(f'computing AUC for {_atk_name} attacked test samples')

                    # Get data from original and attacked samples 
                    idx = (cv_atk._corevds['test']['attack_success']==1) & (cv._corevds['test']['result']==1)
                    data_ori = cv._corevds['test'][idx]['coreVectors'][layer][:,:cv_size]
                    data_atk = cv_atk._corevds['test'][idx]['coreVectors'][layer][:,:cv_size]
                    test_data = torch.cat([data_ori, data_atk]) 

                    # Get labels for original and attacked samples
                    labels_ori = cv._corevds['test'][idx]['label']
                    labels_atk = cv_atk._corevds['test'][idx]['pred']
                    test_labels = torch.cat([labels_ori, labels_atk])
                    print('test labels: ', test_labels)

                    probs = torch.zeros(test_data.shape[0], len(labels))
                    for i, _avg in enumerate(avg.values()):
                        probs[:, i] = (test_data-_avg).norm(dim=1)

                    print(f'probs: ', probs) 
                    #probs /= probs.sum(dim=1, keepdim=True)
                    #print(f'sum: ', probs.sum(dim=1))
                    on_labels = probs[torch.linspace(0, len(test_labels)-1, len(test_labels)).int(), test_labels.int()]
                    minimum, _ = probs.min(dim=1)
                    print(on_labels, minimum)
                    nz = (on_labels==minimum).count_nonzero()
                    auc = nz/len(test_labels)
                    print(f"{nz}/{len(test_labels)} = {auc}")
                    # TODO: get output for idx samples and multiply by probs

                    _res_layer.append(layer)
                    _res_auc.append(auc)
                    _res_atk.append(_atk_name)
    _res_layer, _res_auc, _res_atk = np.array(_res_layer), np.array(_res_auc), np.array(_res_atk) 
    df = pandas.DataFrame({
        'layer': _res_layer,
        'auc': _res_auc,
        'atk': _res_atk,
        })
    df.to_pickle((Path.cwd()/'aa.pk').as_posix())

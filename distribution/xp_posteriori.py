import sys
sys.path.insert(0, '/home/leandro/repos/peepholelib')

# python stuff
from pathlib import Path
from contextlib import ExitStack

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
from torchkde import KernelDensity

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'
    
    dataset = 'CIFAR100' 
    name_model = 'vgg16'
    verbose = True 
    cv_dim = 300

    cvs_name = 'corevectors'
    out_name = 'output'
    cvs_path = Path(f'/srv/newpenny/XAI/generated_data/corevectors/{dataset}/{name_model}')
    cvs_atks_home = Path('/srv/newpenny/XAI/generated_data')

    cv_datasets_path = Path.cwd()/f'../data/cv_datasets'
    cv_datasets_path.mkdir(parents=True, exist_ok=True)

    attack_names = ['BIM', 'CW', 'PGD', 'DeepFool']
    
    config = {
            'kernel': 'gaussian',
            'bandwidth': 1.0,
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
        cv_atk_path = cvs_atks_home+f'/corevectors_attacks=my{atk_name}/{dataset}/{name_model}'
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
            train_data = cv._corevds['train']['corevectors'][layer][:, :cv_size] 
            
            kde = KernelDensity(**kde_params)
            if verbose: print(f'------\n fitting detector {detector}\n------')
            _ = kde.fit(train_data)
            
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
                    
                    if verbose: print(f'computing {metric_type} for {_atk_name} attacked test samples')

                    # TODO: review this with Lorenzo
                    idx_test = torch.argwhere(((cv_atk._corevds['test']['attack_success']==1) & (cv._corevds['test']['result']==1)))#.squeeze().detach().cpu().numpy()
                    data_ori = out._corevds['test']['coreVectors'][_layer][idx_test,:cv_size]
                    data_atk = out_atk._corevds['test']['coreVectors'][_layer][idx_test,:cv_size]
                    
                    test_data = torch.cat([data_ori, data_atk]) 
                    logprog = kde.score_samples(test_data) 





        
        

    

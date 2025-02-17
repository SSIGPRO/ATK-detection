import sys
sys.path.insert(0, '/home/leandro/repos/peepholelib')

# python stuff
import os
sys.path.append('..')
from pathlib import Path
from paretoset import paretoset
import functools
from time import time

# torch stuff
import torch
from torch.utils.data import DataLoader
from cuda_selector import auto_cuda

# tensordict stuff
from tensordict import PersistentTensorDict as PTD

# Our stuff
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.utils.samplers import dist_preserving 
from estimators.gp import GPModel, parser_fn 

def gp_wrap(config, **kwargs):
    cv_size = config.pop('cv_size')
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    cv = kwargs['cv'] 
    testsets = kwargs['testsets']
    max_epochs = kwargs['max_epochs']

    #--------------------------------
    # dataloaders 
    #--------------------------------
    collate_fn = functools.partial(parser_fn, cv_size=cv_size)

    with cv:
        cv.load_only(
                loaders = ['train'],
                verbose = verbose 
                )

        print(dist_preserving)
        ds, _ = dist_preserving(cv._corevds['train'], ss, weights='label')
        print('_key: ', len(ds), ' samples')

        cv_dl = DataLoader(
            ds,
            batch_size = bs,
            shuffle = True,
            collate_fn = collate_fn,
            num_workers = 4,
            pin_memory = True,
            )
        x, y = next(iter(cv_dl))
        x, y = x.detach(), y.detach()

    testloaders = {}
    for _k, _d in testsets.items():
        testloaders[_k] = DataLoader(
            _d.detach(),
            batch_size = bs,
            shuffle = False,
            collate_fn = collate_fn,
            num_workers = 4,
            pin_memory = True,
            )
                                     
    #--------------------------------
    # Create Discriminator 
    #--------------------------------
    if verbose: print('Creating Discriminator')
    model = GPModel(
            x = x, 
            y = y,
            **config,
            device = device,
            )

    #--------------------------------
    # Computation 
    #--------------------------------
    n_params = model.num_parameters
    for epoch in range(max_epochs): 
        loss = discriminator.train_iteration()
        if verbose: print("epoch: ", epoch, ' - loss: ', loss)
                
    return loss

if __name__ == '__main__':
    #--------------------------------
    # Directories definitions
    #--------------------------------
    name_model = 'vgg16'
    dataset = 'CIFAR100' 
    verbose = True 
    
    cvs_name = 'corevectors'
    cvs_path = f'/srv/newpenny/XAI/generated_data/corevectors/{dataset}/{name_model}'

    #results_home = Path('/srv/newpenny/atk-detection/results/gp')
    results_home = Path.cwd()/'../data/results/gp'
    results_path = results_home/'tuning_results'
    results_path.mkdir(exist_ok=True, parents=True)

    datasets_path = Path(f'/srv/newpenny/XAI/generated_data/cv_datasets')
    
    atk_list = ['PGD', 'BIM', 'CW', 'DeepFool']

    # Tuning defs
    resources = {'cpu': 16, 'gpu':1}
    max_cv_size = 300 
    max_epochs = 1001
    num_samples = 1
    checkpoint_every = 50
    max_concurrent = 1
    
    #--------------------------------
    # CoreVectors 
    #--------------------------------
    cv = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            verbose = verbose
            )

    #--------------------------------
    # Tune configurations 
    #--------------------------------
    config = {
            'cv_size': tune.randint(5, max_cv_size+1),
            'lr': tune.uniform(1e-2, 1),
            'kernel_kwargs': {
                'nu': tune.loguniform(1e-6, 1e-4),
                'power': tune.randint(1, 5+1),
                },
            'perc': tune.uniform(0.01, 0.2),
            }
    
    #--------------------------------
    # Testsets 
    #--------------------------------
    testsets = {}
    for atk in atk_list:
        _f = datasets_path/f'test_only={atk}'
        testsets[atk]  = PTD.from_h5(_f, mode='r')
    testsets['all'] = PTD.from_h5(datasets_path/'test_all', mode='r')
    
    #--------------------------------
    # Iterate tunning all configurations 
    #--------------------------------
    print('\n------------------')
    print('Loading Datasets')
    print('------------------\n')
    for atk in atk_list:
        save_path = results_path/f'{atk}'
        save_path.mkdir(exist_ok=True, parents=True)
        
        t0 = time()
        loss = gp_wrap(
                cv = cv,
                testsets = testsets,
                max_epochs = max_epochs,
                verbose = verbose,
                )
        
        print('time: ', time() - t0)
        print('loss: ', loss)

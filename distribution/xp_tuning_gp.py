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

# Tuner stuff
import tempfile
from functools import partial
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.train.torch import get_device as ray_get_device
import ray.cloudpickle as pickle

# Our stuff
from peepholelib.utils.samplers import dist_preserving as dpss 
from estimators.gp import GPModel, parser_fn 

def gp_wrap(config, **kwargs):
    cv_size = config.pop('cv_size')
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    cv = kwargs['cv'] 
    testsets = kwargs['testsets']
    max_epochs = kwargs['max_epochs']
    tune_path = kwargs['tune_path']
    save_path = kwargs['save_path']
    checkpoint_every = kwargs['checkpoint_every'] 
    device = ray_get_device() 

    #--------------------------------
    # dataloaders 
    #--------------------------------
    collate_fn = functools.partial(parser_fn, cv_size=cv_size)

    with cv:
        cv.load_only(
                loaders = ['train'],
                verbose = verbose 
                )

        ds, _ = dpss(cv._corevds['train'], ss, weights='label')
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

        print('x: ', x)
        print('y: ', y)

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
    # Create checkpoint 
    #--------------------------------
    checkpoint = get_checkpoint()
    if checkpoint != None:
        with checkpoint.as_directory() as checkpoint_dir:
            sd = torch.load(checkpoint_dir+'state_dict.pt')
            
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
        
        #--------------------------------
        # Save checkpoint 
        #--------------------------------
        if epoch%checkpoint_every == 0:
            with tempfile.TemporaryDirectory() as tempdir: 
                checkpoint = Checkpoint.from_directory(tempdir)
                torch.save(model.state_dict(), tempdir+'/model.pt')
                auc = {}
                for k, dl in testloaders.items():
                    auc[k] = discriminator.AUC_test(dl)

                train.report({
                    'loss_train': loss_train,
                    'loss_val': loss_val,
                    'n_params': discriminator.num_parameters,
                    **auc
                    },
                    checkpoint=checkpoint
                    )

    return

if __name__ == '__main__':
    #--------------------------------
    # Directories definitions
    #--------------------------------
    name_model = 'vgg16'
    dataset = 'CIFAR100' 
    verbose = True 
    
    cvs_name = 'corevectors'
    cvs_path = f'/srv/newpenny/XAI/generated_data/corevectors/{dataset}/{name_model}'

    results_home = Path('/srv/newpenny/atk-detection/results/gp')
    results_path = results_home/'tuning_results')
    results_path.mkdir(exist_ok=True, parents=True)

    tune_path_home = results_home/'tuning'
    tune_path_home.mkdir(exist_ok=True, parents=True)

    datasets_path = Path(f'/srv/newpenny/XAI/generated_data/cv_datasets')
    
    atk_list = ['PGD', 'BIM', 'CW', 'DeepFool']

    # Tuning defs
    resources = {'cpu': 16, 'gpu':1}
    max_cv_size = 300 
    max_epochs = 1001
    num_samples = 50 
    checkpoint_every = 50
    
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
            'lr': tune.uniform(1e-2, 1)
            'kernel_kwargs': {
                'nu': tune.loguniform(1e-6, 1e-4)
                'power': tune.randint(1, 5+1)
                },
            'perc': tune.uniform(0.01, 0.2)
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

        hyperp_file = save_path/f'hyperparams.pk'
        tune_path = tune_path_home/f'{atk}'
        tune_path.mkdir(exist_ok=True, parents=True)

        if hyperp_file.exists():
            print("Already tunned parameters found in %s. Skipping"%(hyperp_file.as_posix()))
            quit() 

        searcher = OptunaSearch(metric=['loss'], mode = ['min'])
        algo = ConcurrencyLimiter(searcher, max_concurrent=4)
        scheduler = AsyncHyperBandScheduler(grace_period=5, max_t=100, metric='loss', mode="min") 
        
        trainable = tune.with_resources(
                partial(
                    gp_wrap,
                    cv = cv,
                    testsets = testsets,
                    max_epochs = max_epochs,
                    tune_path = tune_path,
                    save_path = save_path,
                    checkpoint_every = checkpoint_every,
                    verbose = verbose,
                    ),
                resources,
                )
        
        tuner = tune.Tuner( 
                trainable,
                tune_config = tune.TuneConfig(
                    search_alg = algo,
                    num_samples = num_samples, 
                    scheduler = scheduler,
                    ),
                run_config = train.RunConfig(
                    storage_path = tune_path,
                    ),
                param_space = config,
                )
        t0 = time()
        result = tuner.fit()
        results_df = result.get_dataframe()
        if verbose: print('results: ', results_df)
        results_df.to_pickle(hyperp_file)
        print('time: ', time() - t0)

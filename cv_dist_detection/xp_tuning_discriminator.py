import sys
sys.path.insert(0, '/home/leandro/repos/peepholelib')

# python stuff
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
from detectors.discriminator import Discriminator
from detectors.discriminator import parser_fn

def discriminator_wrap(config, **kwargs):
    cv_size = config.pop('cv_size')
    bs = config.pop('batch_size')
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    datasets = kwargs['datasets'] 
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
    dataloaders = {}
    for _k, _d in datasets.items():
        dataloaders[_k] = DataLoader(
            _d.detach(),
            batch_size = bs,
            shuffle = True,
            collate_fn = collate_fn,
            num_workers = 4,
            pin_memory = True,
            )

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
    discriminator = Discriminator(
            **config,
            dataloaders = dataloaders,
            save_path = save_path,
            device = device,
            )

    #--------------------------------
    # Computation 
    #--------------------------------
    n_params = discriminator.num_parameters
    for epoch in range(max_epochs): 
        loss_train, loss_val = discriminator.train_epoch()
        if verbose: print("epoch: ", epoch, ' - train loss: ', loss_train, ' - val loss: ', loss_val)
        
        #--------------------------------
        # Save checkpoint 
        #--------------------------------
        if epoch%checkpoint_every == 0:
            with tempfile.TemporaryDirectory() as tempdir: 
                checkpoint = Checkpoint.from_directory(tempdir)
                torch.save(discriminator.state_dict(), tempdir+'/model.pt')
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
    
    results_path = Path.cwd()/'results/discriminator/tuning_results'
    results_path.mkdir(exist_ok=True, parents=True)

    tune_path_home = Path.cwd()/'results/discriminator/tuning'
    tune_path_home.mkdir(exist_ok=True, parents=True)

    datasets_path = Path(f'/srv/newpenny/XAI/generated_data/cv_datasets')
    
    attack_configs = {
                      'c0': {'train': ['PGD','BIM','CW'], 'test': 'DeepFool'},
                      'c1': {'train': ['BIM','CW','DeepFool'], 'test': 'PGD'},
                      'c2': {'train': ['CW','DeepFool','PGD'], 'test': 'BIM'},
                      'c3': {'train': ['DeepFool','PGD','BIM'], 'test': 'CW'}
                     }

    atk_list = ['PGD', 'BIM', 'CW', 'DeepFool']

    resources = {'cpu': 16, 'gpu':1}
    max_cv_size = 300 
    max_epochs = 1001
    num_samples = 50 
    checkpoint_every = 50

    #--------------------------------
    # Tune configurations 
    #--------------------------------
    config = {
            'n_layers': tune.randint(8, 32),
            'layer_size': tune.randint(8, 32),
            'optim_kwargs': {'lr': tune.loguniform(1e-6, 1e-4)},
            'loss_kwargs': {'reduction': tune.choice(['sum'])},
            'cv_size': tune.randint(5, max_cv_size+1),
            'batch_size': tune.choice([2**i for i in range(6, 10)]), # 64 to 512
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
    for _c in attack_configs.values():
        suffix = f'train={_c['train']}_test={_c['test']}'

        datasets = {}
        for _p in ['train', 'val']:
            _f = datasets_path/suffix/f'{_p}'
            datasets[_p] = PTD.from_h5(_f, mode='r')
        
        save_path = results_path/f'{suffix}'
        save_path.mkdir(exist_ok=True, parents=True)

        hyperp_file = save_path/f'hyperparams.pk'
        tune_path = tune_path_home/f'{suffix}'
        tune_path.mkdir(exist_ok=True, parents=True)

        if hyperp_file.exists():
            print("Already tunned parameters fount in %s. Skipping"%(hyperp_file.as_posix()))
            quit() 

        searcher = OptunaSearch(metric=['loss_val'], mode = ['min'])
        algo = ConcurrencyLimiter(searcher, max_concurrent=4)
        scheduler = AsyncHyperBandScheduler(grace_period=5, max_t=100, metric='loss_val', mode="min") 
        
        trainable = tune.with_resources(
                partial(
                    discriminator_wrap,
                    datasets = datasets,
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

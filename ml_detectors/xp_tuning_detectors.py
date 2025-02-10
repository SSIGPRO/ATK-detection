import sys
sys.path.insert(0, '/home/leandro/repos/peepholelib')

# python stuff
import os
sys.path.append('..')
import pandas 
import pickle
from pathlib import Path
from contextlib import ExitStack
from paretoset import paretoset

# torch stuff
import torch

# detectors stuff
from detectors.ml_based import OCSVM, LOF, IF  
from detectors.gaussian_distribution_based import MD 

# Corevectors 
from peepholelib.coreVectors.coreVectors import CoreVectors 

# Tuner
import tempfile
from functools import partial
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.optuna import OptunaSearch
import ray.cloudpickle as pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['SCIPY_USE_PROPACK'] = "True"
threads = "64"
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads

def detector_wrap(config, **kwargs):
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    cv = kwargs['cv']
    cv_atks = kwargs['cv_atks']
    layer = kwargs['layer']
    tune_path = kwargs['tune_path']

    cv_size = config.pop('cv_size')
    
    checkpoint = get_checkpoint()
    if checkpoint != None:
        with checkpoint.as_directory() as checkpoint_dir:
            #TODO

    #--------------------------------
    # Create detector 
    #--------------------------------
    if verbose: print('Creating detectors')
    print('Detectors: ', OCSVM, LOF, IF, MD)
    detector = eval(_det_name)(**config)

    #--------------------------------
    # Computation 
    #--------------------------------

    # fit detector
    with cv:
        # get corevectors from the original dataset 
        cv.load_only(
                loaders = ['train', 'test'],
                verbose = verbose 
                ) 

        if verbose: print(f'------fitting detector {detector}------')
        detector.fit(cv._corevds['train']['coreVectors'][layer][:,:cv_size])

        aucs = torch.zeros(len(cv_atks))
        i = 0
        for _atk_name in cv_atks:
            if verbose: print(f'---------Loading dataset for attack: {_atk_name}')
            
            # get corevectors from atk dataset 
            with cv_atks[_atk_name] as cv_atk:
                cv_atk.load_only(
                        loaders = ['test'],
                        verbose = verbose 
                        )
                 
                if verbose: print(f'computing {metric_type} for {_atk_name} attacked test samples')
                data_ori = cv._corevds['test']['coreVectors'][layer][:,:cv_size]
                data_atk = cv_atk._corevds['test']['coreVectors'][layer][:,:cv_size]

                aucs[i] = detector.test(data_ori, data_atk, metric_type) 
    
    #--------------------------------
    # Save checkpoint 
    #--------------------------------
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        pd = ((aucs.mean() - 0.5).abs() + 0.5).item()
        train.report({'pd': pd}, checkpoint=checkpoint)

    return

if __name__ == '__main__':
    #--------------------------------
    # Directories definitions
    #--------------------------------
    name_model = 'vgg16'
    dataset = 'CIFAR100' 
    verbose = True
    
    results_path = Path.cwd()/'results/ml/tuning_results'
    results_path.mkdir(exist_ok=True, parents=True)
    tune_path_home = Path.cwd()/'results/ml/tuning'
    tune_path_home.mkdir(exist_ok=True, parents=True)

    cvs_name = 'output'
    cvs_path = f'/srv/newpenny/XAI/generated_data/corevectors/{dataset}/{name_model}'
    cvs_atks_home = f'/srv/newpenny/XAI/generated_data/'
    max_cv_size = 100 

    verbose = True
    attack_names = ['BIM', 'CW', 'PGD', 'DeepFool']
    metric_type = 'AUC'#, P_D
    
    target_layers = ['data']

    resources = {'cpu': 64, 'gpu':2}
    num_samples = 50 

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    cv = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            verbose = verbose
            )
    
    cv_atks = {} 
    for _atk_name in attack_names:
        cv_atk_path = cvs_atks_home+f'/corevectors_attacks=my{_atk_name}/{dataset}/{name_model}'
        cv_atks[_atk_name] = CoreVectors(
                path = cv_atk_path,
                name = cvs_name,
                verbose = verbose
                ) 
        
    #--------------------------------
    # Tune configurations 
    #--------------------------------
    detector_configs = {
            'OCSVM': {
                'cv_size': 100,
                # 'cv_size': tune.randint(5, max_cv_size+1), 
                'kernel': tune.choice(['linear', 'poly', 'rbf', 'sigmoid']),
                'nu': tune.uniform(0.01, 1.0),
                'max_iter': 10000, # we do not want this one in the final config
                },
            'LOF': {
                'cv_size': 100,
                # 'cv_size': tune.randint(5, max_cv_size+1), 
                'algorithm': tune.choice(['ball_tree', 'kd_tree']),
                'leaf_size': tune.randint(5, 501),
                'n_neighbors': tune.randint(2, 101),
                'n_jobs': tune.choice([64]), # has to be in tune config to end up in the final congig
                },
            'IF': {
                'cv_size': 100,
                # 'cv_size': tune.randint(5, max_cv_size+1), 
                'n_estimators': tune.randint(10, 501),
                'n_jobs': tune.choice([64]),
                }
            # 'MD': {
                # 'cv_size': tune.randint(5, max_cv_size+1), 
                # }
            }
    
    #--------------------------------
    # Iterate tunning all detectors for all layers 
    #--------------------------------
    for _det_name in detector_configs:
        for _layer in target_layers:
            hyperp_file = results_path/f'hyperparams.{_det_name}.{_layer}.pk'
            tune_path = tune_path_home/f'{_det_name}.{_layer}'

            if hyperp_file.exists():
                print("Already tunned parameters fount in %s. Skipping"%(hyperp_file.as_posix()))
            
            else: 
                # get config
                config = detector_configs[_det_name] 

                searcher = OptunaSearch(metric = 'pd', mode = 'max')
                algo = ConcurrencyLimiter(searcher, max_concurrent=4)
                scheduler = AsyncHyperBandScheduler(grace_period=5, max_t=100, metric='pd', mode="max") 
                
                trainable = tune.with_resources(
                        partial(
                            detector_wrap,
                            cv = cv, 
                            cv_atks = cv_atks,
                            layer = _layer,
                            tune_path = tune_path, 
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
                result = tuner.fit()

                results_df = result.get_dataframe()
                if verbose: print('results: ', results_df)
                results_df.to_pickle(hyperp_file)

    # Convert hyperparams to a dictionary to facilatate `xp_detectors.py`
    save_configs = {}
    for _det_name in detector_configs:
        config_labels = detector_configs[_det_name].keys()
        _df_config_labels = ['config/'+k for k in config_labels]
        
        save_configs[_det_name] = {}
        for _layer in target_layers:
            if verbose: print(f'\n----------\ndetector {_det_name}, layer {_layer}\n')
            hyperp_file = results_path/f'hyperparams.{_det_name}.{_layer}.pk'
            df = pandas.read_pickle(hyperp_file)
            metrics = df.filter(['pd'])
            configs = df.filter(_df_config_labels)
            print(df)
            print(metrics)
            print(configs)
            mask = paretoset(metrics, sense=['max'])
            
            best_configs = configs.get(mask)
            best_metrics = metrics.get(mask) 
            if verbose: print('Best configs: ', best_configs)
            if verbose: print('\nBest metrics: ', best_metrics)
            
            save_configs[_det_name][_layer] = {}
            for _cl, _df_cl in zip(config_labels, _df_config_labels):
                v = best_configs.head(1)[_df_cl].values[0] 
                save_configs[_det_name][_layer][_cl] = v 

    if verbose: print('\n----------\nconfigs to save: ', save_configs)
    pickle.dump(save_configs, open(results_path/'best_configs_output.pk', 'wb'))

import sys
sys.path.insert(0, '/home/leandro/repos/peepholelib')

# python stuff
import os
sys.path.append('..')
import pandas 
from pathlib import Path
from contextlib import ExitStack

# detectors stuff
from detectors.ml_based import OCSVM, LOF, IF  
from detectors.gaussian_distribution_based import MD 

# Corevectors 
from peepholelib.coreVectors.coreVectors import CoreVectors 

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['SCIPY_USE_PROPACK'] = "True"
threads = "64"
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads

if __name__ == '__main__':
    #--------------------------------
    # Directories definitions
    #--------------------------------
    name_model = 'vgg16'
    dataset = 'CIFAR100' 
    verbose = True
    
    results_path = Path.cwd()/'../data'
    results_name = 'results_detectors.pk'

    cvs_name = 'corevectors'
    cvs_path = f'/srv/newpenny/XAI/generated_data/corevectors/{dataset}/{name_model}'
    
    cvs_atks_home = f'/srv/newpenny/XAI/generated_data/toy_case'

    verbose = True
    #attack_names = ['BIM', 'CW', 'PGD', 'DeepFool']
    attack_names = ['BIM', 'PGD']
    metric_type = 'AUC'#, P_D

    detector_configs = {
            'OCSVM': {
                'classifier.0': {'cv_size':5, 'kernel': 'rbf', 'nu': 0.01},
                'classifier.3': {'cv_size':10, 'kernel': 'poly', 'nu': 0.02},
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
    if verbose: print('Creating detectors')
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
    # saves ['metric', 'layer', 'detector', 'attack']
    _res_met = [] 
    _res_lay = [] 
    _res_det = [] 
    _res_atk = [] 
    
    for _atk_name in attack_names:
        for _det_name in detectors:
            for _layer in detectors[_det_name]:
                detector = detectors[_det_name][_layer]
                cv_size = detector_configs[_det_name][_layer]['cv_size']

                with ExitStack() as stack:
                    # get dataloader for corevectors from the original dataset 
                    stack.enter_context(cv) # enter context manager
                    cv.load_only(
                            loaders = ['train', 'test'],
                            verbose = verbose 
                            ) 
                    
                    if verbose: print(f'------\n fitting detector {detector}\n------')
                    detector.fit(cv._corevds['train']['coreVectors'][_layer][:,:cv_size])

                    # get dataloader for corevectors from atks dataset 
                    if verbose: print(f'\n---------\nLoading dataset for attack: {_atk_name}')
                    stack.enter_context(cv_atks[_atk_name]) # enter context manager
                    cv_atks[_atk_name].load_only(
                            loaders = ['train', 'test'],
                            verbose = verbose 
                            )
                     
                    if verbose: print(f'computing {metric_type} for {_atk_name} attacked test samples')
                    data_ori = cv._corevds['test']['coreVectors'][_layer][:,:cv_size]
                    data_atk = cv_atks[_atk_name]._corevds['test']['coreVectors'][_layer][:,:cv_size]

                    metric = detector.test(data_ori, data_atk, metric_type) 
                    if verbose: print(f'metric {metric_type}: ', metric)

                    # saving metric, layer, and detector configuration
                    _res_met.append(metric)
                    _res_lay.append(_layer)
                    _res_det.append(_det_name)
                    _res_atk.append(_atk_name)
    
    # Save results
    results_df = pandas.DataFrame({
        'metric':_res_met,
        'layer': _res_lay,
        'detector': _res_det,
        'attack': _res_atk 
        })

    results_path.mkdir(parents=True, exist_ok=True)
    results_df.to_pickle((results_path/results_name).as_posix())

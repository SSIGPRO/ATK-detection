import sys
sys.path.insert(0, '/home/leandro/repos/peepholelib')

# python stuff
import os
sys.path.append('..')
import pandas 
from pathlib import Path
from contextlib import ExitStack

import torch

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
    
    results_path = Path.cwd()/'results/ml'
    results_name = 'results_detectors.pk'
    
    cvs_name = 'output'
    cvs_path = f'/srv/newpenny/XAI/generated_data/corevectors/{dataset}/{name_model}'
    
    cvs_atks_home = f'/srv/newpenny/XAI/generated_data'

    verbose = True
    attack_names = ['BIM', 'CW', 'PGD', 'DeepFool']
    metric_type = 'AUC'#, P_D
    
    detector_configs = pandas.read_pickle(results_path/'tuning_results/best_configs_output.pk') 
    for k, v in detector_configs.items():
        print('\n', k)
        for kk, vv in v.items():
            print(kk)
            print(vv)
    #--------------------------------
    # Detectors 
    #--------------------------------
    if verbose: print('Creating detectors')
    detectors = {}
    for _det_name in detector_configs:
        detectors[_det_name] = {}
        for _layer in detector_configs[_det_name]:
            conf = detector_configs[_det_name][_layer].copy()
            conf.pop('cv_size')
            detectors[_det_name][_layer] = eval(_det_name)(**conf)

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    cv = CoreVectors(
            path = cvs_path,
            name = 'corevectors',
            verbose = verbose
            )
    out = CoreVectors(
            path = cvs_path,
            name = 'output',
            verbose = verbose
            )
    
    # cv_atks = {} 
    # for atk_name in attack_names:
    #     cv_atk_path = cvs_atks_home+f'/corevectors_attacks=my{atk_name}/{dataset}/{name_model}'
    #     cv_atks[atk_name] = CoreVectors(
    #             path = cv_atk_path,
    #             name = cvs_name,
    #             verbose = verbose
    #             ) 

    # #--------------------------------
    # # For saving results 
    # #--------------------------------
    # # saves ['metric', 'layer', 'detector', 'attack']
    # _res_met = [] 
    # _res_lay = [] 
    # _res_det = [] 
    # _res_atk = [] 
    
    # for _det_name in detectors:
    #     for _layer in detectors[_det_name]:
    #         detector = detectors[_det_name][_layer]
    #         cv_size = detector_configs[_det_name][_layer]['cv_size']

    #         with cv:
    #             # get dataloader for corevectors from the original dataset 
    #             cv.load_only(
    #                     loaders = ['train', 'test'],
    #                     verbose = verbose 
    #                     ) 
    #             if verbose: print(f'------\n fitting detector {detector}\n------')
    #             detector.fit(cv._corevds['train']['coreVectors'][_layer][:,:cv_size])
                
    #             for _atk_name in attack_names:
    #                 # get dataloader for corevectors from atks dataset 
                    
    #                 with cv_atks[_atk_name] as cv_atk:
    #                     if verbose: print(f'\n---------\nLoading dataset for attack: {_atk_name}')
    #                     cv_atk.load_only(
    #                             loaders = ['test'],
    #                             verbose = verbose 
    #                             )
                         
    #                     if verbose: print(f'computing {metric_type} for {_atk_name} attacked test samples')
    #                     idx_test = torch.argwhere(((cv_atk._corevds['test']['attack_success']==1) & (cv._corevds['test']['result']==1))).squeeze().detach().cpu().numpy()
    #                     data_ori = cv._corevds['test']['coreVectors'][_layer][idx_test,:cv_size]
    #                     data_atk = cv_atk._corevds['test']['coreVectors'][_layer][idx_test,:cv_size]

    #                     metric = detector.test(data_ori, data_atk, metric_type) 
    #                     if verbose: print(f'metric {metric_type}: ', metric)

    #                     # saving metric, layer, and detector configuration
    #                     _res_met.append(metric)
    #                     _res_lay.append(_layer)
    #                     _res_det.append(_det_name)
    #                     _res_atk.append(_atk_name)
    
    # # Save results
    # results_df = pandas.DataFrame({
    #     'metric':_res_met,
    #     'layer': _res_lay,
    #     'detector': _res_det,
    #     'attack': _res_atk 
    #     })

    # results_path.mkdir(parents=True, exist_ok=True)
    # results_df.to_pickle((Path('/srv/newpenny/XAI/generated_data/attacks_ijcnn/results_detectors_output_.pk')).as_posix())


    cv_atks = {} 
    out_atks = {}
    for atk_name in attack_names:
        cv_atk_path = cvs_atks_home+f'/corevectors_attacks=my{atk_name}/{dataset}/{name_model}'
        cv_atks[atk_name] = CoreVectors(
                path = cv_atk_path,
                name = 'corevectors',
                verbose = verbose
                ) 

        out_atks[atk_name] = CoreVectors(
                path = cv_atk_path,
                name = 'output',
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
    
    for _det_name in detectors:
        for _layer in detectors[_det_name]:
            detector = detectors[_det_name][_layer]
            cv_size = detector_configs[_det_name][_layer]['cv_size']

            with cv, out:
                # get dataloader for corevectors from the original dataset 
                cv.load_only(
                        loaders = ['train', 'test'],
                        verbose = verbose 
                        ) 
                out.load_only(
                        loaders = ['train', 'test'],
                        verbose = verbose 
                        ) 
                if verbose: print(f'------\n fitting detector {detector}\n------')
                detector.fit(out._corevds['train']['coreVectors'][_layer][:,:cv_size])
                
                for _atk_name in attack_names:
                    # get dataloader for corevectors from atks dataset 
                    
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
                        idx_test = torch.argwhere(((cv_atk._corevds['test']['attack_success']==1) & (cv._corevds['test']['result']==1))).squeeze().detach().cpu().numpy()
                        data_ori = out._corevds['test']['coreVectors'][_layer][idx_test,:cv_size]
                        data_atk = out_atk._corevds['test']['coreVectors'][_layer][idx_test,:cv_size]

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
    results_df.to_pickle((Path('/srv/newpenny/XAI/generated_data/attacks_ijcnn/results_detectors_output_.pk')).as_posix())
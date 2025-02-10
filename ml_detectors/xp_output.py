import sys

sys.path.insert(0, '/home/lorenzocapelli/repos/peepholelib')

# python stuff
import os
sys.path.append('..')
import pandas as pd
from pathlib import Path
from contextlib import ExitStack
from tqdm import tqdm
import numpy as np

# detectors stuff
from detectors.ml_based import OCSVM, LOF, IF  
from detectors.gaussian_distribution_based import MD 

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap 

# Attcks
import torchattacks
from peepholelib.adv_atk.attacks_base import ftd
from peepholelib.adv_atk.PGD import myPGD
from peepholelib.adv_atk.BIM import myBIM
from peepholelib.adv_atk.DeepFool import myDeepFool
from peepholelib.adv_atk.CW import myCW
from peepholelib.utils.testing import trim_dataloaders

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights
from cuda_selector import auto_cuda
from torch.utils.data import DataLoader
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT
from tensordict import PersistentTensorDict as PTD
from tensordict import TensorDict

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 1
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")
    
    ds_path = '/srv/newpenny/dataset/CIFAR100'
    name_model = 'vgg16'
    dataset = 'CIFAR100' 
    seed = 29
    bs = 32 
    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'

    verbose = True 
    
    results_path = Path.cwd()/'../data'
    results_name = 'results_detectors.pk'

    cvs_name = 'corevectors'
    cvs_path = f'/srv/newpenny/XAI/generated_data/toy_case_/corevectors/{dataset}/{name_model}'
    
    cvs_atks_home = f'/srv/newpenny/XAI/generated_data/toy_case_'

    verbose = True
    attack_names = ['BIM', 'CW', 'PGD', 'DeepFool']
    # attack_names = ['BIM', 'PGD']
    metric_type = 'P_D'#, 'AUC' 

    # detector_configs = {
    #         'OCSVM': {'kernel': 'rbf', 'nu': 0.1},
                
    #         'LOF': {'n_neighbors': 20},
                
    #         'IF': {'n_estimators':100},
                
    #         'MD': {},
    #         }

    #--------------------------------
    # Dataset 
    #--------------------------------
    
    ds = Cifar(dataset=dataset, data_path=ds_path)
    ds.load_data(
            batch_size = bs,
            data_kwargs = {'num_workers': 4, 'pin_memory': True},
            seed = seed,
            )

    ds_loaders = ds.get_dataset_loaders()
    loaders = {
              'train': ds_loaders['train'],
              'test': ds_loaders['test']
              }
    # loaders = trim_dataloaders(ds_loaders, 0.02)

    #--------------------------------
    # Model 
    #--------------------------------
    
    nn = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    in_features = 4096
    num_classes = len(ds.get_classes()) 
    nn.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    model = ModelWrap(device=device)
    model.set_model(model=nn, path=model_dir, name=model_name, verbose=True)
    
    #--------------------------------
    # Output extraction 
    #--------------------------------

    td = TensorDict()
    for key, dl in loaders.items():
        td[key] = MMT.empty(shape=torch.Size((len(dl.dataset),100)))
        for bn, data in enumerate(tqdm(dl)):
            image, _ = data
            n_in = len(image)
            image = image.to(device)
            output = model._model(image).detach().cpu()
            td[key][bn*bs:bn*bs+n_in] = output

    mean_vec = td['train'].mean(dim=0, keepdim=True)
    std_vec = td['train'].std(dim=0, keepdim=True)
    print(mean_vec.shape)
    
    file_norm = Path(f'/srv/newpenny/XAI/generated_data/corevectors/{dataset}/{name_model}/output.normalization.pt')
    if not file_norm.exists():
        torch.save((mean_vec,std_vec), file_norm)

    for key in td.keys():
        data = (td[key]-mean_vec)/std_vec
        file_path = f'/srv/newpenny/XAI/generated_data/corevectors/{dataset}/{name_model}/output.{key}'
        output = PTD(filename=file_path, batch_size=[td[key].shape[0]], mode='w')
        n_samples = td[key].shape[0]
        output['coreVectors'] = TensorDict(batch_size=td[key].shape[0])
        output['coreVectors']['data'] = MMT.zeros(shape=(td[key].shape))
        output['coreVectors']['labels'] = MMT.zeros(shape=(n_samples,))
        output['coreVectors']['data'] = data
        print(output['coreVectors']['data'], td[key])
        
    
    atcks = {
         'myPGD':
                 {'model': model._model,
                  'eps' : 8/255, 
                  'alpha' : 2/255, 
                  'steps' : 10,
                  'device' : device,
                  'path' : '/srv/newpenny/XAI/generated_data/attacks/PGD',
                  'name' : 'PGD',
                  'dl' : loaders,
                  'name_model' : name_model,
                  'verbose' : True,
                  'mode' : 'random',},
         'myBIM': 
                 {'model': model._model,
                  'eps' : 8/255, 
                  'alpha' : 2/255, 
                  'steps' : 10,
                  'device' : device,
                  'path' : '/srv/newpenny/XAI/generated_data/attacks/BIM',
                  'name' : 'BIM',
                  'dl' : loaders,
                  'name_model' : name_model,
                  'verbose' : True,
                  'mode' : 'random',},
         'myCW':{
                  'model': model._model,
                  'device' : device,
                  'path' : '/srv/newpenny/XAI/generated_data/attacks/CW',
                  'name' : 'CW',
                  'dl' : loaders,
                  'name_model' : name_model,
                  'verbose' : True,
                  'nb_classes' : ds.config['num_classes'],
                  'confidence': 0,
                  'c_range': (1e-3, 1e10),
                  'max_steps': 1000,
                  'optimizer_lr': 1e-2,
                  'verbose': True,},
         'myDeepFool':{
                       'model': model._model,
                        'steps' : 50,
                        'overshoot' : 0.02,
                        'device' : device,
                        'path' : '/srv/newpenny/XAI/generated_data/attacks/DeepFool',
                        'name' : 'DeepFool',
                        'dl' : loaders,
                        'name_model' : name_model,
                        'verbose' : True,
                        }
                  }
        
    for atk_type, kwargs in atcks.items():
        if verbose: print(f'ATTACK {atk_type}')
        atk = eval(atk_type)(**kwargs)

        if not atk.atk_path.exists():
            atk.get_ds_attack()
        
        atk_loaders = {key: DataLoader(value, batch_size=bs, collate_fn = lambda x: x, shuffle=False) for key, value in atk._atkds.items()}
        
        ta_ = MMT.empty(shape=torch.Size((10000,100)))
        for bn, data in enumerate(tqdm(atk_loaders['test'])):
            image = data['image']
            n_in = len(image)
            image = image.to(device)
            output = model._model(image).detach().cpu()
            output = (output-mean_vec)/std_vec
            ta_[bn*bs:bn*bs+n_in] = output
        print(ta_)
        
        file_path = f'/srv/newpenny/XAI/generated_data/corevectors_attacks={atk_type}/{dataset}/{name_model}/output.{key}.pt'
        ta = PTD(filename=file_path, batch_size=(10000,), mode='w')
        ta['coreVectors'] = TensorDict(batch_size=10000)
        ta['coreVectors']['data'] = ta_
        
        ta['coreVectors']['labels'] = MMT.ones(shape=(10000,))
        print('AGGIORNAMENTO')
        print(ta, ta['coreVectors']['data'], ta['coreVectors']['labels'])
        
        
        # for bn, data in enumerate(tqdm(atk_loaders['test'])):
           
        #     image = data['image']
        #     n_in = len(image)
        #     image = image.to(device)
        #     output = model._model(image).detach().cpu()
        #     print('corevectors prima dell assegnamento')
        #     print(ta['coreVectors']['data'][bn*bs:bn*bs+n_in])
            
        #     output = (output-mean_vec)/std_vec
        #     ta['coreVectors']['data'][bn*bs:bn*bs+n_in] = output.to(device)
        #     print(output.shape, output)
        #     print('corevectors dopo assegnamento')
        #     print(ta['coreVectors']['data'][bn*bs:bn*bs+n_in], ta['coreVectors']['data'][bn*bs:bn*bs+n_in].shape)
        #     quit()
        # print(ta['coreVectors']['labels'])
    #     n_samples = td[key].shape[0]
    #     ta['coreVectors'] = TensorDict(batch_size=td[key].shape[0])
        
    #     output['coreVectors']['data'] = td[key]


        
    #     results[atk_type] = {}
    #     if verbose: print(f'ATTACK {atk_type}')
    #     atk = eval(atk_type)(**kwargs)

    #     if not atk.atk_path.exists():
    #         atk.get_ds_attack()
        
    #     atk_loaders = {key: DataLoader(value, batch_size=bs, collate_fn = lambda x: x, shuffle=False) for key, value in atk._atkds.items()}
        
    #     ta[atk_type] = MMT.empty(shape=torch.Size((len(atk_loaders['test'].dataset),100)))
    #     for bn, data in enumerate(tqdm(atk_loaders['test'])):
           
    #         image = data['image']
    #         n_in = len(image)
    #         image = image.to(device)
    #         output = model._model(image).detach().cpu()
    #         ta[atk_type][bn*bs:bn*bs+n_in] = output
               
    #     data_ori = td['test']
    #     data_atk = ta[atk_type]
    #     for key, detector in detectors.items():
    #         # if key == 'MD':
    #         #     data_train = (td['train']-mean_single)/std_single
    #         #     data_ori = (data_ori-mean_single)/std_single
    #         #     data_atk = (data_atk-mean_single)/std_single   
    #         # else:
            
    #         data_train = (td['train']-mean_vec)/std_vec
    #         data_ori = (data_ori-mean_vec)/std_vec
    #         data_atk = (data_atk-mean_vec)/std_vec
            

    # #--------------------------------
    # # Detectors 
    # #--------------------------------
    # # if verbose: print('Creating detectors')
    # # detectors = {}
    # # for _det_name in detector_configs:
    # #     conf = detector_configs[_det_name].copy()

    # #     detectors[_det_name] = eval(_det_name)(**conf)

    # # print(detectors)

    # #--------------------------------
    # # Attack extraction 
    # #--------------------------------

    # atcks = {
    #          'myPGD':
    #                  {'model': model._model,
    #                   'eps' : 8/255, 
    #                   'alpha' : 2/255, 
    #                   'steps' : 10,
    #                   'device' : device,
    #                   'path' : '/srv/newpenny/XAI/generated_data/attacks/PGD',
    #                   'name' : 'PGD',
    #                   'dl' : loaders,
    #                   'name_model' : name_model,
    #                   'verbose' : True,
    #                   'mode' : 'random',},
    #          'myBIM': 
    #                  {'model': model._model,
    #                   'eps' : 8/255, 
    #                   'alpha' : 2/255, 
    #                   'steps' : 10,
    #                   'device' : device,
    #                   'path' : '/srv/newpenny/XAI/generated_data/attacks/BIM',
    #                   'name' : 'BIM',
    #                   'dl' : loaders,
    #                   'name_model' : name_model,
    #                   'verbose' : True,
    #                   'mode' : 'random',},
    #          'myCW':{
    #                   'model': model._model,
    #                   'device' : device,
    #                   'path' : '/srv/newpenny/XAI/generated_data/attacks/CW',
    #                   'name' : 'CW',
    #                   'dl' : loaders,
    #                   'name_model' : name_model,
    #                   'verbose' : True,
    #                   'nb_classes' : ds.config['num_classes'],
    #                   'confidence': 0,
    #                   'c_range': (1e-3, 1e10),
    #                   'max_steps': 1000,
    #                   'optimizer_lr': 1e-2,
    #                   'verbose': True,},
    #          'myDeepFool':{
    #                        'model': model._model,
    #                         'steps' : 50,
    #                         'overshoot' : 0.02,
    #                         'device' : device,
    #                         'path' : '/srv/newpenny/XAI/generated_data/attacks/DeepFool',
    #                         'name' : 'DeepFool',
    #                         'dl' : loaders,
    #                         'name_model' : name_model,
    #                         'verbose' : True,
    #                         }
    #               }
    # ta = TensorDict()
    # results = {}
    # for atk_type, kwargs in atcks.items():
    #     results[atk_type] = {}
    #     if verbose: print(f'ATTACK {atk_type}')
    #     atk = eval(atk_type)(**kwargs)

    #     if not atk.atk_path.exists():
    #         atk.get_ds_attack()
        
    #     atk_loaders = {key: DataLoader(value, batch_size=bs, collate_fn = lambda x: x, shuffle=False) for key, value in atk._atkds.items()}
        
    #     ta[atk_type] = MMT.empty(shape=torch.Size((len(atk_loaders['test'].dataset),100)))
    #     for bn, data in enumerate(tqdm(atk_loaders['test'])):
           
    #         image = data['image']
    #         n_in = len(image)
    #         image = image.to(device)
    #         output = model._model(image).detach().cpu()
    #         ta[atk_type][bn*bs:bn*bs+n_in] = output
               
    #     data_ori = td['test']
    #     data_atk = ta[atk_type]
    #     for key, detector in detectors.items():
    #         # if key == 'MD':
    #         #     data_train = (td['train']-mean_single)/std_single
    #         #     data_ori = (data_ori-mean_single)/std_single
    #         #     data_atk = (data_atk-mean_single)/std_single   
    #         # else:
            
    #         data_train = (td['train']-mean_vec)/std_vec
    #         data_ori = (data_ori-mean_vec)/std_vec
    #         data_atk = (data_atk-mean_vec)/std_vec
            
    #         detector.fit(data_train)
    #         metric = detector.test(data_ori, data_atk, metric_type) 
    #         results[atk_type][key] = metric
    #         if verbose: print(f'{key} {metric_type}: ', metric)
    # df = pd.DataFrame.from_dict(results, orient='index')
    # df.to_csv("output.csv")

    # df_ = pd.DataFrame.from_dict(results)
    # df_.to_csv("output_.csv")
    # print(df)
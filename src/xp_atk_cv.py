# python stuff
import os
import sys
sys.path.insert(0, '/home/lorenzocapelli/repos/peepholelib/peepholelib')
import numpy as np
from pathlib import Path
from contextlib import ExitStack

# Attcks
import torchattacks
from peepholelib.adv_atk.attacks_base import ftd
from peepholelib.adv_atk.PGD import myPGD
from peepholelib.adv_atk.BIM import myBIM
from peepholelib.adv_atk.DeepFool import myDeepFool
from peepholelib.adv_atk.CW import myCW

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.coreVectors.svd_coreVectors import reduct_matrices_from_svds as parser_fn
from peepholelib.utils.testing import trim_dataloaders

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights
from cuda_selector import auto_cuda
from torch.utils.data import DataLoader

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'
    name_model = 'vgg16'
    dataset = 'CIFAR10' 
    seed = 29
    bs = 64 
    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'vgg16_pretrained=True_dataset=CIFAR10-augmented_policy=CIFAR10_seed=29.pth'

    verbose = True

    svds_path = Path.cwd()/f'../../../XAI/generated_data/svds/{dataset}/{name_model}'
    svds_name = 'svds' 
    
    cvs_path = f'/srv/newpenny/XAI/generated_data/toy_case_{dataset}/corevectors/{dataset}/{name_model}'
    cvs_name = 'corevectors'

    #--------------------------------
    # Dataset 
    #--------------------------------
    
    ds = Cifar(dataset=dataset, data_path=ds_path)
    ds.load_data(
            batch_size = bs,
            data_kwargs = {'num_workers': 4, 'pin_memory': True},
            seed = seed,
            )

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
    # Attacks
    #--------------------------------
    ds_loaders = ds.get_dataset_loaders()
    loaders = {
              'train': ds_loaders['train'],
              'test': ds_loaders['test']
              }
    loaders = trim_dataloaders(loaders, 0.2)

    atcks = {
             'myPGD':
                     {'model': model._model,
                      'eps' : 8/255, 
                      'alpha' : 2/255, 
                      'steps' : 10,
                      'device' : device,
                      'path' : f'/srv/newpenny/XAI/generated_data/toy_case_{dataset}/attacks/PGD',
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
                      'path' : f'/srv/newpenny/XAI/generated_data/toy_case_{dataset}/attacks/BIM',
                      'name' : 'BIM',
                      'dl' : loaders,
                      'name_model' : name_model,
                      'verbose' : True,
                      'mode' : 'random',},
             'myCW':{
                      'model': model._model,
                      'device' : device,
                      'path' : f'/srv/newpenny/XAI/generated_data/toy_case_{dataset}/attacks/CW',
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
                            'path' : f'/srv/newpenny/XAI/generated_data/toy_case_{dataset}/attacks/DeepFool',
                            'name' : 'DeepFool',
                            'dl' : loaders,
                            'name_model' : name_model,
                            'verbose' : True,
                            }
                  }

    atk_loaders = {}
    for atk_type, kwargs in atcks.items():
        atk = eval(atk_type)(**kwargs)

        if not atk.atk_path.exists():
            atk.get_ds_attack()
        
        atk_loaders[atk_type] = {key: DataLoader(value, batch_size=bs, collate_fn = lambda x: x, shuffle=False) for key, value in atk._atkds.items()}
            
    #--------------------------------
    # Model implementation 
    #--------------------------------
    
    target_layers = [
            # 'features.7',
            # 'features.14',
            'features.24', 
            'features.26', 
            'features.28', 
            'classifier.0',
            'classifier.3',
            ]
    model.set_target_layers(target_layers=target_layers, verbose=True)

    direction = {'save_input':True, 'save_output':False}
    model.add_hooks(**direction, verbose=False) 

    dry_img, _ = ds._train_ds.dataset[0]
    dry_img = dry_img.reshape((1,)+dry_img.shape)
    model.dry_run(x=dry_img)

    #--------------------------------
    # SVDs 
    #--------------------------------
    print('target layers: ', model.get_target_layers()) 
    model.get_svds(path=svds_path, name=svds_name, verbose=verbose)
    for k in model._svds.keys():
        for kk in model._svds[k].keys():
            print('svd shapes: ', k, kk, model._svds[k][kk].shape)  
            
    #--------------------------------
    # Corevectors original
    #--------------------------------
    
    cvs_path = f'/srv/newpenny/XAI/generated_data/toy_case_{dataset}/corevectors/{dataset}/{name_model}'
    corevecs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        model = model,
        )
    with corevecs as cv: 
        # copy dataset to coreVect dataset
        cv.get_coreVec_dataset(
            loaders = loaders, 
            verbose = verbose
            ) 
    
        cv.get_activations(
                batch_size = bs,
                loaders = loaders,
                verbose = verbose
                )
    
        cv.get_coreVectors(
                batch_size = bs,
                reduct_matrices = model._svds,
                parser = parser_fn,
                verbose = verbose
                )
    
        cv.normalize_corevectors(
                wrt='train',
                verbose=verbose,
                to_file=Path(cvs_path)/(cvs_name+'.normalization.pt')
                )
        
    #--------------------------------
    # Corevectors attacks 
    #--------------------------------
    
    for atk_type, atk_loader in atk_loaders.items():
        cvs_path_atk = f'/srv/newpenny/XAI/generated_data/toy_case_{dataset}/corevectors_attacks={atk_type}/{dataset}/{name_model}'

        corevecs = CoreVectors(
            path = cvs_path_atk,
            name = cvs_name,
            model = model,
            )

        with corevecs as cv: 
            # copy dataset to coreVect dataset
            cv.get_coreVec_dataset(
                    loaders = atk_loader, 
                    verbose = verbose,
                    parser = ftd,
                    key_list = list(atk._atkds['test'].keys())
                    ) 
    
            cv.get_activations(
                    batch_size = bs,
                    loaders = atk_loader,
                    verbose = verbose
                    )
    
            cv.get_coreVectors(
                    batch_size = bs,
                    reduct_matrices = model._svds,
                    parser = parser_fn,
                    verbose = verbose
                    )

            cv.normalize_corevectors(
                    target_layers = target_layers,
                    from_file=Path(f'/srv/newpenny/XAI/generated_data/corevectors/{dataset}/{name_model}/corevectors.normalization.pt'),
                    verbose=True
                    )

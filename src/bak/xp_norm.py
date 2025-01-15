# python stuff
from pathlib import Path as Path
from numpy.random import randint
from time import time
from tqdm import tqdm

import sys
sys.path.insert(0, '/home/lorenzocapelli/repos/peepholelib')

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.adv_atk.attacks_base import ftd
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.coreVectors.svd_coreVectors import reduct_matrices_from_svds as parser_fn
from peepholelib.utils.testing import trim_dataloaders

# torch stuff
import torch
from torch.utils.data import DataLoader
from torchvision.models import vgg16, VGG16_Weights
from cuda_selector import auto_cuda


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    # model parameters
    pretrained = True
    name_model = 'vgg16'
    dataset = 'CIFAR100' 
    seed = 29
    bs = 512 
    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
    print(Path.cwd())
    svds_path = Path.cwd()/f'../../../XAI/generated_data/svds/{dataset}/{name_model}'
    svds_name = 'svds' 
    
    cvs_path = Path.cwd()/'../data/corevectors'
    cvs_name = 'corevectors'
    
    phs_path = Path.cwd()/'../data/peepholes'
    phs_name = 'peepholes'
    
    verbose = True 
    
    #--------------------------------
    # Dataset 
    #--------------------------------

    ds = Cifar(
            data_path = ds_path,
            dataset=dataset
            )
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
    n_classes = len(ds.get_classes()) 
    nn.classifier[-1] = torch.nn.Linear(in_features, n_classes)
    model = ModelWrap(device=device)
    model.set_model(model=nn, path=model_dir, name=model_name, verbose=verbose)

    target_layers = [
            'classifier.0',
            'classifier.3',
            #'features.7',
            'features.14',
            'features.28',
            ]
    model.set_target_layers(target_layers=target_layers, verbose=verbose)

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

    ds_loaders = ds.get_dataset_loaders()
    
    #--------------------------------
    # ATTACKS
    #--------------------------------    
    
    from peepholelib.adv_atk.PGD import myPGD
    name = 'PGD'
    path = f'/srv/newpenny/XAI/generated_data/attacks/{name}'
    loader = {'train': ds_loaders['train'],
              'test': ds_loaders['test']}
    kwargs = {'model': model._model,
          'eps' : 8/255, 
          'alpha' : 2/255, 
          'steps' : 10,
          'device' : device,
          'path' : path,
          'name' : name,
          'dl' : loader,
          'name_model' : name_model,
          'verbose' : True,
          'mode' : 'random', 
    }
    atk = myPGD(**kwargs)
    
    #--------------------------------
    # CoreVectors 
    #--------------------------------
    atk_loaders = {key: DataLoader(value, batch_size=bs, collate_fn = lambda x: x, shuffle=False) for key, value in atk._atkds.items()}
    trim_loaders = trim_dataloaders(atk_loaders, 0.001)
    cvs_path_norm = Path(f'/srv/newpenny/XAI/generated_data/corevectors/{dataset}/{name_model}')
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )

    with corevecs as cv: 
        # copy dataset to coreVect dataset
        cv.get_coreVec_dataset(
                loaders = trim_loaders, 
                verbose = verbose,
                parser = ftd,
                key_list = list(atk._atkds['train'].keys())
                ) 

        cv.get_activations(
                batch_size = bs,
                loaders = trim_loaders,
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
                from_file=cvs_path_norm/(cvs_name+'.normalization.pt'),
                verbose=True
                )

        

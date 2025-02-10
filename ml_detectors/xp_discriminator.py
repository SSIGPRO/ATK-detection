import sys
sys.path.insert(0, '/home/lorenzocapelli/repos/peepholelib')

# python stuff
from tqdm import tqdm
from pathlib import Path
import functools

# Our stuff
from detectors.discriminator import Discriminator, parser_fn

# torch stuff
import torch
from cuda_selector import auto_cuda
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from tensordict import PersistentTensorDict as PTD

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    datasets_path = Path(f'/srv/newpenny/XAI/generated_data/cv_datasets')
    models_path = Path.cwd()/f'../data/models'
    n_layers = 8
    layer_size = 16
    bs = 256 
    verbose = True 
    cv_size = 32 
    portions = ['train', 'val']
    max_epochs = 2 

    #--------------------------------
    # Attacks dataset
    #--------------------------------

    attacks_config = {
                      'c0': {'train': ['PGD','BIM','CW'], 'test': 'DeepFool'},
                      'c1': {'train': ['BIM','CW','DeepFool'], 'test': 'PGD'},
                      'c2': {'train': ['CW','DeepFool','PGD'], 'test': 'BIM'},
                      'c3': {'train': ['DeepFool','PGD','BIM'], 'test': 'CW'}
                     }
    c = 'c0'

    atk_train = attacks_config[c]['train']
    atk_test = attacks_config[c]['test'] 
    atk_list = atk_train+[atk_test]

    #--------------------------------
    #  Dataloaders definition
    #--------------------------------
    print('\n------------------')
    print('Loading Datasets')
    print('------------------\n')
    collate_fn = functools.partial(parser_fn, cv_size=cv_size)
    dl_kwargs = {}
    if use_cuda:
        dl_kwargs['num_workers'] = 4
        dl_kwargs['pin_memory'] = True
    
    dataloaders ={}
    for portion in portions:
        _f = datasets_path/f'train={atk_train}_test={atk_test}/{portion}'
        _d = PTD.from_h5(_f, mode='r')
        dataloaders[portion] = DataLoader(
                _d,
                batch_size = bs,
                shuffle = True,
                collate_fn = collate_fn,
                **dl_kwargs
                )

    testloaders = {}
    for atk in atk_list:
        _f = datasets_path/f'test_only={atk}'
        _d  = PTD.from_h5(_f, mode='r')
        testloaders[atk] = DataLoader(
                _d,
                batch_size = bs,
                shuffle = False,
                collate_fn = collate_fn,
                **dl_kwargs
                )

    testloaders['all'] = DataLoader(
            PTD.from_h5(datasets_path/'test_all', mode='r'),
            batch_size = bs,
            shuffle = False,
            collate_fn = collate_fn,
            **dl_kwargs
            )

    print('\n------------------')
    print('Defining the discriminator')
    print('------------------\n')
    save_path = models_path/f'cv_size={cv_size}_atk_train={atk_train}_atk_test={atk_test}'    
    save_path.mkdir(exist_ok=True, parents=True)

    discriminator = Discriminator(
        device = device,
        n_layers = n_layers,
        layer_size = layer_size,
        dataloaders = dataloaders,
        optim_type = optim.Adam,
        optim_kwargs = {'lr': 1e-3},
        scheduler_kwargs = {'early_stopping_patience': 10, 'patience': 5, 'factor': 0.1 },
        loss_type = nn.BCELoss,
        loss_kwargs = {'reduction': 'sum'},
        save_path = save_path,
        verbose = verbose,
        )  
    
    print('\n------------------')
    print('Training')
    print('------------------\n')
    for num_epoch in range(max_epochs):
        train_loss, val_loss = discriminator.train_epoch()
        print('train, val losses: ', train_loss, val_loss)
        if discriminator.stop_training:
            break

    print('\n------------------')
    print('Computing AUCs')
    print('------------------\n')
    for key, dl in testloaders.items():
        auc = discriminator.AUC_test(dl)
        print(f'{key} AUC: {auc}')
    

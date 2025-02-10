import sys
sys.path.insert(0, '/home/lorenzocapelli/repos/peepholelib')

# python stuff
from tqdm import tqdm
from pathlib import Path
from contextlib import ExitStack
import functools

# math stuff
from sklearn.metrics import roc_auc_score
import numpy as np

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.utils.testing import trim_dataloaders

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights
from cuda_selector import auto_cuda
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from tensordict import MemoryMappedTensor as MMT 
from tensordict import TensorDict as TD 
from tensordict import PersistentTensorDict as PTD

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'
    
    dataset = 'CIFAR100' 
    seed = 29
    bs = 64 
    name_model = 'vgg16'
    verbose = True 
    cv_dim = 300
    portion_list = ['train', 'val', 'test']
            
    #--------------------------------
    # Attacks configuration
    #--------------------------------
    
    attacks_config = {'c0': {'train': ['PGD','BIM','CW'], 'test': 'DeepFool'},
                      'c1': {'train': ['BIM','CW','DeepFool'], 'test': 'PGD'},
                      'c2': {'train': ['CW','DeepFool','PGD'], 'test': 'BIM'},
                      'c3': {'train': ['DeepFool','PGD','BIM'], 'test': 'CW'}}
    #--------------------------------
    # Layers selection 
    #--------------------------------
    
    target_layers = [
            'classifier.0',
            'classifier.3',
            # 'features.7',
            'features.14',
            'features.28'
            ]   

    #--------------------------------
    # Check the dataset 
    #--------------------------------
    cv_dataset_path = Path(f'/srv/newpenny/XAI/generated_data/cv_datasets')
    cv_dataset = {}
    test_only_dataset = {}

    for config in attacks_config:
        atk_train = attacks_config[config]['train']
        atk_test = attacks_config[config]['test']
        print(f'training: {atk_train} test: {atk_test}')

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    
        cvs_name = 'corevectors'
        cvs_path = f'/srv/newpenny/XAI/generated_data/corevectors/{dataset}/{name_model}'
        cvs_atks_home = f'/srv/newpenny/XAI/generated_data'
        
        cv = CoreVectors(
                path = cvs_path,
                name = cvs_name,
                device = device
                )
        
        cv_atks_train = {} 
        for atk_name in atk_train:
            cvs_atk_path = cvs_atks_home+f'/corevectors_attacks=my{atk_name}/{dataset}/{name_model}'
            cv_atks_train[atk_name] = CoreVectors(
                    path = cvs_atk_path,
                    name = cvs_name,
                    device = device
                    ) 
        cvs_atk_path = cvs_atks_home+f'/corevectors_attacks=my{atk_test}/{dataset}/{name_model}'
        
        cv_atks_test = CoreVectors(
                path = cvs_atk_path,
                name = cvs_name,
                device = device
                )
    
        with ExitStack() as stack:
            # get dataloader for corevectors from the original dataset 
            stack.enter_context(cv) # enter context manager
    #--------------------------------
    #     cv from ori 
    #--------------------------------
            cv.load_only(
                    loaders = ['train',
                               'val', 
                               'test'],
                    verbose = verbose 
                    )
            td = {'train':None, 
                  'val':None, 
                  'test':None}
            
            for key in td.keys():
                t = []
                for layer in target_layers:
                    
                    t.append(cv._corevds[key]['coreVectors'][layer][:,:cv_dim]) 
                     
                td[key] = (torch.stack(t, dim=1), torch.zeros(cv._corevds[key]['coreVectors'].shape)) 
                td[key] = {'data': torch.stack(t, dim=1), 
                           'labels': torch.zeros(cv._corevds[key]['coreVectors'].shape)}
                
            _test_ori = td['test']  
    
    #--------------------------------
    #     cv from atcks in train and val
    #--------------------------------
            n_samples_train = 40000
            n_samples_test = 10000
    
            # get dataloader for corevectors from atks dataset 
            test_ds_atk = {}
            for atk_name in atk_train:
                if verbose: print(f'\n---------\nLoading dataset for attack: {atk_name}')
                stack.enter_context(cv_atks_train[atk_name]) # enter context manager
                cv_atks_train[atk_name].load_only(
                        loaders = ['train', 'test'],
                        verbose = verbose 
                        )
                idx = torch.argwhere(((cv_atks_train[atk_name]._corevds['train']['attack_success']==1) & (cv._corevds['train']['result']==1)))
    
                idx_train = idx[:int(np.ceil(n_samples_train*0.3333))].squeeze().detach().cpu().numpy()
                
                idx_val = idx[int(np.ceil(n_samples_test*0.3333)):int(np.ceil(n_samples_test*0.6666))].squeeze().detach().cpu().numpy()
    
                idx_test_ = torch.argwhere(((cv_atks_train[atk_name]._corevds['test']['attack_success']==1) & (cv._corevds['test']['result']==1)))
                
                idx_test = idx_test_.squeeze().detach().cpu().numpy()
    
                idx_test_single = idx_test_[:int(np.ceil(n_samples_test*0.25))].squeeze().detach().cpu().numpy()
                print(f'Attack success rate {atk_name}:{idx_test_.shape}')
                

                t_train = []
                t_val = []
                t_test = []
                
                for layer in target_layers:
                    t_train.append(cv_atks_train[atk_name]._corevds['train']['coreVectors'][layer][idx_train,:cv_dim]) 
                    t_val.append(cv_atks_train[atk_name]._corevds['train']['coreVectors'][layer][idx_val,:cv_dim]) 
                    t_test.append(cv_atks_train[atk_name]._corevds['test']['coreVectors'][layer][:,:cv_dim])
                    
                _stacked_a_train = torch.stack(t_train, dim=1)
                la_train = torch.ones(cv_atks_train[atk_name]._corevds['train']['coreVectors'][idx_train].shape)
                
                td['train']['data'] = torch.cat((td['train']['data'], _stacked_a_train), dim=0)
                td['train']['labels'] = torch.cat((td['train']['labels'], la_train), dim=0)
    
                _stacked_a_val = torch.stack(t_val, dim=1)
                la_val = torch.ones(cv_atks_train[atk_name]._corevds['train']['coreVectors'][idx_val].shape)
                
                td['val']['data'] = torch.cat((td['val']['data'], _stacked_a_val), dim=0)
                td['val']['labels'] = torch.cat((td['val']['labels'], la_val), dim=0)
                
                _stacked_a_test = torch.stack(t_test, dim=1)
                la_test = torch.ones(cv_atks_train[atk_name]._corevds['test']['coreVectors'].shape)
    
                test_ds_atk[atk_name] = {'data': torch.cat((_test_ori['data'][idx_test], _stacked_a_test[idx_test]), dim=0),
                                         'labels': torch.cat((_test_ori['labels'][idx_test], la_test[idx_test]), dim=0)}
    
                td['test']['data'] = torch.cat((td['test']['data'], _stacked_a_test[idx_test_single]), dim=0)
                td['test']['labels'] = torch.cat((td['test']['labels'], la_test[idx_test_single]), dim=0)
            
            
    
            stack.enter_context(cv_atks_test) # enter context manager
            cv_atks_test.load_only(
                    loaders = ['test'],
                    verbose = verbose 
                    )    
    #--------------------------------
    #     cv from atcks in test
    #--------------------------------
            
            t_test = []
                
            for layer in target_layers: 
                t_test.append(cv_atks_test._corevds['test']['coreVectors'][layer][:,:cv_dim]) 
                
            idx_test_ = torch.argwhere(((cv_atks_test._corevds['test']['attack_success']==1) & (cv._corevds['test']['result']==1)))
            idx_test = idx_test_.squeeze().detach().cpu().numpy()
            print(f'Attack success rate {atk_test}:{idx_test_.shape}')
            
    
            # idx_test_single = idx[:int(np.ceil(len(idx)*0.25))].squeeze().detach().cpu().numpy()
            idx_test_single = idx_test_[:int(np.ceil(n_samples_test*0.25))].squeeze().detach().cpu().numpy()
                 
            _stacked_a_test = torch.stack(t_test, dim=1)
            la_test = torch.ones(cv_atks_test._corevds['test']['coreVectors'].shape)
    
            test_ds_atk[atk_test] = {'data': torch.cat((_test_ori['data'][idx_test], _stacked_a_test[idx_test]), dim=0), 
                                     'labels': torch.cat((_test_ori['labels'][idx_test], la_test[idx_test]), dim=0)}
       
            td['test']['data'] = torch.cat((td['test']['data'], _stacked_a_test[idx_test_single]), dim=0)
            td['test']['labels'] = torch.cat((td['test']['labels'],la_test[idx_test_single]), dim=0)
    
    #--------------------------------
    #  Finalization of Dataset
    #--------------------------------
        cv_dataset_path.mkdir(parents=True, exist_ok=True)
        for key,td in td.items():
            if key == 'test':
                cv_dataset_file = cv_dataset_path/'test_all'
                if not cv_dataset_file.exists():
                    cv_dataset[key] = PTD(filename=cv_dataset_file, batch_size=[td['data'].shape[0]], mode='w')
            
                    cv_dataset[key]['data'] = MMT.zeros(shape=(td['data'].shape))
                    cv_dataset[key]['labels'] = MMT.zeros(shape=(td['labels'].shape))
        
                    cv_dataset[key]['data'] = td['data']
                    cv_dataset[key]['labels'] = td['labels']
                    print(f'{key}: {cv_dataset[key]}, shape: {cv_dataset[key].size()}')
            else:
                cv_dataset_dir = cv_dataset_path/f'train={atk_train}_test={atk_test}'
                cv_dataset_dir.mkdir(parents=True, exist_ok=True)
                cv_dataset_file = cv_dataset_dir/f'{key}'
                
                cv_dataset[key] = PTD(filename=cv_dataset_file, batch_size=[td['data'].shape[0]], mode='w')
                
                cv_dataset[key]['data'] = MMT.zeros(shape=(td['data'].shape))
                cv_dataset[key]['labels'] = MMT.zeros(shape=(td['labels'].shape))
    
                cv_dataset[key]['data'] = td['data']
                cv_dataset[key]['labels'] = td['labels']
                print(f'{key}: {cv_dataset[key]}, shape: {cv_dataset[key].size()}')



        for atk_name, td in test_ds_atk.items():
            cv_dataset_file = cv_dataset_path/f'test_only={atk_test}'
            if  not cv_dataset_file.exists():
                test_only_dataset[atk_name] = PTD(filename=cv_dataset_file, batch_size=[td['data'].shape[0]], mode='w')
                
                test_only_dataset[atk_name]['data'] = MMT.zeros(shape=(td['data'].shape))
                test_only_dataset[atk_name]['labels'] = MMT.zeros(shape=(td['labels'].shape))
    
                test_only_dataset[atk_name]['data'] = td['data']
                test_only_dataset[atk_name]['labels'] = td['labels']
                print(f'{atk_name}: {test_only_dataset[atk_name]}, shape: {test_only_dataset[atk_name].size()}')
                
        

    
        






























































    

    
    
            
                
        
    



        
        

    

# torch stuff
import torch
from torch import nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

# Sklearn stuff
from sklearn.metrics import roc_auc_score

# TODO: test-only imports, clean it up later
from pathlib import Path
from tensordict import PersistentTensorDict as PTD
from tensordict import MemoryMappedTensor as MMT 
from tensordict import TensorDict as TD 
import sys
import numpy as np
import functools
from tqdm import tqdm

def parser_fn(x, cv_size):
    _d = x['data'][:,:,:cv_size]
    _l = x['labels']
    dd = _d.contiguous().view(_d.shape[0], _d.shape[1]*_d.shape[2])
    ll = _l.view(_l.shape[0], 1)
    return dd, ll

class Discriminator(nn.Module):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()
        self.verbose = kwargs['verbose'] if 'verbose' in kwargs else False 
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        
        #--------------------------
        # structure-related kwargs
        #--------------------------
        self.n_layers = kwargs['n_layers'] 
        self.layer_size = kwargs['layer_size'] 
        self.l_sizes = [self.layer_size for i in range(self.n_layers)] 
        self.act_fn = kwargs['activation_fn'] if 'activation_fn' in kwargs else nn.ReLU
        
        #--------------------------
        # training related hyperp
        #--------------------------
        self.scheduler_kwargs = kwargs['scheduler_kwargs'] if 'scheduler_kwargs' in kwargs else None 
        if self.scheduler_kwargs != None:
            self.early_stopping_patience = self.scheduler_kwargs.pop('early_stopping_patience')

        self.optim_type = kwargs['optim_type'] if 'optim_type' in kwargs else optim.Adam
        self.optim_kwargs = kwargs['optim_kwargs'] if 'optim_kwargs' in kwargs else {}
        
        self.loss_type = kwargs['loss_fn'] if 'loss_fn' in kwargs else nn.BCELoss
        self.loss_kwargs = kwargs['loss_kwargs'] if 'loss_kwargs' in kwargs else {}
        self.loss_fn = self.loss_type(**self.loss_kwargs)

        self.epoch = 0
        self.stop_training = False
        self.best_val_loss = float('inf')

        #--------------------------
        # data related parameters 
        #--------------------------
        self.dl = kwargs['dataloaders']
        self.save_path = kwargs['save_path'] if 'save_path' in kwargs else None
        
        #------------------------
        # constructing network
        #------------------------
        # inferring input size from data
        _d, _l = next(iter(self.dl['train']))
        _in_size = _d[0].shape[0] 
        
        _layers = []
        _layers.append(nn.Linear(_in_size, self.l_sizes[0]))
        for i in range(1, len(self.l_sizes)):
            _layers.append(nn.Linear(self.l_sizes[i-1], self.l_sizes[i]))
            _layers.append(self.act_fn())
        _layers.append(nn.Linear(self.l_sizes[-1], 1))
        _layers.append(nn.Sigmoid())

        self.nn = nn.Sequential(*_layers)
        self.optim = self.optim_type(self.parameters(), **self.optim_kwargs)
        
        if self.scheduler_kwargs != None:
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', **self.scheduler_kwargs)
        else:
            self.scheduler = None

        # count the number of parameters in the NN
        _n_params = 0
        for p in self.parameters():
            _n_params += p.numel()
        self.num_parameters = _n_params 
        self = self.to(self.device)
        return 
        
    def forward(self, x): 
        return self.nn(x)

    def train_iteration(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # training part
        self.optim.zero_grad()
        outputs = self.forward(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optim.step()
        
        return loss

    def evaluate(self, dl):
        test_loss = 0.0

        with torch.no_grad():
            inputs, targets = dl.collate_fn(dl.dataset)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            outputs = self.forward(inputs)
            loss = self.loss_fn(outputs, targets)
            test_loss += loss.item()
        return test_loss/len(dl)  

    def AUC_test(self, dl):
        n_samples = dl.dataset['labels'].shape[0]
        scores = torch.zeros(n_samples, 1)

        with torch.no_grad():
            _n = 0
            inputs, targets = dl.collate_fn(dl.dataset) 
            _ni = targets.shape[0]
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            scores[_n:_n+_ni] = self.forward(inputs)
            _n += _ni
        scores = scores.detach().cpu().numpy()
        labels = dl.dataset['labels'].detach().cpu().numpy()
                
        return 1-roc_auc_score(labels, scores)
        
    def train_epoch(self):
        train_loss = 0.0
        
        for data, labels in tqdm(self.dl['train'], disable=not self.verbose):
            loss = self.train_iteration(data, labels)
            train_loss += loss.item()
            
        train_loss /= len(self.dl['train'])
        val_loss = self.evaluate(self.dl['val']) 
        if self.verbose: print(f'Epoch [{self.epoch}], Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')

        if self.scheduler != None: 
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                if self.save_path != None: torch.save(self.state_dict(), self.save_path/f'model.pt')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    if self.verbose: print("Early stopping: Validation loss hasn't improved for", self.early_stopping_patience, "epochs.")
                    self.stop_training = True
                        
                # step the scheduler
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
    
                current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.optim.param_groups[0]['lr']
                if self.verbose: print(f'Current lr: {current_lr:.6f}')
        
        self.epoch += 1
        return train_loss, val_loss 

if __name__ == "__main__":
    verbose = True
    portion = ['train', 'val']
    target_layers = ['l1', 'l2', 'l3']
    device = torch.device('cuda:0')

    cv_size = 2
    ds1 = 10 
    ds2 = 15 
    n = 5
    bs = 3
    max_size = min(ds1,ds2)

    save_path = Path.cwd()/'banana_learning'
    save_path.mkdir(exist_ok=True, parents=True)

    # constructing dataloader
    collate_fn = functools.partial(parser_fn, cv_size=cv_size)

    dataloaders = {}
    for p in portion: 
        path1 = Path(f'./banana{p}')
        if path1.exists():
            t1 = PTD.from_h5(path1, mode='r') 
        else:
            t1 = PTD(filename=path1, batch_size=[n], mode='w')
            t1['l1'] = MMT.zeros(shape=(n, ds1))
            t1['l2'] = MMT.zeros(shape=(n, ds1))
            t1['l3'] = MMT.zeros(shape=(n, ds1))
            t1['l1'] = torch.randint(0, 50, size=(n, ds1))
            t1['l2'] = torch.randint(0, 50, size=(n, ds1))
            t1['l3'] = torch.randint(0, 50, size=(n, ds1))
    
        path2 = Path('./banana2{p}')
        if path2.exists():
            t2 = PTD.from_h5(path2, mode='r') 
        else:
            t2 = PTD(filename=path2, batch_size=[n], mode='w')
            t2['l1'] = MMT.zeros(shape=(n, ds2))
            t2['l2'] = MMT.zeros(shape=(n, ds2))
            t2['l3'] = MMT.zeros(shape=(n, ds2))
            t2['l4'] = MMT.zeros(shape=(n, ds2))
            t2['l1'] = torch.randint(0, 50, size=(n, ds2))
            t2['l2'] = torch.randint(0, 50, size=(n, ds2))
            t2['l3'] = torch.randint(0, 50, size=(n, ds2))
            t2['l4'] = torch.randint(0, 50, size=(n, ds2))
    
        _t1, _t2 = [], []
        for _l in target_layers:
            _t1.append(t1[_l][:,:max_size]) 
            _t2.append(t2[_l][:,:max_size]) 
        _stacked_t1 = torch.stack(_t1, dim=1)
        _stacked_t2 = torch.stack(_t2, dim=1)
        
        data = torch.cat([_stacked_t1, _stacked_t2], dim=0) 
        labels = torch.randint(0,2,size=(2*n,), dtype=torch.float)
    
        dataloaders[p] = DataLoader(
                TD({'data':data, 'labels': labels}, batch_size=data.shape[0]),
                batch_size = bs,
                shuffle = True,
                collate_fn = collate_fn,
                )

    discriminator = Discriminator(
        device = device,
        n_layers = 4,
        layer_size = 5,
        dataloaders = dataloaders,
        optim_type = optim.Adam,
        optim_kwargs = {'lr': 1e-4},
        #scheduler_kwargs = {'early_stopping_patience':10, 'patience': 5, 'factor': 0.1 },
        loss_type = nn.BCELoss,
        loss_kwargs = {'reduction': 'sum'},
        save_path = save_path,
        verbose = verbose,
        )

    max_epochs = 1000
    for num_epoch in range(max_epochs):
        train_loss, val_loss = discriminator.train_epoch()
        print('train, val looses: ', train_loss, val_loss)
        if discriminator.stop_training:
            break

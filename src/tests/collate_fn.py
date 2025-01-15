from tensordict import PersistentTensorDict as PTD
from tensordict import MemoryMappedTensor as MMT 
from tensordict import LazyStackedTensorDict as LSTD
from tensordict import lazy_stack
import torch
from torch.utils.data import DataLoader

def c1(x):
    return x['a'], x['b']

def c2(x):
    return x['b'], x['a']

if __name__ == '__main__':
    n = 5 
    ds = 3
    bs = 5 

    t1 = PTD(filename='banana1', batch_size=[n], mode='w')
    t1['a'] = MMT.zeros(shape=(n, ds))
    t1['b'] = MMT.zeros(shape=(n, ds))
    t1['a'] = torch.rand(n, ds)
    t1['b'] = torch.rand(n, ds)
    
    dl = DataLoader(t1, batch_size=bs, collate_fn=c1) 
    print('---- t1\n')
    for d in dl:
        a, b = d
        print('a: ', a)
        print('b: ', b)

    dl.collate_fn = c2
    print('---- t2\n')
    for d in dl:
        a, b = d
        print('a: ', a)
        print('b: ', b)


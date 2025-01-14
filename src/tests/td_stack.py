from tensordict import PersistentTensorDict as PTD
from tensordict import MemoryMappedTensor as MMT 
from tensordict import LazyStackedTensorDict as LSTD
from tensordict import lazy_stack
import torch
from torch.utils.data import DataLoader

if __name__ == '__main__':
    n = 5 
    ds = 3
    bs = 5 

    t1 = PTD(filename='banana1', batch_size=[n], mode='w')
    t1['a'] = MMT.zeros(shape=(n, ds))
    t1['b'] = MMT.zeros(shape=(n, ds))
    t1['a'] = torch.rand(n, ds)
    t1['b'] = torch.rand(n, ds)
    
    t2 = PTD(filename='banana2', batch_size=[n], mode='w')
    t2['a'] = MMT.zeros(shape=(n, ds))
    t2['b'] = MMT.zeros(shape=(n, ds))
    t2['a'] = torch.rand(n, ds)
    t2['b'] = torch.rand(n, ds)

    t3 = torch.cat([t1, t2])

    dl1 = DataLoader(t1, batch_size=bs, collate_fn=lambda x:x) 
    dl2 = DataLoader(t2, batch_size=bs, collate_fn=lambda x:x) 
    dl3 = DataLoader(t3, batch_size=2*bs, collate_fn=lambda x:x, shuffle=True) 

    print('---- t1\n')
    for d in dl1:
        print('a: ', d['a'])
        print('b: ', d['b'])

    print('---- t2\n')
    for d in dl2:
       print('a: ', d['a'])
       print('b: ', d['b'])

    print('---- t3\n')
    for d in dl3:
        print('a: ', d['a'])
        print('b: ', d['b'])

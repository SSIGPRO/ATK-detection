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
    tl = ['1', '2']

    t1 = PTD(filename='banana1', batch_size=[n], mode='w')
    t1['a'] = PTD(filename='second_level', batch_size=[n], mode='w')
    t1['a']['1'] = MMT.zeros(shape=(n, ds))
    t1['a']['1'] = torch.rand(n, ds)
    t1['a']['2'] = MMT.zeros(shape=(n, ds))
    t1['a']['2'] = torch.rand(n, ds)
    t1['a']['3'] = MMT.zeros(shape=(n, ds))
    t1['a']['3'] = torch.rand(n, ds)
    
    
    t2 = PTD(filename='banana2', batch_size=[n], mode='w')
    t2['a'] = PTD(filename='second_level', batch_size=[n], mode='w')
    t2['a']['1'] = MMT.zeros(shape=(n, ds))
    t2['a']['1'] = torch.rand(n, ds)
    t2['a']['2'] = MMT.zeros(shape=(n, ds))
    t2['a']['2'] = torch.rand(n, ds)
    print(t1, t2)
    t3 = {}

    for l in tl:
        t3 = torch.cat
    
    t3 = torch.cat([t1['a'], t2['a']])
    quit()

    dl1 = DataLoader(t1, batch_size=bs, collate_fn=lambda x:x) 
    dl2 = DataLoader(t2, batch_size=bs, collate_fn=lambda x:x) 
    # dl3 = DataLoader(t3, batch_size=2*bs, collate_fn=lambda x:x) 

    print('---- t1\n')
    for d in dl1:
        print('a: ', d['a'])
        d.pop('a')
        print('b: ', d['b'])
    print(t1)

    print('---- t2\n')
    for d in dl2:
       print('a: ', d['a'])
       print('b: ', d['b'])
    t3 = torch.cat([t1, t2])
    dl3 = DataLoader(t3, batch_size=2*bs, collate_fn=lambda x:x)

    print('---- t3\n')
    for d in dl3:
        print('a: ', d['a'])
        print('b: ', d['b'])

    # t3.pop('a')

    # prova = PTD.from_h5('banana1', mode='r')
    # print(prova['a'])

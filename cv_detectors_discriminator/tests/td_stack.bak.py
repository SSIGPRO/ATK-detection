from tensordict import PersistentTensorDict as PTD
from tensordict import MemoryMappedTensor as MMT 
from tensordict import LazyStackedTensorDict as LSTD
from tensordict import TensorDict as TD 
from tensordict import lazy_stack
import tensordict
import torch
from torch.utils.data import DataLoader 
from pathlib import Path
from tensordict.utils import set_lazy_legacy

def p(td):
    print('\n-----------------')
    for k in td.keys():
        print(k, '\n', td[k])

def foo(x):
    shape = x.shape
    y = x.view(shape[1],shape[0]*shape[2])
    return y

if __name__ == '__main__':
    pause = False 
    n = 4 
    cv_size = 2
    bs = 3
    ds1 = 3
    ds2 = 4
    tl = ['a', 'b']
    path1 = Path('./banana1')
    path2 = Path('./banana2')

    t1 = PTD(filename=path1, batch_size=[n], mode='w')
    t1['a'] = MMT.zeros(shape=(n, ds1))
    t1['b'] = MMT.zeros(shape=(n, ds1))
    t1['c'] = MMT.zeros(shape=(n, ds1))
    t1['a'] = torch.randint(0, 50, size=(n, ds1))
    t1['b'] = torch.randint(0, 50, size=(n, ds1))
    t1['c'] = torch.randint(0, 50, size=(n, ds1))
    
    t2 = PTD(filename=path2, batch_size=[n], mode='w')
    t2['a'] = MMT.zeros(shape=(n, ds2))
    t2['b'] = MMT.zeros(shape=(n, ds2))
    t2['a'] = torch.randint(0, 50, size=(n, ds2))
    t2['b'] = torch.randint(0, 50, size=(n, ds2))

    if pause: input('labels')
    l = MMT(torch.randint(0, 2, size=(2,n)))
    print('l: ', l)

    p(t1)
    p(t2)
    if pause: input('t3')
    t3 = LSTD(*[t1, t2], stack_dim=0, hook_out=foo)
    print('aaa: ', t3.get_nestedtensor('a')) 
    if pause: input('_td')
    _td = t3#TD({'data': t3, 'labels': l}, batch_size=2*n)
    dl3 = DataLoader(_td, batch_size=bs, collate_fn=lambda x:x, shuffle=False) 

    if pause: input('printing')
    print('---- t3\n')
    for d in dl3:
        _d = d
        #_d = d['data']
        #_l = d['labels'] 
        print('a: ', _d['a'])
        print('b: ', _d['b'])
        #print('l: ', _l)

import torch
from tensordict import PersistentTensorDict as PTD
from tensordict import MemoryMappedTensor as MMT 
from torch.utils.data import DataLoader, WeightedRandomSampler


def p(t):
    for k, v in t.items():
        print(f'{k}: ', v)
    return

if __name__ == "__main__":
    n = 10000 
    ds = 3
    bs = 5000

    t = PTD(filename='banana1', batch_size=[n], mode='w')
    t['a'] = MMT.zeros(shape=(n, ds))
    t['b'] = MMT.zeros(shape=(n, ds))
    t['l'] = MMT.zeros(shape=(n,), dtype=torch.int)
    t['a'] = torch.rand(n, ds)
    t['b'] = torch.rand(n, ds)
    t['l'] = torch.randint(0, 4, size=(n,))
    p(t)

    _dists = torch.bincount(t['l'])
    dists = _dists/_dists.sum()
    print('dists: ', dists)
    weights = torch.Tensor([dists[x] for x in t['l']])
    sampler = WeightedRandomSampler(weights, len(weights))
    dl = DataLoader(t, batch_size = bs, sampler=sampler, collate_fn=lambda x:x)
    ds_data = next(iter(dl))
    _dists = torch.bincount(ds_data['l'])
    dists = _dists/_dists.sum()
    print('dists: ', dists)
    print(ds_data)
    p(ds_data)

    ndl = DataLoader(ds_data, batch_size=round(bs/3), collate_fn=lambda x:x)

    for i, data in enumerate(ndl):
        print('\n-------- ', i)
        p(data)

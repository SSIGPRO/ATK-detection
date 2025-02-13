# python stuff
from matplotlib import pyplot as plt
from pathlib import Path

# torch stuff
import torch
from tensordict import PersistentTensorDict as PTD
from tensordict import MemoryMappedTensor as MMT 
from torch.utils.data import DataLoader, WeightedRandomSampler

# GP stuff
import gpytorch

# out stuff
from estimators.gp import GPModel

def p(t):
    for k, v in t.items():
        print(f'{k}: ', v)
    return

def foo(d, os):
    res = torch.zeros(d.shape[0], os)
    nf = d.shape[1]
    for i in range(os):
        for j in range(nf):
            res[:,i] += torch.cos(d[:,j]*torch.pi)
    return res

if __name__ == "__main__":
    device = 'cuda'
    alpha = 9  
    nv = 25 # number validation samples 
    ds = 20 # input data dimension
    n = round(alpha*ds) # number of training data
    os = 5
    lr = 0.2
    max_iter = 500 
    kernel_kwargs = {'nu': 1.5, 'power':2}
    model_file = Path.cwd()/'gp_model_test.pt' 

    t = PTD(filename='./banana1', batch_size=[n], mode='w')
    t['x'] = MMT.zeros(shape=(n, ds))
    t['l'] = MMT.zeros(shape=(n, os))
    t['x'] = torch.rand(n, ds)
    t['l'] = foo(t['x'], os) 
    
    v = PTD(filename='./banana2', batch_size=[nv], mode='w')
    v['x'] = MMT.zeros(shape=(nv, ds))
    v['l'] = MMT.zeros(shape=(nv, os))
    v['x'] = torch.rand(nv, ds)
    v['l'] = foo(v['x'], os) 

    model = GPModel(x=t['x'], y=t['l'], lr=lr, kernel_kwargs=kernel_kwargs, device=device)   
    
    print('-- training')
    for i in range(max_iter):
        loss = model.train_iteration() 
        if i%10==0: print('Iter %d/%d - Loss: %.3f'%(i + 1, max_iter, loss.item()))
        
    # Set into eval mode
    model.eval()
    
    deltas = 0.1*torch.linspace(0.1, 1, 10)
    c = torch.zeros(len(deltas), os)

    for i, delta in enumerate(deltas):
        test_data = torch.normal(t['x'], delta)

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = model.get_likelihood(test_data.to(device))
            mean = predictions.mean
            lower, upper = predictions.confidence_region()
            conf = (upper-lower)
        conf = conf.cpu()
        
        c[i,:] = conf.mean(dim=0, keepdim=True)

    fig = plt.figure()
    labels = ['feature %d'%_os for _os in range(os)]
    print('confs: ', c)
    plt.plot(deltas, c, '.', label=labels)
    plt.xlabel(r'$\sigma$')
    plt.legend()

    fig.tight_layout()
    plt.savefig(Path.cwd()/'gp_scale.png', dpi=300)
    plt.show()


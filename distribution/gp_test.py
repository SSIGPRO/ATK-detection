# python stuff
from matplotlib import pyplot as plt
from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

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

def foo(d, nc):
    res = torch.zeros(d.shape[0])
    for i in range(d.shape[1]):
        res += ((-1)**i)*torch.cos(d[:,i]*2*torch.pi)
    res = res.round().int()
    res -= res.min() # labels must be from positive int
    return res.clamp(0, nc-1) 

if __name__ == "__main__":
    plot = False
    recalc = True
    device = 'cuda'
    n =  300 # num train samples
    nc = 4 # num classes 
    nv = 50 # num val samples 
    ds = 300 # n features 

    # learning hyperp
    lr = 0.05
    max_iter = 500
    kernel_kwargs = {'nu': 1.5, 'power':2}
    lh_kwargs = {'alpha_epsilon': 0.01}

    model_file = Path.cwd()/'gp_model_test.pt' 

    t = PTD(filename='./banana1', batch_size=[n], mode='w')
    t['x'] = MMT.zeros(shape=(n, ds))
    t['l'] = MMT.zeros(shape=(n,), dtype=torch.int)
    t['x'] = torch.rand(n, ds)
    t['l'] = foo(t['x'], nc) 
    
    v = PTD(filename='./banana2', batch_size=[nv], mode='w')
    v['x'] = MMT.zeros(shape=(nv, ds))
    v['l'] = MMT.zeros(shape=(nv,), dtype=torch.int)
    v['x'] = torch.rand(nv, ds)
    v['l'] = foo(v['x'], nc) 
    
    model = GPModel(
            x = t['x'],
            y = t['l'],
            n_classes = nc,
            lr = lr,
            kernel_kwargs = kernel_kwargs,
            likelihood_kwargs = lh_kwargs,
            device=device
            )   
    
    if not model_file.exists() or recalc:
        print('-- training')
        for i in range(max_iter):
            loss = model.train_iteration() 
            if i%10==0: print('Iter %d/%d - Loss: %.3f'%(i + 1, max_iter, loss.item()))
            
        # save model
        print('saving: ', len(model.state_dict()))
        torch.save(model.state_dict(), model_file)
    else:
        sd = torch.load(model_file)
        print('loading: ', len(sd))
        model.load_state_dict(sd)

    # Set into eval mode
    model.eval()

    if plot:
        pallet = list(mcolors.XKCD_COLORS)
        cs = []
        for i in range(nc):
            cs.append(pallet[i*10+5])
        labels = ['train', 'test', 'data']
        markers = ['o', 's', 'v']
        fig, axs = plt.subplots(1, nc, subplot_kw={'projection': '3d'}, figsize=(nc*4, 4))
        
        for i in range(nc):
            axs[i].scatter(t['x'][:,0], t['x'][:,1], marker=markers[0], c=[cs[x] for x in t['l']], alpha=0.5)
            axs[i].scatter(v['x'][:,0], v['x'][:,1], marker=markers[1], c=[cs[x] for x in v['l']], alpha=0.5)

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        _x = v['x'].to(device)
        _y = v['l'].to(device)
        _lh = model.get_likelihood(_x)
        #mean = _lh.mean
        lower, upper = _lh.confidence_region()
        conf = upper - lower

        # normalization
        norm_samples = _lh.sample(torch.Size((256,))).exp()
        prob = (norm_samples/norm_samples.sum(-2, keepdim=True)).mean(0)
        
        pred = prob.max(dim=0)[1] # [1] to get indixes (classes) 
        pred_conf = conf.gather(dim=0, index=pred.view(1,-1))
        loss = -model.mll(model(_x), _y).mean()
    print('loss: ', loss)
    print('pred conf: ', pred_conf)

    if not plot:
        quit()

    for i in range(nc):
        axs[i].scatter(v['x'][:,0], v['x'][:,1], marker=markers[2], c=[cs[x] for x in pred], alpha=0.5)
    
    # plot confidence
    n_grid = 25 
    low, high = t['x'].min()-0.1, t['x'].max()+0.1
    grid = torch.linspace(low, high, n_grid)
    x_mat, y_mat = torch.meshgrid(grid, grid)
    t = torch.cat([x_mat.reshape(-1,1), y_mat.reshape(-1,1)], dim = 1)

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        _lh = model.get_likelihood(t.to(device))
        mean = _lh.mean
        lower, upper = _lh.confidence_region()
        conf = (upper-lower)
        
        # normalization
        norm_samples = _lh.sample(torch.Size((256,))).exp()
        prob = (norm_samples/norm_samples.sum(-2, keepdim=True)).mean(0)
        
        pred = prob.max(dim=0)[1] # [1] to get indixes (classes) 
   

    conf = conf.cpu()
    for i in range(nc):
        x = t[:,0].reshape(n_grid, n_grid)
        y = t[:,1].reshape(n_grid, n_grid)
        z = conf[i].reshape(n_grid, n_grid)
        axs[i].plot_surface(
                x, y, z,
                cmap = plt.cm.coolwarm,
                alpha = 0.5
                )
        axs[i].set_title(f'class {i}')
    
    handles = [plt.plot([],[], color='g', marker=m, label=l)[0] for l, m in zip(labels, markers)]
    leg = axs[0].legend(handles=handles, loc=(0.8, 1.01), title='data')

    handles = [mpatches.Patch(color=c, label=f'{i}') for i, c in enumerate(cs)]
    axs[0].add_artist(leg)
    axs[0].legend(handles=handles, loc=(0.0, 1.01), title='classes')

    fig.tight_layout()
    plt.savefig(Path.cwd()/'gp_test.png', dpi=300)
    plt.show()


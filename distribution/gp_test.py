# python stuff
from matplotlib import pyplot as plt
from pathlib import Path
import matplotlib.colors as mcolors

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
        res += torch.cos(d[:,i]*torch.pi)
    res = res.round().int()
    res -= res.min() # labels must be from positive int
    return res.clamp(0, nc-1) 

if __name__ == "__main__":
    device = 'cuda'
    n =  30 
    nc = 4 
    nv = 25 
    ds = 2
    bs = 5
    lr = 0.9
    max_iter = 200
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
    
    if not model_file.exists():
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

    pallet = list(mcolors.XKCD_COLORS)
    cs = []
    for i in range(nc):
        cs.append(pallet[i])

    fig, axs = plt.subplots(1, nc+1, subplot_kw={'projection': '3d'}, figsize=((nc+1)*4, 4))

    axs[0].scatter(t['x'][:,0], t['x'][:,1], marker='.', c=[cs[x] for x in t['l']], label='train')
    axs[0].scatter(v['x'][:,0], v['x'][:,1], marker='s', c=[cs[x] for x in v['l']], label='test')

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = model.get_likelihood(v['x'].to(device))
        mean = predictions.mean
        print('mean: ', mean)
        print('mean loc: ', predictions.loc)
        lower, upper = predictions.confidence_region()

    axs[0].scatter(v['x'][:,0], v['x'][:,1], marker='v', c=[cs[x] for x in mean], label='pred')
    axs[0].legend()

    # plot confidence
    n_grid = 25 
    low, high = t['x'].min()-0.1, t['x'].max()+0.1
    grid = torch.linspace(low, high, n_grid)
    x_mat, y_mat = torch.meshgrid(grid, grid)
    t = torch.cat([x_mat.reshape(-1,1), y_mat.reshape(-1,1)], dim = 1)

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = model.get_likelihood(t.to(device))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()
        conf = (upper-lower)

    conf = conf.cpu()
    for i in range(nc):
        x = t[:,0].reshape(n_grid, n_grid)
        y = t[:,1].reshape(n_grid, n_grid)
        z = conf[i].reshape(n_grid, n_grid)
        axs[i+1].plot_surface(
                x, y, z,
                cmap = plt.cm.coolwarm,
                alpha = 0.5
                )
        axs[i+1].set_title(f'class {i}')
    fig.tight_layout()
    plt.savefig(Path.cwd()/'gp_test.png', dpi=300)
    plt.show()


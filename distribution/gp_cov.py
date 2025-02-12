# python stuff
from matplotlib import pyplot as plt

# torch stuff
import torch
from tensordict import PersistentTensorDict as PTD
from tensordict import MemoryMappedTensor as MMT 
from torch.utils.data import DataLoader, WeightedRandomSampler

# GP stuff
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.kernels import MultitaskKernel, MaternKernel, PeriodicKernel, RBFKernel, PolynomialKernel
from gpytorch.distributions import MultitaskMultivariateNormal

class GPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, **kernel_kwargs):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        n_tasks = train_y.shape[1]
        ard_num_dim = train_x.shape[1]

        self.mean_module = MultitaskMean(ConstantMean(), num_tasks = n_tasks)
        self.covar_module = MultitaskKernel(
            MaternKernel(nu = kernel_kwargs['nu'], ard_num_dims = ard_num_dim)+
            PeriodicKernel(ard_num_dims = ard_num_dim),#+
            #RBFKernel(ard_num_dims = ard_num_dim)+
            #PolynomialKernel(power = kernel_kwargs['power'], ard_num_dims = ard_num_dim),
            num_tasks = n_tasks,
            rank = 1,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)

def p(t):
    for k, v in t.items():
        print(f'{k}: ', v)
    return

def foo(d, os):
    res = torch.zeros(d.shape[0], os)
    for i in range(os):
        res[:,i] = torch.cos(d[:,0]*torch.pi)**(i+1) + torch.sin(d[:,1]*torch.pi)**(i+1)
    return res

if __name__ == "__main__":
    device = 'cuda'
    n =  15 
    nv = 25 
    ds = 2
    os = 3
    bs = 5
    lr = 0.3
    max_iter = 200
    kernel_kwargs = {'nu': 1.5}

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

    dl = DataLoader(t, batch_size = bs, collate_fn=lambda x:x)
    for d in dl:
        p(d)

    likelihood = MultitaskGaussianLikelihood(os)
    model = GPModel(t['x'], t['l'], likelihood, **kernel_kwargs)   
    likelihood = likelihood.to(device)
    model = model.to(device)

    print('-- training')
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
    mll = ExactMarginalLogLikelihood(likelihood, model) 

    for i in range(max_iter):
        optimizer.zero_grad()
        output = model(t['x'].to(device))
        loss = -mll(output, t['l'].to(device))
        loss.backward()
        if i%10==0: print('Iter %d/%d - Loss: %.3f'%(i + 1, max_iter, loss.item()))
        optimizer.step()

    # Set into eval mode
    model.eval()
    likelihood.eval()
    
    lm = t['l'].min()
    ld = t['l'].max()-t['l'].min()
    def scale(x):
        return (x-lm)/ld
    
    fig, axs = plt.subplots(1, os, subplot_kw={'projection': '3d'}, figsize=(os*4, 4))
    for i in range(os):
        axs[i].plot(t['x'][:,0], t['x'][:,1], scale(t['l'][:,i]), '.b', label='train')
        axs[i].plot(v['x'][:,0], v['x'][:,1], scale(v['l'][:,i]), '.r', label='test')

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(v['x'].to(device)))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

    for i in range(os):
        axs[i].plot(v['x'][:,0], v['x'][:,1], scale(mean[:,i].cpu()), '.g', label='pred')

    # plot confidence
    n_grid = 25 
    low, high = t['x'].min()-0.1, t['x'].max()+0.1
    grid = torch.linspace(low, high, n_grid)
    x_mat, y_mat = torch.meshgrid(grid, grid)
    t = torch.cat([x_mat.reshape(-1,1), y_mat.reshape(-1,1)], dim = 1)

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(t.to(device)))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()
        conf = (upper-lower)

    conf = conf.cpu()
    for i in range(os):
        x = t[:,0]
        y = t[:,1]
        z = conf[:,i]
        axs[i].plot_surface(
                x.reshape(n_grid, n_grid),
                y.reshape(n_grid, n_grid),
                z.reshape(n_grid, n_grid),
                cmap = plt.cm.coolwarm,
                alpha = 0.5
                )
    plt.legend()
    fig.tight_layout()
    plt.show()

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
from gpytorch.kernels import ScaleKernel, RBFKernel, MultitaskKernel
from gpytorch.distributions import MultitaskMultivariateNormal

class GPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        n_tasks = train_y.shape[1]
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks = n_tasks)
        self.covar_module = MultitaskKernel(
            RBFKernel(),
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
        res[:,i] = torch.sin(d[:,0]*0.1*torch.pi*(i+1)) + torch.cos(d[:,1]*0.1*torch.pi*(i+1))
    return res

if __name__ == "__main__":
    device = 'cuda'
    n = 100 
    nv = 10
    ds = 2
    os = 3
    bs = 5
    max_iter = 50

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
    model = GPModel(t['x'], t['l'], likelihood)   
    likelihood = likelihood.to(device)
    model = model.to(device)

    print('-- training')
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5)  # Includes GaussianLikelihood parameters
    mll = ExactMarginalLogLikelihood(likelihood, model) 

    for i in range(max_iter):
        optimizer.zero_grad()
        output = model(t['x'].to(device))
        loss = -mll(output, t['l'].to(device))
        loss.backward()
        print('Iter %d/%d - Loss: %.3f'%(i + 1, max_iter, loss.item()))
        optimizer.step()

    # Set into eval mode
    model.eval()
    likelihood.eval()

    plt.figure()
    plt.plot(t['l'][:,0], t['l'][:,1], '.b', label='train')
    plt.plot(v['l'][:,0], v['l'][:,1], '.r', label='test')

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(v['x'].to(device)))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()
        print('mean: ', mean)
        print('lower: ', lower)
        print('upper: ', upper)
    plt.plot(mean[:,0].cpu(), mean[:,1].cpu(), '.g', label='pred')

    # plot confidence
    n_grid = 5 
    low, high = t['x'].min()-1, t['x'].max()+1
    grid = torch.linspace(low, high, n_grid)
    x_mat, y_mat = torch.meshgrid(grid, grid)
    t = torch.cat([x_mat.reshape(-1,1), y_mat.reshape(-1,1)], dim = 1)
    print('plot cov: ', t, t.shape)

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(t.to(device)))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()
        print('mean: ', mean, mean.shape)
        print('upper: ', upper.shape, lower.shape)
        conf = (upper-lower).norm(dim=1, keepdim=True).reshape(n_grid, n_grid)
        print('conf: ', conf, conf.shape)
    im = plt.contourf(mean[:,0].reshape(n_grid, n_grid).cpu(), mean[:,1].reshape(n_grid, n_grid).cpu(), conf.cpu())
    plt.colorbar(im, alpha=0.5)
    plt.legend()
    plt.show()

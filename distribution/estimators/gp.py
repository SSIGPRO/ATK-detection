# torch stuff
import torch

# torch stuff# GP stuff
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.kernels import MultitaskKernel, MaternKernel, PeriodicKernel, RBFKernel, PolynomialKernel
from gpytorch.distributions import MultitaskMultivariateNormal

def parser_fn(x, cv_size):
    _d = x['data'][:,:,:cv_size]
    _l = x['labels']
    dd = _d.contiguous().view(_d.shape[0], _d.shape[1]*_d.shape[2])
    ll = _l.view(_l.shape[0], 1)
    return dd, ll


class GPModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(GPModel, self).__init__()
        self.x = kwargs['x']
        self.y = kwargs['y']
        self.kernel_kwargs = kwargs['kernel_kwargs']
        self.lr = kwargs['lr']
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'

        n_tasks = self.y.shape[1]
        ard_num_dim = self.x.shape[1]

        # define likelihood (lh)
        self.lh = MultitaskGaussianLikelihood(n_tasks)
        self.gp = GPWrap(
                x = self.x,
                y = self.y,
                lk = self.lh,
                n_tasks = n_tasks,
                ard_num_dim = ard_num_dim,
                kernel_kwargs = self.kernel_kwargs
                )

        # send to device
        self.lh = self.lh.to(self.device)
        self.gp = self.gp.to(self.device)
        self.x = self.x.to(self.device)
        self.y = self.y.to(self.device)

        # Find optimal model hyperparameters
        self.gp.train()
        self.lh.train()

        # ---- Training related definitions
        self.optimizer = torch.optim.Adam(self.gp.parameters(), lr=self.lr)
        # Includes GaussianLikelihood parameters
        self.mll = ExactMarginalLogLikelihood(self.lh, self.gp) 

        return
        
    def forward(self, x):
        return self.gp(x)

    def train_iteration(self):
        self.optimizer.zero_grad()
        output = self.gp(self.x)
        loss = -self.mll(output, self.y)
        loss.backward()
        self.optimizer.step()
        return loss

    def get_likelihood(self, x):
        return self.lh(self.gp(x))

class GPWrap(ExactGP):
    def __init__(self, **kwargs):
        x = kwargs['x']
        y = kwargs['y']
        lk = kwargs['lk']
        ard_num_dim = kwargs['ard_num_dim']
        n_tasks = kwargs['n_tasks']
        kernel_kwargs = kwargs['kernel_kwargs']
        
        super(GPWrap, self).__init__(x, y, lk)

        # define means module
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks = n_tasks)
        
        # define covariance module
        kernel = \
                MaternKernel(nu = kernel_kwargs['nu'], ard_num_dims = ard_num_dim) + \
                PeriodicKernel(ard_num_dims = ard_num_dim) + \
                RBFKernel(ard_num_dims = ard_num_dim) + \
                PolynomialKernel(power = kernel_kwargs['power'], ard_num_dims = ard_num_dim)
            
        self.covar_module = MultitaskKernel(
            kernel,
            num_tasks = n_tasks,
            rank = 1,
        )
    
        return

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)

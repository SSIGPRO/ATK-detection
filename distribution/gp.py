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
        res[:,i] = torch.sin(d*0.1*i).mean(dim=1)
    return res

if __name__ == "__main__":
    n = 10 
    ds = 3
    os = 2
    bs = 5
    max_iter = 50

    t = PTD(filename='banana1', batch_size=[n], mode='w')
    t['x'] = MMT.zeros(shape=(n, ds))
    t['l'] = MMT.zeros(shape=(n, os))
    t['x'] = torch.rand(n, ds)
    t['l'] = foo(t['x'], os) 
    
    v = PTD(filename='banana2', batch_size=[n], mode='w')
    v['x'] = MMT.zeros(shape=(n, ds))
    v['l'] = MMT.zeros(shape=(n, os))
    v['x'] = torch.rand(n, ds)
    v['l'] = foo(t['x'], os) 

    dl = DataLoader(t, batch_size = bs, collate_fn=lambda x:x)
    for d in dl:
        p(d)

    likelihood = MultitaskGaussianLikelihood(os)
    model = GPModel(t['x'], t['l'], likelihood)   

    print('-- training')
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    mll = ExactMarginalLogLikelihood(likelihood, model) 

    for i in range(max_iter):
        optimizer.zero_grad()
        output = model(t['x'])
        loss = -mll(output, t['l'])
        loss.backward()
        print('Iter %d/%d - Loss: %.3f'%(i + 1, max_iter, loss.item()))
        optimizer.step()

    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(v['x']))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

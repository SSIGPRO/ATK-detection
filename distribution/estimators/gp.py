# torch stuff
import torch

# torch stuff# GP stuff
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.kernels import ScaleKernel, MaternKernel, PeriodicKernel, RBFKernel, PolynomialKernel
from gpytorch.distributions import MultivariateNormal

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
        self.lr = kwargs['lr']
        n_classes = kwargs['n_classes'] 
        
        # hyperparams
        self.kernel_kwargs = kwargs['kernel_kwargs']
        self.lh_kwargs = kwargs['likelihood_kwargs'] if 'likelihood_kwargs' in kwargs else {}

        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'

        self.x = self.x.to(self.device)
        self.y = self.y.to(self.device)

        print(self.y)
        # define likelihood (lh)
        self.lh = DirichletClassificationLikelihood(self.y, **self.lh_kwargs, learn_additional_noise = True)
        self.lh = self.lh.to(self.device)

        # The gp model
        self.gp = GPWrap(
                x = self.x,
                y = self.y,
                lh = self.lh,
                n_classes = n_classes,
                kernel_kwargs = self.kernel_kwargs
                )
        self.gp = self.gp.to(self.device)

        # Find optimal model hyperparameters
        self.gp.train()
        self.lh.train()

        # ---- Training related definitions
        self.optimizer = torch.optim.Adam(self.gp.parameters(), lr=self.lr)

        # Loss 
        self.mll = ExactMarginalLogLikelihood(self.lh, self.gp) 
        return
        
    def forward(self, x):
        return self.gp(x)

    def train_iteration(self):
        self.optimizer.zero_grad()
        output = self.gp(self.x)
        loss = -self.mll(output, self.lh.transformed_targets).sum()
        loss.backward()
        self.optimizer.step()
        return loss

    def get_likelihood(self, x):
        return self.lh(self.gp(x))
   
    def AUC_test(self, dl):
        n_samples = dl.dataset['labels'].shape[0]

        with torch.no_grad():
            inputs, targets = dl.collate_fn(dl.dataset) 
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            predictions = self.get_likelihood(inputs)
            mean = predictions.mean
            lower, upper = predictions.confidence_region()
            conf = upper - lower
        scores = conf.mean(dim=1).detach().cpu().numpy()
        labels = targets.detach().cpu().numpy()
                
        return roc_auc_score(labels, scores)

class GPWrap(ExactGP):
    def __init__(self, **kwargs):
        x = kwargs['x']
        y = kwargs['y']
        lh = kwargs['lh']
        n_classes = kwargs['n_classes']
        kernel_kwargs = kwargs['kernel_kwargs']
        
        super(GPWrap, self).__init__(x, y, lh)
        
        batch_shape = torch.Size((n_classes, ))

        # define means module
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        
        # define covariance module
        kernel = \
                MaternKernel(nu=kernel_kwargs['nu'], batch_shape=batch_shape) + \
                PeriodicKernel(batch_shape=batch_shape) + \
                RBFKernel(batch_shape=batch_shape) + \
                PolynomialKernel(power=kernel_kwargs['power'], batch_shape=batch_shape)
            
        self.covar_module = ScaleKernel(
            kernel,
            batch_shape = batch_shape 
        )
    
        return

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

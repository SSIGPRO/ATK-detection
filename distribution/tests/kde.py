from sklearn.neighbors import KernelDensity
import torch
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    n = 10000
    ds = 2
    ns = 5000

    d = torch.zeros(n, ds)
    x = torch.pi*2*torch.rand(n)
    kdes = []
    for i in range(ds):
        d[:,i] = torch.sin(1.5*x)    

    kde = KernelDensity(kernel='gaussian', bandwidth='silverman')
    #print('fitting data: ', d, d.shape)
    kde.fit(d)

    # create samples
    samples = 2*torch.rand(ns, ds)-1

    #print('samples: ', samples)
    scores = kde.score_samples(samples)
    probs = np.exp(scores)
    print('probs: ', probs)

    print('s0: ',
          samples[0,:].reshape(1,-1),
          np.exp(kde.score_samples(samples[0,:].reshape(1,-1)))
          )
    plt.figure()
    plt.hist2d(d[:,0], d[:,1], density=True)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(samples[:,0].flatten(), samples[:,1].flatten(), probs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

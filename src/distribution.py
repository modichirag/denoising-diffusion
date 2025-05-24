import torch
from torch.utils.data import IterableDataset, DataLoader
from sklearn.datasets import make_moons

class DistributionDataLoader:
    _is_my_custom_distribution_data_loader = True
    def __init__(self, distribution, batch_size, fwd_func=None, use_latents=None):
        self.distribution = distribution
        self.batch_size = batch_size
        self.fwd_func = fwd_func
        if self.fwd_func is not None:
            assert isinstance(use_latents, bool), "use_latents should be a boolean when fwd_func is provided"
            self.use_latents = use_latents

    def __iter__(self):
        return self

    def __next__(self):
        samples = self.distribution.sample(self.batch_size)
        if self.fwd_func is None:
            return samples
        else:
            corrupted, latents = self.fwd_func(samples, return_latents=True)
            latents = latents if self.use_latents else None
            return samples, corrupted, latents

    def __len__(self):
        return float('inf')  # Infinite length, as it generates samples on-the-fly

class CheckerDistribution:
    def __init__(self, device='cpu'):
        self.device = device

    def sample(self, n_samples):
        # Generate checkerboard pattern data
        x1 = torch.rand(n_samples) * 4 - 2
        x2_ = torch.rand(n_samples) - torch.randint(2, (n_samples,)) * 2
        x2 = x2_ + (torch.floor(x1) % 2)
        return (torch.cat([x1[:, None], x2[:, None]], 1) * 2).to(self.device)

class MoonDistribution:
    def __init__(self, noise=0.1, shuffle=True, random_state=None, device='cpu'):
        super().__init__()
        self.device = device
        self.noise = noise
        self.shuffle = shuffle
        self.random_state = random_state

    def sample(self, n_samples):
        # Generate moon-shaped data with the specified number of samples
        X_moon, _ = make_moons(n_samples=n_samples,
                              noise=self.noise,
                              shuffle=self.shuffle,
                              random_state=self.random_state)

        # Convert to torch tensor with float32
        return 4.0*(torch.tensor(X_moon, dtype=torch.float32).to(self.device)-0.5)

import math
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.multivariate_normal import  MultivariateNormal

class Prior(torch.nn.Module):
    """
    Abstract class for prior distributions of normalizing flows. The interface
    is similar to `torch.distributions.distribution.Distribution`, but batching
    is treated differently. Parameters passed to constructors are never batched,
    but are aware of the target (single) sample shape. The `forward` method then
    accepts just the batch size and produces a batch of samples of the known
    shape.
    """
    def forward(self, batch_size):
        raise NotImplementedError()
    def log_prob(self, x):
        raise NotImplementedError()
    def draw(self, batch_size):
        """Alias of `forward` to allow explicit calls."""
        return self.forward(batch_size)

class GMM(Prior):
    def __init__(self, loc=None, var=None, scale = 1.0, ndim = None, nmix= None, device='cpu', requires_grad=False):
        super().__init__()

        self.device = device
        self.scale = scale       ### only specify if loc is None
        def _compute_mu(ndim):
            return self.scale*torch.randn((1, ndim))
        self.sample = self.forward

        if loc is None:
            self.nmix = nmix
            self.ndim = ndim
            loc = torch.cat([_compute_mu(ndim) for i in range(1, self.nmix + 1)], dim=0)
            var = torch.stack([1.0*torch.ones((ndim,)) for i in range(nmix)])
        else:
            self.nmix = loc.shape[0]
            self.ndim = loc.shape[1] ### locs should have shape [n_mix, ndim]

        self.loc = loc   ### locs should have shape [n_mix, ndim]
        self.var = var   ### should have shape [n_mix, ndim]

        if requires_grad:
            self.loc.requires_grad_()
            self.var.requires_grad_()

        mix = Categorical(torch.ones(self.nmix,))
        comp = Independent(Normal(
            self.loc, self.var), 1)
        self.dist = MixtureSameFamily(mix, comp)

    def log_prob(self, x):
        logp = self.dist.log_prob(x)
        return logp

    def forward(self, batch_size):
        x = self.dist.sample((batch_size,))
        return x

    def rsample(self, batch_size):
        x = self.dist.rsample((batch_size,))
        return x

distribution_dict = {
    'checker': CheckerDistribution,
    'moon': MoonDistribution,
    'gmm': GMM,
}
import torch
from sklearn.datasets import make_moons

class DistributionDataLoader:
    def __init__(self, distribution, batch_size):
        self.distribution = distribution
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        samples = self.distribution.sample(self.batch_size)
        return samples

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

distribution_dict = {
    'checker': CheckerDistribution,
    'moon': MoonDistribution
}
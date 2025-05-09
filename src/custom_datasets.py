import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

class ImagesOnly(Dataset):
        def __init__(self, base_dataset):
            self.base = base_dataset

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, _ = self.base[idx]
            return img

username = os.getenv('USER')

# 1) Define your transforms
mnist_transforms = transforms.Compose([
    transforms.Pad(2),                              # [0,255]→[0,1]
    transforms.ToTensor(),                              # [0,255]→[0,1]
    transforms.Normalize((0.1307,), (0.3081,))          # mean/std for MNIST
])

mnist_transforms_raw = transforms.Compose([
    transforms.Pad(2),                              # [0,255]→[0,1]
    transforms.ToTensor(),                              # [0,255]→[0,1]
])

mnist_inverse_transforms = transforms.Compose([
    transforms.Normalize(                              # mean/std for CIFAR-10
            mean=(0.),
            std=(1/0.3081)
    ),
    transforms.Normalize(                              # mean/std for CIFAR-10
            mean=(-0.1307),
            std=(1.)
    )
])

cifar10_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),                  # data aug only on train
    #transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(                              # mean/std for CIFAR-10
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
])

cifar10_inverse_transforms = transforms.Compose([
    transforms.Normalize(                              # mean/std for CIFAR-10
            mean=(0., 0., 0.),
            std=(1/0.2470, 1/0.2435, 1/0.2616)
    ),
    transforms.Normalize(                              # mean/std for CIFAR-10
            mean=(-0.4914, -0.4822, -0.4465),
            std=(1., 1., 1.)
    )
])

cifar10_transforms_raw = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])


# 2) Instantiate the datasets
mnist_train = datasets.MNIST(
    root=f"/mnt/ceph/users/{username}/ML_data/mnist",      # download location
    train=True,
    download=True,
    transform=mnist_transforms
)
mnist_test = datasets.MNIST(
    root=f"/mnt/ceph/users/{username}/ML_data/mnist",
    train=False,
    download=True,
    transform=mnist_transforms
)

cifar10_train = datasets.CIFAR10(
    root=f"/mnt/ceph/users/{username}/ML_data/cifar10",
    train=True,
    download=True,
    transform=cifar10_transforms
)

cifar10_test = datasets.CIFAR10(
    root=f"/mnt/ceph/users/{username}/ML_data/cifar10",
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        ),
    ])
)

dataset_dict = {
        'cifar10':[cifar10_train, 32, 3],
        'mnist':[mnist_train, 32, 1]
        }


# dataset classes
class CustomDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)



class NumpyImageDataset(Dataset):
    def __init__(self, array, channel_first=True, transform=None):
        """
        array: np.ndarray, shape (N, H, W, C) or (N, C, H, W)
        transform: torchvision.transforms (expects PIL or Tensor)
        """
        self.array = array
        self.transform = transform
        self.channel_first = channel_first

    def __len__(self):
        return len(self.array)

    def __getitem__(self, idx):
        img = self.array[idx]

        if isinstance(img, np.ndarray):
            if self.channel_first:
                img = torch.from_numpy(img).float()
            else:
                img = torch.from_numpy(img).permute(2,0,1).float()

        if self.transform:
            img = self.transform(img)
        return img


class CorruptedDataset(Dataset):
    def __init__(self, base_dataset, corruption_fn, tied_rng=True, base_seed: int = 0):
        """
        base_dataset   : any Dataset returning (img, label)
        corruption_fn  : fn(img, *, generator) -> img_corrupted
        base_seed      : optional global offset for all seeds
        """
        self.base = base_dataset
        self.corrupt = corruption_fn
        self.base_seed = base_seed
        self.tied_rng = tied_rng

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img = self.base[idx]
        # make a fresh generator, seed it with (base_seed + idx)
        if self.tied_rng:
            gen = torch.Generator()
            gen.manual_seed(self.base_seed + idx)
        else:
            gen = None
        # apply your corruption; it must accept a `generator` kwarg
        img_corrupted, latents = self.corrupt(img, return_latents=True, generator=gen)
        return img, img_corrupted, latents


class ManifoldDataset(Dataset):
    def __init__(self, npz_filepath, epsilon):
        loaded_data = np.load(npz_filepath)
        self.x_data = torch.from_numpy(loaded_data['x']).float()
        self.y_data = torch.from_numpy(loaded_data['y']).float()
        self.A_data = torch.from_numpy(loaded_data['A']).float()
        padded_data = torch.randn((self.x_data.shape[0], self.x_data.shape[1] - self.y_data.shape[1]))
        self.y_data = torch.cat((self.y_data, padded_data), dim=1)

        if not (len(self.x_data) == len(self.y_data) == len(self.A_data)):
            raise ValueError("All arrays must have the same number of samples (first dimension)")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_x = self.x_data[idx]
        sample_y = self.y_data[idx]
        sample_A = self.A_data[idx]
        return sample_x, sample_y, sample_A

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import CelebA
import torch.nn.functional as F
from PIL import Image
import os
from forward_maps import compute_At_y

username = os.getenv('USER')
download_dataset = False  # set to True if you want to download datasets

class ImagesOnly(Dataset):
        def __init__(self, base_dataset):
            self.base = base_dataset

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, _ = self.base[idx]
            return img


class ImageOnlyFolder(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')  # ensure 3 channels
        if self.transform:
            img = self.transform(img)
        return img


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

celebA_inverse_transforms = transforms.Compose([
    transforms.Normalize(                              # mean/std for CIFAR-10
            mean=(0., 0., 0.),
            std=(1/0.5, 1/0.5, 1/0.5)
    ),
    transforms.Normalize(                              # mean/std for CIFAR-10
            mean=(-0.5, -0.5, -0.5),
            std=(1., 1., 1.)
    )
])

cifar10_transforms_raw = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])


# 2) Instantiate the datasets
def get_mnist_train_dataset():
    return datasets.MNIST(
    root=f"/mnt/ceph/users/{username}/ML_data/mnist",      # download location
    train=True,
    download=download_dataset,
    transform=mnist_transforms
)
def get_mnist_test_dataset():
    return datasets.MNIST(
    root=f"/mnt/ceph/users/{username}/ML_data/mnist",
    train=False,
    download=download_dataset,
    transform=mnist_transforms
)

def get_cifar10_train_dataset():
    return datasets.CIFAR10(
    root=f"/mnt/ceph/users/{username}/ML_data/cifar10",
    train=True,
    download=download_dataset,
    transform=cifar10_transforms
)

def get_cifar10_test_dataset():
    return datasets.CIFAR10(
    root=f"/mnt/ceph/users/{username}/ML_data/cifar10",
    train=False,
    download=download_dataset,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        ),
    ])
)

celebA_transforms = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),  # → [-1, 1]
])

# celebA = datasets.CelebA(root="/mnt/ceph/users/cmodi/ML_data/celebA",
#                 split='train',
#                 download=False, transform=celebA_transforms)

def get_celebA_dataset():
    return ImageOnlyFolder(f"/mnt/ceph/users/{username}/ML_data/celebA/img_align_celeba/", \
                         transform=celebA_transforms)

dataset_dict = {
        'cifar10':[get_cifar10_train_dataset, 32, 3],
        'mnist':[get_mnist_train_dataset, 32, 1],
        'celebA':[get_celebA_dataset, 64, 3]
        }


# # dataset classes
# class CustomDataset(Dataset):
#     def __init__(
#         self,
#         folder,
#         image_size,
#         exts = ['jpg', 'jpeg', 'png', 'tiff'],
#         augment_horizontal_flip = False,
#         convert_image_to = None
#     ):
#         super().__init__()
#         self.folder = folder
#         self.image_size = image_size
#         self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

#         maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

#         self.transform = T.Compose([
#             T.Lambda(maybe_convert_fn),
#             T.Resize(image_size),
#             T.RandomHorizontalFlip() if augment_horizontal_flip else torch.nn.Identity(),
#             T.CenterCrop(image_size),
#             T.ToTensor()
#         ])

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, index):
#         path = self.paths[index]
#         img = Image.open(path)
#         return self.transform(img)


class NumpyImageDataset(Dataset):
    def __init__(self, array, channel_first=True, transform=None):
        """
        array: np.ndarray, shape (N, H, W, C) or (N, C, H, W)
        transform: torchvision.transforms (expects PIL or Tensor)
        """
        # self.data = np.load(path, mmap_mode='r')  # doesn't load full file into RAM
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


class NumpyArrayDataset(Dataset):
    def __init__(self, data_array, transform=None):
        self.data = torch.from_numpy(data_array).float()  # or .long() for labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.data[idx])
        else:
            return self.data[idx]


class CombinedNumpyDataset(Dataset):
    def __init__(self, folder, transform=None):
        import os
        files = os.listdir(folder)
        file_list = [os.path.join(folder, f) for f in files if f.endswith('.npy')]
        self.data = [np.load(f) for f in file_list]  # load into memory
        self.cumsum = np.cumsum([len(arr) for arr in self.data])
        self.transform = transform

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, idx):
        # figure out which array and local index
        file_idx = np.searchsorted(self.cumsum, idx, side='right')
        local_idx = idx if file_idx == 0 else idx - self.cumsum[file_idx - 1]
        x = self.data[file_idx][local_idx]
        x = torch.from_numpy(x)
        if self.transform is not None:
            x = self.transform(x)
        return x


class CombinedLazyNumpyDataset(Dataset):
    def __init__(self, folder, transform=None):
        import os
        files = os.listdir(folder)
        file_list = [os.path.join(folder, f) for f in files if f.endswith('.npy')]
        self.files = file_list
        self.lengths = [np.load(f, mmap_mode='r').shape[0] for f in file_list]
        self.cumsum = np.cumsum(self.lengths)
        self.transform = transform

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumsum, idx, side='right')
        local_idx = idx if file_idx == 0 else idx - self.cumsum[file_idx - 1]
        data = np.load(self.files[file_idx], mmap_mode='r')  # no 'with'
        data = torch.from_numpy(data[local_idx])
        if self.transform is not None:
            data = self.transform(data)        
        return data


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
    def __init__(self, npz_filepath, obs_type):
        loaded_data = np.load(npz_filepath)
        self.x_data = torch.from_numpy(loaded_data['x']).float()
        self.y_data_original = torch.from_numpy(loaded_data['y']).float()
        self.A_data = torch.from_numpy(loaded_data['A']).float()
        # y_data_original is 2-dim while y_data is 5-dim
        if obs_type == 'vec':
            self.y_data = compute_At_y(self.A_data, self.y_data_original)
        elif obs_type == 'coeff':
            padded_data = torch.randn((self.x_data.shape[0], self.x_data.shape[1] - self.y_data_original.shape[1]))
            self.y_data = torch.cat((self.y_data_original, padded_data), dim=1)

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

class Manifold_A_Dataset(Dataset):
    def __init__(self, npz_filepath):
        loaded_data = np.load(npz_filepath)
        self.A_data = torch.from_numpy(loaded_data['A']).float()

    def __len__(self):
        return len(self.A_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.A_data[idx]

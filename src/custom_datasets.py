import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class ImagesOnly(Dataset):
        def __init__(self, base_dataset):
            self.base = base_dataset
            
        def __len__(self):
            return len(self.base)
        
        def __getitem__(self, idx):
            img, _ = self.base[idx]
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

cifar10_transforms_raw = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])


# 2) Instantiate the datasets
mnist_train = datasets.MNIST(
    root="/mnt/ceph/users/cmodi/ML_data/mnist",      # download location
    train=True,
    download=True,
    transform=mnist_transforms
)
mnist_test = datasets.MNIST(
    root="/mnt/ceph/users/cmodi/ML_data/mnist",
    train=False,
    download=True,
    transform=mnist_transforms
)

cifar10_train = datasets.CIFAR10(
    root="/mnt/ceph/users/cmodi/ML_data/cifar10",
    train=True,
    download=True,
    transform=cifar10_transforms
)

cifar10_test = datasets.CIFAR10(
    root="/mnt/ceph/users/cmodi/ML_data/cifar10",
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


dataset_dict = {
        'cifar10':[cifar10_train, 32, 3],
        'mnist':[mnist_train, 32, 1]
        }


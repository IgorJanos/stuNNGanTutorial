
import os
import torchvision
import torchvision.datasets as ds
import torchvision.transforms as tvf

from torch.utils.data import DataLoader


DATASET_BASEDIR = ".scratch/data"

def _get_root_dir(name):
    rootdir = os.path.join(DATASET_BASEDIR, name)
    os.makedirs(rootdir, exist_ok=True)
    return rootdir

def get_transform():
    return tvf.Compose([
        tvf.ToTensor(),
        tvf.Normalize(mean=0.5, std=1)        # <0;1> -> <-1;-1>
    ])


def create_MNIST():
    d = ds.MNIST(
        _get_root_dir("mnist"), 
        transform=get_transform(),
        train=True, 
        download=True
        )
    shape = (1, 28, 28)
    return d, shape

def create_FMNIST():
    d = ds.FashionMNIST(
        _get_root_dir("fmnist"), 
        transform=get_transform(),
        train=True, 
        download=True
        )
    shape = (1, 28, 28)
    return d, shape

def create_CIFAR10():
    d = ds.CIFAR10(
        _get_root_dir("cifar10"), 
        transform=get_transform(),
        train=True, 
        download=True
        )
    shape = (3, 32, 32)
    return d, shape


DATASETS = {
    "mnist": create_MNIST,
    "fmnist": create_FMNIST,
    "cifar10": create_CIFAR10
}



class DataSource:
    def __init__(self, args, device):
        self.device = device

        # Create dataset & data loader
        self.ds, self.shape = DATASETS[args.dataset]()
        self.loader = DataLoader(
            self.ds, 
            batch_size=args.batch_size,
            num_workers=2,
            shuffle=True
        )
        self.iterLoader = iter(self.loader)


    def get(self):
        try:
            x, labels = next(self.iterLoader)
        except (OSError, StopIteration):
            # Reset the loader and start from the beginning
            self.iterLoader = iter(self.loader)
            x, labels = next(self.iterLoader)

        return x.to(self.device)
    






from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import pickle
from torch.utils.data import DataLoader
import torchvision.transforms as tfs
from .distribute_dataset import distribute_dataset
from datasets import load_dataset

class SubsetDataset(Dataset):
    def __init__(self, dataset, num_samples):
        self.dataset = dataset
        self.num_samples = min(num_samples, len(dataset))
        self.indices = list(range(self.num_samples))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx >= self.num_samples:
            raise IndexError(
                f"Index {idx} is out of bounds for dataset of size {self.num_samples}"
            )
        return self.dataset[self.indices[idx]]

class TinyImageNet(Dataset):
    def __init__(self, root="./data", train=True, transform=None):
        root = os.path.join(root, "tiny-imagenet")
        if train:
            root = "/mnt/csp/mmvision/home/lwh/DLS/tiny-imagenet_train.pkl"
        else:
            root = "/mnt/csp/mmvision/home/lwh/DLS/tiny-imagenet_val.pkl"
        with open(root, "rb") as f:
            dat = pickle.load(f)
        self.data = dat["data"]
        self.targets = dat["targets"]
        self.transform = transform

    def __getitem__(self, item):
        data, targets = Image.fromarray(self.data[item]), self.targets[item]
        if self.transform is not None:
            data = self.transform(data)
        return data, targets

    def __len__(self):
        return len(self.data)


def load_tinyimagenet(
    root,
    transforms=None,
    image_size=32,
    train_batch_size=64,
    valid_batch_size=64,
    distribute=False,
    split=1.0,
    rank=0,
    seed=666,
    return_dataloader=False,
    debug=False,
):
    if transforms is None:
        transforms = tfs.Compose(
            [
                tfs.Resize((image_size, image_size)),
                tfs.ToTensor(),
                tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    if train_batch_size is None:
        train_batch_size = 1
    if split is None:
        split = [1.0]
    train_set = TinyImageNet(root, train=True, transform=transforms)
    valid_set = TinyImageNet(root, train=False, transform=transforms)
    if debug:
        # only take the first 50 samples
        train_set = SubsetDataset(train_set, train_batch_size)
        valid_set = SubsetDataset(valid_set, 5 * valid_batch_size)
    
    if distribute:
        train_set = distribute_dataset(train_set, split, rank, seed=seed)
    if return_dataloader:
        train_loader = DataLoader(
            train_set, batch_size=train_batch_size, shuffle=False, drop_last=True
        )
        valid_loader = DataLoader(
            valid_set, batch_size=valid_batch_size, drop_last=True
        )

        return train_loader, valid_loader, (3, image_size, image_size), 200
    return train_set, valid_set, (3, image_size, image_size), 200

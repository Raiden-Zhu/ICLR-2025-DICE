import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from .distribute_dataset import distribute_dataset
from torch.utils.data import Dataset


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


def load_cifar100(
    root,
    transforms=None,
    image_size=32,
    train_batch_size=64,
    valid_batch_size=64,
    # distribute=False,
    split=1.0,
    rank=0,
    seed=666,
    return_dataloader=False,
    debug=False,
):
    # print("train batch size",train_batch_size)
    if transforms is None:
        transforms = tfs.Compose(
            [
                tfs.Resize((image_size, image_size)),
                tfs.ToTensor(),
                tfs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]
        )

    if train_batch_size is None:
        train_batch_size = 1
    if split is None:
        split = [1.0]
    train_set = CIFAR100(root, True, transforms, download=True)
    valid_set = CIFAR100(root, False, transforms, download=True)
    # if distribute:
        # train_set = distribute_dataset(
            # train_set, split, rank, seed=seed, dirichlet=True
        # )
    # train_loader = DataLoader(
    #     train_set, batch_size=train_batch_size, shuffle=True, drop_last=True
    # )
    # valid_loader = DataLoader(valid_set, batch_size=valid_batch_size, drop_last=True)
    # return train_loader, valid_loader, (3, image_size, image_size), 10

    if debug:
        # only take the first 50 samples
        train_set = SubsetDataset(train_set, 128)
        valid_set = SubsetDataset(valid_set, 128)

    if return_dataloader:
        train_loader = DataLoader(
            train_set, batch_size=train_batch_size, shuffle=False, drop_last=True
        )
        valid_loader = DataLoader(
            valid_set, batch_size=valid_batch_size, drop_last=True
        )
        # print(f"Batch size: {train_loader.batch_size}")
        # print(f"Batch size: {valid_loader.batch_size}")
        return train_loader, valid_loader, (3, image_size, image_size), 100
    print('return dataloader', return_dataloader)
    return train_set, valid_set, (3, image_size, image_size), 100

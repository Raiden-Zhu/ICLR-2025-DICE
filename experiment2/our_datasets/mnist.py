import torch
from torchvision import datasets, transforms as tfs
from torch.utils.data import DataLoader, Subset
from utils.random import set_seed
def load_mnist(
    root,
    transforms=None,
    image_size=28,
    train_batch_size=50,
    valid_batch_size=50,
    split=1.0,
    rank=0,
    seed=666,
    return_dataloader=False,
    debug=False,
):
    set_seed(seed)
    # 默认的 transforms
    if transforms is None:
        transforms = tfs.Compose(
            [
                tfs.Resize((image_size, image_size)),
                tfs.ToTensor(),
                tfs.Normalize((0.1307,), (0.3081,)),  # MNIST 的均值和标准差
            ]
        )


    train_set = datasets.MNIST(root, train=True, transform=transforms, download=True)
    valid_set = datasets.MNIST(root, train=False, transform=transforms, download=True)


    if debug:
        train_set = Subset(train_set, range(train_batch_size))
        valid_set = Subset(valid_set, range(5*valid_batch_size))

    if return_dataloader:
        train_loader = DataLoader(
            train_set, batch_size=train_batch_size, shuffle=False, drop_last=True
        )
        valid_loader = DataLoader(
            valid_set, batch_size=valid_batch_size, drop_last=True
        )
        return train_loader, valid_loader, (1, image_size, image_size), 10

    return train_set, valid_set, (1, image_size, image_size), 10
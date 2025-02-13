import time
from .cifar10 import load_cifar10
from .tinyimagenet import load_tinyimagenet
from .cifar100 import load_cifar100
from .mnist import load_mnist
def load_dataset(
    root,
    name,
    image_size,
    return_dataloader,
    train_batch_size=64,
    valid_batch_size=64,
    # distribute=False,
    split=None,
    rank=0,
    seed=666,
    debug=False,
):
    if name.lower() == "mnist":
        return load_mnist(root=root,
            image_size=image_size,
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size,
            # distribute=distribute,
            split=split,
            rank=rank,
            seed=seed,
            debug=debug,
            return_dataloader=return_dataloader,)
    
    if name.lower() == "cifar10_test":
        return load_cifar10(
            root=root,
            image_size=image_size,
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size,
            # distribute=distribute,
            split=split,
            rank=rank,
            seed=seed,
            debug=True,
            return_dataloader=return_dataloader,
        )
    if name.lower() == 'cifar100':
        # print('load cifar100')
        return load_cifar100(
            root=root,
            image_size=image_size,
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size,
            # distribute=distribute,
            split=split,
            rank=rank,
            seed=seed,
            return_dataloader=return_dataloader,
            debug=debug,
        )
    if name.lower() == "cifar10":
        # print('load cifar 10')
        return load_cifar10(
            root=root,
            image_size=image_size,
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size,
            # distribute=distribute,
            split=split,
            rank=rank,
            seed=seed,
            debug=debug,
            return_dataloader=return_dataloader,
        )
    if name.lower() == "tinyimagenet":
        return load_tinyimagenet(
            root=root,
            image_size=image_size,
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size,
            # distribute=distribute,
            split=split,
            rank=rank,
            seed=seed,
        )
   
        

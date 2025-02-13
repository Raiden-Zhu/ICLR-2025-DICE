import numpy as np
from torch.utils.data import DataLoader, Subset, RandomSampler
import torch
from tqdm import tqdm
from utils.random import set_seed
import random
import time
import os
import json

def dirichlet_split_noniid(train_labels, alpha, n_clients):
    """
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    """
    n_classes = train_labels.max() + 1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例 K 是 10个类别， N代表N个clients 每个clients拥有该类别数据的比例
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合 10个类别 每个类别对应的所有索引
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # k idcs 是每个类别的所有索引 fracs是每个类别针对所有clients的比例
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        # 这里(np.cumsum(fracs)[:-1]*len(k_idcs))生成每个类别的对不同clients的划分点，然后用划分点来划分每个类别的所有索引

        for i, idcs in enumerate(
            np.split(k_idcs, (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))
        ):
            client_idcs[i] += [idcs]
    # 将每个clients不同类别的索引列表拼成一个列表
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

def dirichlet_split(n, num_classes, dir_alpha):

    # 生成 Dirichlet 分布权重的 numpy 矩阵
    np.random.seed(42)
    weights = np.random.dirichlet([dir_alpha] * num_classes, n)
    print(f"dirichlet weights: {weights}")
    return weights

# nam samples 指的是 原本 dataset 按照 batchsize 为512 划分之后的的length（97）再按照node size 划分的数量 这里是 97//16 = 6
class nonIIDSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_samples, class_weights, nb_classes, limit_samples=1000):
        self.dataset = dataset
        self.num_samples = num_samples
        self.class_weights = class_weights
        self.nb_classes = nb_classes
        if self.num_samples > 200:
            limit_samples = 5000
        self.class_indices = [[] for _ in range(self.nb_classes)]
        with tqdm(total=min(len(dataset), limit_samples), desc="Initializing Sampler") as pbar:
            for idx, (_, label) in enumerate(dataset):
                self.class_indices[label].append(idx)
                pbar.update(1)

    def __iter__(self):
        samples = []
        for _ in range(self.num_samples):
            class_idx = np.random.choice(self.nb_classes, p=self.class_weights)
            sample_idx = np.random.choice(self.class_indices[class_idx])
            samples.append(sample_idx)
        return iter(samples)

    def __len__(self):
        return self.num_samples

def record_datasequence(sampler):
    # 获取采样器生成的数据顺序索引
    sampled_indices = list(sampler)
    sampled_indices = [int(idx) for idx in sampled_indices]
    # 获取当前时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 创建保存目录
    save_dir = "/mnt/csp/mmvision/home/lwh/DLS/datasequence/"
    os.makedirs(save_dir, exist_ok=True)

    # 保存数据顺序索引到 JSON 文件
    file_path = os.path.join(save_dir, f"sampled_indices_{timestamp}.json")
    with open(file_path, 'w') as f:
        json.dump(sampled_indices, f, indent=4)

    print(f"Sampled indices saved to {file_path}")


# Create n dataloaders
def create_dataloaders(dataset, n, samples_per_loader, batch_size=32, all_class_weights=None, nb_class=10):
    dataloaders = []
    for i in range(n):
        # Create a unique class distribution for each dataloader
        if all_class_weights is not None:
            class_weights = all_class_weights[i]
        else:
            class_weights = np.random.dirichlet(np.ones(nb_class))

        sampler = nonIIDSampler(dataset, samples_per_loader, class_weights, nb_class)
        # record_datasequence(sampler)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        dataloaders.append(dataloader)

    return dataloaders


def create_simple_preference(n, nb_class, important_prob=0.5):
    all_class_weights = np.zeros((n, nb_class))
    if nb_class > n:
        nb_important = nb_class // n
    else:
        nb_important = 1
    for i in range(n):
        # generate nb_important int between 0 and nb_class-1 (inclusive)
        important_classes = np.random.randint(0, nb_class, nb_important)
        all_class_weights[i, important_classes] = important_prob / nb_important
        # the rest index which is not in the important_class should be (1-important_prob) / (nb_class - nb_important)
        all_class_weights[i, np.setdiff1d(np.arange(nb_class), important_classes)] = (
            1 - important_prob
        ) / (nb_class - nb_important)
    return all_class_weights


def create_IID_preference(n, nb_class):
    all_class_weights = np.zeros((n, nb_class))
    for i in range(n):
        all_class_weights[i] = np.ones(nb_class) / nb_class
    return all_class_weights


if __name__ == "__main__":
    # from torchvision.datasets import CIFAR10
    # from torchvision import transforms

    # # Create 5 dataloaders with 10000 samples each
    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )

    # # Load CIFAR-10 dataset
    # cifar10 = CIFAR10(root="./data", train=True, download=True, transform=transform)
    # n_loaders = 5
    # samples_per_loader = 10000
    # dataloaders = create_nonIID_dataloaders(cifar10, n_loaders, samples_per_loader)

    # # Verify the class distribution in each dataloader
    # for i, dataloader in enumerate(dataloaders):
    #     class_counts = [0] * 10
    #     for _, labels in dataloader:
    #         for label in labels:
    #             class_counts[label] += 1
    #     print(f"Dataloader {i} class distribution:")
    #     print(class_counts)
    #     print()
    print(create_simple_preference(16, 10))

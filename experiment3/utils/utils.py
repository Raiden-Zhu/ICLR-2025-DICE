import os
import time
import torch
import random
import datetime
import argparse
import numpy as np
import torch.nn as nn
import copy
import json
import re
# set random seed
# def set_seed(args):
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.random.manual_seed(args.seed)
#     if args.device >= 0:
#         torch.cuda.manual_seed(args.seed)
#         torch.cuda.manual_seed_all(args.seed)


def set_seed(seed, nb_devices):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if nb_devices >= 0:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# get ArgumentParser
def get_args():
    parser = argparse.ArgumentParser()

    ## dataset
    parser.add_argument("--dataset_path", type=str, default="datasets")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="CIFAR10",
        choices=["CIFAR10", "TinyImageNet"],
    )
    parser.add_argument("--image_size", type=int, default=32, help="input image size")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_swap", type=int, default=None)

    # mode parameter
    parser.add_argument("--mode", type=str, default="csgd")
    parser.add_argument(
        "--shuffle", type=str, default="fixed", choices=["fixed", "random"]
    )
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--port", type=int, default=29500)
    parser.add_argument("--backend", type=str, default="gloo")
    # deep model parameter
    parser.add_argument(
        "--model",
        type=str,
        default="ResNet18",
        choices=["ResNet18", "AlexNet", "DenseNet"],
    )

    # optimization parameter
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="weight decay")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--epoch", type=int, default=6000)
    parser.add_argument(
        "--early_stop", type=int, default=6000, help="w.r.t., iterations"
    )
    parser.add_argument("--milestones", type=int, default=[2400, 4800])
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    return args


def add_identity(args, dir_path):
    # set output file name and path, etc.
    args.identity = (
        f"{args.dataset_name}s{args.image_size}-"
        + f"{args.batch_size}-"
        + f"{args.mode}-"
        + f"{args.shuffle}-"
        +
        # f"{args.affine_type}-"+
        # f"{args.loc_step}-"+
        f"{args.size}-"
        + f"{args.model}-"
        + f"{args.pretrained}-"
        + f"{args.lr}-"
        + f"{args.wd}-"
        + f"{args.gamma}-"
        + f"{args.momentum}-"
        + f"{args.warmup_step}-"
        + f"{args.seed}-"
        + f"{args.amp}"
    )
    args.logs_perf_dir = os.path.join(dir_path, "logs_perf")
    if not os.path.exists(args.logs_perf_dir):
        os.mkdir(args.logs_perf_dir)
    args.perf_data_dir = os.path.join(args.logs_perf_dir, args.dataset_name)
    if not os.path.exists(args.perf_data_dir):
        os.mkdir(args.perf_data_dir)
    args.perf_xlsx_dir = os.path.join(args.perf_data_dir, "xlsx")
    args.perf_imgs_dir = os.path.join(args.perf_data_dir, "imgs")
    args.perf_dict_dir = os.path.join(args.perf_data_dir, "dict")
    args.perf_best_dir = os.path.join(args.perf_data_dir, "best")

    args.logs_runs_dir = os.path.join(dir_path, "logs_runs")
    if not os.path.exists(args.logs_runs_dir):
        os.mkdir(args.logs_runs_dir)
    args.runs_data_dir = os.path.join(args.logs_runs_dir, args.dataset_name)
    if not os.path.exists(args.runs_data_dir):
        os.mkdir(args.runs_data_dir)

    return args


def eval_vision(worker, train_loader, valid_loader, epoch, iteration, tb, device):
    criterion = nn.CrossEntropyLoss()
    worker.model.train()

    print(f"\r")
    total_loss, total_correct, total, step = 0, 0, 0, 0
    start = datetime.datetime.now()
    for batch in train_loader:
        step += 1
        if isinstance(batch[1], (int)):
            batch  = list(batch)
            batch[1] = torch.tensor([batch[1]], dtype=torch.long)
        data, target = batch[0].to(device), batch[1].to(device)
        # print('data', data.shape)
        # print('target', target.shape)
        # print('len of dataloader', len(train_loader))
        output = worker.model(data)
        p = torch.softmax(output, dim=1).argmax(1)
        total_correct += p.eq(target).sum().item()
        total += len(target)
        loss = criterion(output, target)
        total_loss += loss.item()
        end = datetime.datetime.now()
        print(
            f"\r" + f"| Evaluate Train | step: {step}, time: {(end - start).seconds}s",
            flush=True,
            end="",
        )
    total_train_loss = total_loss / step
    total_train_acc = total_correct / total

    print(f"\r")
    total_loss, total_correct, total, step = 0, 0, 0, 0
    total_loss_sum = torch.tensor(0.0, device=device)
    for batch in valid_loader:
        step += 1
        data, target = batch[0].to(device), batch[1].to(device)
        output = worker.model(data)
        p = torch.softmax(output, dim=1).argmax(1)
        total_correct += p.eq(target).sum().item()
        total += len(target)
        loss = criterion(output, target)
        total_loss += loss.item()
        total_loss_sum += loss
        end = datetime.datetime.now()
        print(
            f"\r| Evaluate Valid | step: {step}, time: {(end - start).seconds}s",
            flush=True,
            end="",
        )
    
    total_valid_loss = total_loss / step
    total_valid_acc = total_correct / total
    total_valid_loss_sum = total_loss_sum / step
    worker.loss_mode3.append(total_valid_loss)
    worker.optimizer.zero_grad()
    total_valid_loss_sum.backward()
    params1 = list(worker.model.parameters())
    worker.grads_after_merge = [p.grad for p in params1]
    worker.optimizer.zero_grad()
    
    
    
    if epoch is None:
        tb.add_scalar(
            "valid loss - train loss", total_valid_loss - total_train_loss, iteration
        )
        tb.add_scalar("valid loss", total_valid_loss, iteration)
        tb.add_scalar("train loss", total_train_loss, iteration)
        tb.add_scalar("valid acc", total_valid_acc, iteration)
        tb.add_scalar("train acc", total_train_acc, iteration)
    else:
        tb.add_scalar(
            "valid loss - train loss", total_valid_loss - total_train_loss, epoch
        )
        tb.add_scalar("valid loss", total_valid_loss, epoch)
        tb.add_scalar("train loss", total_train_loss, epoch)
        tb.add_scalar("valid acc", total_valid_acc, epoch)
        tb.add_scalar("train acc", total_train_acc, epoch)

    return total_train_acc, total_train_loss, total_valid_acc, total_valid_loss

def eval_vision_amp(model, train_loader, valid_loader, epoch, iteration, tb, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    print(f"\r")
    total_loss, total_correct, total, step = 0, 0, 0, 0
    start = datetime.datetime.now()
    for batch in train_loader:
        step += 1
        data, target = batch[0].to(device), batch[1].to(device)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            output = model(data)
            p = torch.softmax(output, dim=1).argmax(1)
            total_correct += p.eq(target).sum().item()
            total += len(target)
            loss = criterion(output, target)
            total_loss += loss.item()
        end = datetime.datetime.now()
        print(
            f"\r" + f"| Evaluate Train | step: {step}, time: {(end - start).seconds}s",
            flush=True,
            end="",
        )
    total_train_loss = total_loss / step
    total_train_acc = total_correct / total

    print(f"\r")
    total_loss, total_correct, total, step = 0, 0, 0, 0
    for batch in valid_loader:
        step += 1
        data, target = batch[0].to(device), batch[1].to(device)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            output = model(data)
            p = torch.softmax(output, dim=1).argmax(1)
            total_correct += p.eq(target).sum().item()
            total += len(target)
            loss = criterion(output, target)
            total_loss += loss.item()
        end = datetime.datetime.now()
        print(
            f"\r| Evaluate Valid | step: {step}, time: {(end - start).seconds}s",
            flush=True,
            end="",
        )
    total_valid_loss = total_loss / step
    total_valid_acc = total_correct / total

    if epoch is None:
        tb.add_scalar(
            "valid loss - train loss", total_valid_loss - total_train_loss, iteration
        )
        tb.add_scalar("valid loss", total_valid_loss, iteration)
        tb.add_scalar("train loss", total_train_loss, iteration)
        tb.add_scalar("valid acc", total_valid_acc, iteration)
        tb.add_scalar("train acc", total_train_acc, iteration)
    else:
        tb.add_scalar(
            "valid loss - train loss", total_valid_loss - total_train_loss, epoch
        )
        tb.add_scalar("valid loss", total_valid_loss, epoch)
        tb.add_scalar("train loss", total_train_loss, epoch)
        tb.add_scalar("valid acc", total_valid_acc, epoch)
        tb.add_scalar("train acc", total_train_acc, epoch)

    return total_train_acc, total_train_loss, total_valid_acc, total_valid_loss

def get_secondneighbor(self_rank, neighbormatrix):
    assert neighbormatrix.shape == (16, 16), "neighbormatrix have to be 16x16"

    first_neighbors = torch.nonzero(neighbormatrix[:, self_rank], as_tuple=True)[0]

  
    second_neighbor_dict = {}

 
    for first_neighbor in first_neighbors:

        second_neighbors = torch.nonzero(neighbormatrix[:, first_neighbor], as_tuple=True)[0]



        second_neighbor_weights = {
            int(neighbor): float(neighbormatrix[neighbor, first_neighbor])
            for neighbor in second_neighbors
        }

        second_neighbor_dict[int(first_neighbor)] = second_neighbor_weights

    return second_neighbor_dict

def get_thirdneighbor(self_rank, neighbormatrix):

    assert neighbormatrix.shape == (16, 16), "neighbormatrix 必须是 16x16 的张量"

  
    first_neighbors = torch.nonzero(neighbormatrix[:, self_rank], as_tuple=True)[0]

  
    third_neighbor_dict = {}


    for first_neighbor in first_neighbors:
     
        
        second_neighbors = torch.nonzero(neighbormatrix[:, first_neighbor], as_tuple=True)[0]

    
        second_neighbor_dict = {}

  
        for second_neighbor in second_neighbors:
            

         
            third_neighbors = torch.nonzero(neighbormatrix[:, second_neighbor], as_tuple=True)[0]

            third_neighbor_weights = {
                int(neighbor): float(neighbormatrix[neighbor, second_neighbor]) 
                for neighbor in third_neighbors
            }

       
            second_neighbor_dict[int(second_neighbor)] = third_neighbor_weights

      
        third_neighbor_dict[int(first_neighbor)] = second_neighbor_dict

    return third_neighbor_dict

def generate_P(mode, size):
    result = torch.zeros((size, size))
    if mode == "all":
        result = torch.ones((size, size)) / size
    elif mode == "single":
        for i in range(size):
            result[i][i] = 1
    elif mode == "ring":
        for i in range(size):
            result[i][i] = 1 / 3
            result[i][(i - 1 + size) % size] = 1 / 3
            result[i][(i + 1) % size] = 1 / 3
    elif mode == "right":
        for i in range(size):
            result[i][i] = 1 / 2
            result[i][(i + 1) % size] = 1 / 2
    elif mode == "star":
        for i in range(size):
            result[i][i] = 1 - 1 / size
            result[0][i] = 1 / size
            result[i][0] = 1 / size
    elif mode == "meshgrid":
        assert size > 0
        i = int(np.sqrt(size))
        while size % i != 0:
            i -= 1
        shape = (i, size // i)
        nrow, ncol = shape
        # print(shape, flush=True)
        topo = np.zeros((size, size))
        for i in range(size):
            topo[i][i] = 1.0
            if (i + 1) % ncol != 0:
                topo[i][i + 1] = 1.0
                topo[i + 1][i] = 1.0
            if i + ncol < size:
                topo[i][i + ncol] = 1.0
                topo[i + ncol][i] = 1.0
        topo_neighbor_with_self = [np.nonzero(topo[i])[0] for i in range(size)]
        for i in range(size):
            for j in topo_neighbor_with_self[i]:
                if i != j:
                    topo[i][j] = 1.0 / max(
                        len(topo_neighbor_with_self[i]), len(topo_neighbor_with_self[j])
                    )
            topo[i][i] = 2.0 - topo[i].sum()
        result = torch.tensor(topo, dtype=torch.float)
    elif mode == "exponential":
        x = np.array([1.0 if i & (i - 1) == 0 else 0 for i in range(size)])
        x /= x.sum()
        topo = np.empty((size, size))
        for i in range(size):
            topo[i] = np.roll(x, i)
        result = torch.tensor(topo, dtype=torch.float)
    # print(result, flush=True)
    elif mode == "random":
        result = None
    elif 'special' in mode:
        # 第八个节点和第十一个节点连六个
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        result = torch.tensor(         [[0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                                        [0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.6, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
                                        [0.0, 0.0, 0.1, 0.7, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.2, 0.0, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.3, 0.0, 0.0, 0.0, 0.1, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.6, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
                                        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.6, 0.0, 0.2, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.2, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                                        [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.5, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.2],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1],
                                        [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.6]], dtype=torch.float)
        
        # result = torch.tensor(         [[0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                                 [0.3, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                                 [0.0, 0.0, 0.4, 0.0, 0.1, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0],
        #                                 [0.4, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                                 [0.0, 0.1, 0.0, 0.0, 0.6, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                                 [0.3, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                                 [0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                                 [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                                 [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.6, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                                 [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
        #                                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.1, 0.2],
        #                                 [0.0, 0.2, 0.0, 0.0, 0.0, 0.1, 0.0, 0.4, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
        #                                 [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.4, 0.1, 0.0, 0.0],
        #                                 [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0],
        #                                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.4, 0.1],
        #                                 [0.3, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.3]], dtype=torch.float)
        '''
        result = torch.tensor([     [0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.3, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.5, 0.0, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
                                    [0.4, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.1, 0.0, 0.0, 0.6, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.3, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.1],
                                    [0.2, 0.0, 0.0, 0.2, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.6, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.1, 0.2],
                                    [0.0, 0.2, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0],
                                    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.1, 0.0, 0.0],
                                    [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0],
                                    [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.5, 0.3],
                                    [0.3, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6]], dtype=torch.float)
        
        result = torch.tensor([[0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                               [0.3, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.2, 0.1, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0],
                               [0.2, 0.0, 0.1, 0.6, 0.1, 0.0, 0.0, 0.0],
                               [0.2, 0.0, 0.0, 0.1, 0.6, 0.1, 0.0, 0.0],
                               [0.2, 0.0, 0.0, 0.0, 0.1, 0.6, 0.1, 0.0],
                               [0.2, 0.0, 0.0, 0.0, 0.0, 0.1, 0.6, 0.1],
                               [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.6]], dtype=torch.float)
        
        result = torch.tensor([[0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                               [0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.8, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
                               [0.8, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0],
                               [0.8, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0],
                               [0.8, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0],
                               [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1],
                               [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1]], dtype=torch.float)
        '''
        match = re.search(r'\d+$', mode)
        shift = match.group()
        shift = int(shift)
        if shift > 0:
            result = torch.roll(result, shifts=shift, dims=1)
    return result


def PermutationMatrix(size):
    IdentityMatrix = np.identity(size)
    Permutation = list(range(size))
    np.random.shuffle(Permutation)
    PermutedMatrix = np.take(IdentityMatrix, Permutation, axis=0)

    return PermutedMatrix


def update_csgd(worker_list, center_model):
    for worker in worker_list:
        worker.model.load_state_dict(center_model.state_dict())
        try:
            old_accuracy = worker.get_accuracy(worker.model)
            print(f'rank of {worker.rank} old acc', old_accuracy)
        except:
            pass
        
        worker.step()
        worker.update_grad()
        new_accuracy = worker.get_accuracy(worker.model)
        print(f'rank of {worker.rank} new acc', new_accuracy)


def update_dqn_chooseone(worker_list, iteration, wandb, merge_step=1):
    worker_list_model = [copy.deepcopy(i.model) for i in worker_list]
    for worker in worker_list:
        worker.get_workerlist(worker_list_model)
        worker.step()
        worker.update_grad()
        old_accuracy = worker.get_accuracy(worker.model)
        # wandb.log({f"acc_{worker.rank}": old_accuracy})
        # writein_file(old_accuracy, wandb.name, worker.rank)
        if iteration % merge_step == 0:
            worker.train_step_dqn()
            new_accuracy = worker.get_accuracy(worker.model)
            # wandb.log({f"acc_{worker.rank}": new_accuracy})
            # writein_file(new_accuracy, wandb.name, worker.rank)
            worker.store_buffer(old_accuracy, new_accuracy)

def update_dqn_chooseone_debug_2(worker_list, iteration, wandb, merge_step=1):
    worker_list_model = [copy.deepcopy(i.model) for i in worker_list]
    for worker in worker_list:
        worker.get_workerlist(worker_list_model)
        worker.step()
        worker.update_grad()
        old_accuracy = worker.get_accuracy(worker.model)
        print(f'rank of {worker.rank} old acc', old_accuracy)
        # wandb.log({f"acc_{worker.rank}": old_accuracy})
        # writein_file(old_accuracy, wandb.name, worker.rank)
        worker.step_mergemodel_random(worker_list_model)
        new_accuracy = worker.get_accuracy(worker.model)
        print(f'rank of {worker.rank} new acc', new_accuracy)
        if iteration % merge_step == -1:
            worker.train_step_dqn()
            new_accuracy = worker.get_accuracy(worker.model)
            wandb.log({f"acc_{worker.rank}": new_accuracy})
            writein_file(new_accuracy, wandb.name, worker.rank)
            worker.store_buffer(old_accuracy, new_accuracy)

def update_dqn_chooseone_debug(worker_list, center_model):
    for worker in worker_list:
        # worker.model.load_state_dict(center_model.state_dict())
        try:
            old_accuracy = worker.get_accuracy(worker.model)
            print(f'rank of {worker.rank} old acc', old_accuracy)
        except:
            pass
        
        worker.step()
        worker.update_grad()
        new_accuracy = worker.get_accuracy(worker.model)
        print(f'rank of {worker.rank} new acc', new_accuracy)

def random_p(size):
    P = torch.zeros((size, size))
    P.fill_diagonal_(0.5) 

    for i in range(size):
     
        random_col = random.choice([j for j in range(size) if j != i])
        P[i, random_col] = 0.5
    return P
        

def update_dsgd(worker_list, P, args, probe_valid_loader):
    
    if P is None:
        P = random_p(args.size)
        
    P_perturbed = (
        P
        if args.shuffle == "fixed"
        else np.matmul(
            np.matmul(PermutationMatrix(args.size).T, P), PermutationMatrix(args.size)
        )
    )
    
 
    for worker in worker_list:
        
        worker.step(probe_valid_loader)         
    
        if worker.current_batch_index == worker.choose_batch and worker.now_epoch == worker.choose_epoch and worker.train_to_end == False:
            continue
        
        worker.update_grad()

def merge_model(worker_list, P):
    model_dict_list = [worker.model.state_dict() for worker in worker_list]
    for worker in worker_list:
        for name, param in worker.model.named_parameters():
            param.data = torch.zeros_like(param.data)
            for i in range(worker.size):
                p = P[worker.rank][i]
                param.data += model_dict_list[i][name].data * p

def merge_without_update_old(worker_list, P, args, probe_valid_loader):
    if worker_list[0].current_batch_index == worker_list[0].choose_batch and worker_list[0].train_to_end == False:
        new_worker_list = copy.deepcopy(worker_list)    
        model_dict_list = [worker.model.state_dict() if worker.rank != worker.choose_node else worker.statedict_before_batch for worker in worker_list]
        for worker in new_worker_list:
            for name, param in worker.model.named_parameters():
                param.data = torch.zeros_like(param.data)
                for i in range(worker.size):
                    p = P[worker.rank][i]
                    param.data += model_dict_list[i][name].data * p
        neighbor_worker = list()
        for i in range(args.size):
            p = P[i][args.choose_node]
            if p != 0 and i != args.choose_node:
                # print(f"neighbor node: {i}")
                neighbor_worker.append(new_worker_list[i])
        for worker in neighbor_worker:
            worker.eval(probe_valid_loader, 2)
    else:
        pass

def merge_without_update(worker_list, P, args, probe_valid_loader):
    if worker_list[0].current_batch_index == worker_list[0].choose_batch and worker_list[0].train_to_end == False:
        neighbor_worker = list()
        for i in range(args.size):
            p = P[i][args.choose_node]
            if p != 0 and i != args.choose_node:
                # print(f"neighbor node: {i}")
                neighbor_worker.append(worker_list[i])
        neighbor_worker_statedict = [worker.model.state_dict() for worker in neighbor_worker]
        model_dict_list = [worker.statedict_before_batch if worker.rank == worker.choose_node else worker.model.state_dict() for worker in worker_list]
        for worker in neighbor_worker:
            for name, param in worker.model.named_parameters():
                param.data = torch.zeros_like(param.data)
                for i in range(worker.size):
                    p = P[worker.rank][i]
                    param.data += model_dict_list[i][name].data * p
            worker.eval(probe_valid_loader, 2)
        for target, state_dict in zip(neighbor_worker, neighbor_worker_statedict):
            target.model.load_state_dict(state_dict)
    else:
        pass
    
        
        
        

def update_center_model(worker_list):
    center_model = copy.deepcopy(worker_list[0].model)
    for name, param in center_model.named_parameters():
        for worker in worker_list[1:]:
            param.data += worker.model.state_dict()[name].data
        param.data /= len(worker_list)
    return center_model


def evaluate_and_log(
    probe_train_loader,
    probe_valid_loader,
    iteration,
    epoch,
    writer,
    args,
    wandb,
    mode,
    worker_list,
    train_dataloaders,
    valid_dataloaders,
    P
):
    start_time = datetime.datetime.now()
    if iteration == 0:
        iteration = 1
    
    choose_worker = worker_list[args.choose_node]
    neighbor_worker = list()
    neighbor_weight = list()
    for i in range(args.size):
        if P[i][args.choose_node] != 0 and i != args.choose_node:
            neighbor_weight.append(P[i][args.choose_node])
            neighbor_worker.append(worker_list[i])
            # print(f"neighbor worker: {i}")
            # print(f"neighbor weight: {p}")
            # print(f'weight {P[i][args.choose_node]}')
    
    trainacc_list = list()
    validacc_list = list()
    for worker in worker_list:
        
        train_acc, train_loss, valid_acc, valid_loss = eval_vision(
            worker,
            probe_train_loader,
            probe_valid_loader,
            None,
            iteration,
            writer,
            args.device,
        )
        trainacc_list.append(train_acc)
        validacc_list.append(valid_acc)
   
    estimation, dot1, dot2, _, self_influence, firstnei_influence = compute_estimation(choose_worker, neighbor_weight, neighbor_worker)
    
    return train_acc, train_loss, valid_acc, valid_loss, estimation, dot1, dot2, trainacc_list, validacc_list, self_influence, firstnei_influence

def compute_estimation(choose_worker, neighbor_weight, neighbor_worker):
    dot_list = list()
    lr = choose_worker.current_lr
    dot_product_choosenode = sum(torch.sum(g1 * g2) for g1, g2 in zip(choose_worker.grads_before_choosebatch, choose_worker.grads_train))
    sum_neibornodes_dotproduct = 0
    dot_list.append(dot_product_choosenode.item())
    print(f' choose worker {choose_worker.rank}')
    self_influence = {f'{choose_worker.rank}':(-1) * dot_product_choosenode.item()}
    firstnei_influence = dict()
    for idx, worker in enumerate(neighbor_worker):
    
        # print('worker grads before choose batch', worker.grads_after_merge)
        # print('dot product',sum(torch.sum(g1 * g2) for g1, g2 in zip(worker.grads_after_merge, choose_worker.grads_before_choosebatch)))
        dot_ = sum(torch.sum(g1 * g2) for g1, g2 in zip(worker.grads_after_merge, choose_worker.grads_train)) * neighbor_weight[idx]
        sum_neibornodes_dotproduct += dot_
        dot_list.append(dot_.item())
        firstnei_influence[worker.rank] = dot_.item() * (-1)
        # print('sum_neibornodes_dotproduct',sum_neibornodes_dotproduct)
    dot1 = (-1) * lr * dot_product_choosenode.item()
    dot2 = (-1) * lr * sum_neibornodes_dotproduct.item()
    estimation = lr * ((-1) * dot_product_choosenode - sum_neibornodes_dotproduct)
    return estimation, dot1, dot2, dot_list, self_influence, firstnei_influence

  
def evaluate_last(worker_list, P, args, valid_loader):
    choose_worker = worker_list[args.choose_node]
    neighbor_worker = list()
    loss_list = []
    loss_every_epoch = dict()
    loss_every_epoch['choose_node'] = list()
    loss_every_epoch['neighbor_node'] = list()
    for i in range(args.size):
        p = P[i][args.choose_node]
        if p != 0 and i != args.choose_node:
            # print(f"neighbor node: {i}")
            neighbor_worker.append(worker_list[i])
    for worker in neighbor_worker:
        worker.eval(valid_loader, 4)
        loss_list.append(worker.total_loss)
        loss_every_epoch['neighbor_node'].append([worker.loss_mode3, worker.loss_mode2])
    choose_worker.eval(valid_loader, 4)
    loss_every_epoch['choose_node'].append([choose_worker.loss_mode0, choose_worker.loss_mode1])
    loss_list.append(choose_worker.total_loss)
    return loss_list, loss_every_epoch
    
    
def save_model(center_model, train_acc, epoch, args, log_id):
    state = {"acc": train_acc, "epoch": epoch, "state_dict": center_model.state_dict()}
    if not os.path.exists(args.perf_dict_dir):
        os.mkdir(args.perf_dict_dir)
    torch.save(state, os.path.join(args.perf_dict_dir, f"{log_id}.t7"))

def writein_file(acc, name, rank):
    run_path = "./variable_record/"
    if not os.path.exists(run_path):
    # 如果文件夹不存在，则创建它
        os.makedirs(run_path)
        print(f"Folder '{run_path}' created.")
    
    file_path = os.path.join(run_path, name)
    if not os.path.exists(file_path):
    # 如果文件夹不存在，则创建它
        os.makedirs(file_path)
        print(f"Folder '{file_path}' created.")
    
    rank_file = os.path.join(file_path, f"{rank}.txt")
    with open(rank_file, 'w') as file:
        # 写入内容
        file.write(f"{acc}\n")
        file.write("This is a new text file.")

class Merge_History:
    def __init__(self, size, length):
        self.size = size
        self.length = length
        self.history =  [[[0 for _ in range(self.size)] for _ in range(self.size)] for _ in range(self.length)]
        self.pointer = 0
        self.time = 0
        
    def pointer_step(self):
        if self.pointer ==self.length - 1:
            self.pointer = 0
        else:
            self.pointer += 1 

    def add_history(self, eval_list):
        self.history[self.pointer] = eval_list
        self.pointer_step()
    
def second_largest_index(lst):
    if len(lst) < 2:
        raise ValueError("at lease two elements")

    # 初始化最大值和第二大值
    max_value = max(lst[0], lst[1])
    second_max_value = min(lst[0], lst[1])
    max_index = lst.index(max_value)
    second_max_index = lst.index(second_max_value)

    # 遍历列表找到第二大的元素
    for i in range(2, len(lst)):
        if lst[i] > max_value:
            second_max_value = max_value
            second_max_index = max_index
            max_value = lst[i]
            max_index = i
        elif lst[i] > second_max_value and lst[i] != max_value:
            second_max_value = lst[i]
            second_max_index = i

    return second_max_index

def get_sorted_indices(lst):
    # 使用 enumerate 获取元素及其索引的元组列表
    indexed_list = list(enumerate(lst))
    # 按元素值从大到小排序
    sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
    # 提取排序后的索引
    sorted_indices = [index for index, value in sorted_indexed_list]
    return sorted_indices
  
def choose(eval_result, history, pointer, choose_which):
    
    sequence = get_sorted_indices(eval_result)
    choose_index = sequence[choose_which]
    return choose_index

def choose_merge(worker, eval_result, model_dict_list, history, pointer, second_max):
        max_index = choose(eval_result, history, pointer, second_max)
        for name, param in worker.model.named_parameters():
            param.data += model_dict_list[max_index][name].data
            param.data /= 2
        return max_index

def record_info(eval_all, action, choose_which):
    # 打开一个文件以进行写入操作（如果文件不存在，会创建新文件；如果文件存在，会覆盖原有内容）
    with open(f'./heuristic2_record_choose{choose_which}.json', 'a') as file:
        content = {"eval": eval_all, "worker0": action[0], "worker1": action[1], "worker2": action[2], "worker3": action[3], "worker4": action[4]}
        json.dump(content, file, indent=4)
     

def update_heuristic(worker_list, args, merge_history):
    model_dict_list = [worker.model.state_dict() for worker in worker_list]
    eval_all = list()
    action = list()
    merge_history.time += 1
    for worker in worker_list:
        eval_result = list()
        worker.step()
        worker.update_grad()
        old_acc = worker.get_accuracy(worker.model)
        
        if merge_history.time % 20 == 0:
            for model_state_dict in model_dict_list:
                worker_model = copy.deepcopy(worker.model)
                for name, param in worker_model.named_parameters():
                        param.data += model_state_dict[name].data
                        param.data /= 2
                new_acc = worker.get_accuracy(worker_model)
                acc_improve = new_acc - old_acc
                eval_result.append(acc_improve)
            act = choose_merge(worker, eval_result, model_dict_list, merge_history.history, merge_history.pointer)
            eval_all.append(eval_result)
            action.append(act)
    if merge_history.time % 20 == 0:
        record_info(eval_all, action)
        merge_history.add_history(eval_all)
    
    # merge and step
def update_heuristic_2(worker_list, args, merge_history, choose_which):    
    # get weights of all the models before updating models 
    model_dict_list = [worker.model.state_dict() for worker in worker_list]
    # list to store eval results
    eval_all = list()
    action = list()
    merge_history.time += 1
    # for each worker, update the model and get the eval result
    for worker in worker_list:
        eval_result = list()
        # every 20 step merge model
        if merge_history.time % 20 == 0:
            old_acc = worker.get_accuracy(worker.model)
            print(f'old acc: {old_acc}')
            # for every model weights in model_dict_list, merge it with current model and get the new acc
            for model_state_dict in model_dict_list:
                worker_model = copy.deepcopy(worker.model)
                for name, param in worker_model.named_parameters():
                        param.data += model_state_dict[name].data
                        param.data = param.data / 2
                        
                new_acc = worker.get_accuracy(worker_model)
                print(f'new acc: {new_acc}')
                acc_improve = new_acc - old_acc
                eval_result.append(acc_improve)
            # select the action by eval result and choose_which hyperparameter
            act = choose_merge(worker, eval_result, model_dict_list, merge_history.history, merge_history.pointer, choose_which)
            eval_all.append(eval_result)
            action.append(act)
        worker.step()
        worker.update_grad()
    if merge_history.time % 20 == 0:
        record_info(eval_all, action, choose_which)
        merge_history.add_history(eval_all)
        
def update_heuristic_3(worker_list, args, merge_history, choose_which):    
    model_dict_list = [worker.model.state_dict() for worker in worker_list]
    eval_all = list()
    action = list()
    merge_history.time += 1
    for worker in worker_list:
        eval_result = list()
        if merge_history.time % 20 == 0:
            old_acc = worker.get_accuracy(worker.model)
            for model_state_dict in model_dict_list:
                worker_model = copy.deepcopy(worker.model)
                for name, param in worker_model.named_parameters():
                        param.data += model_state_dict[name].data
                        param.data = param.data / 2
                
                newworker = copy.deepcopy(worker)
                newworker.model = worker_model
                for i in range(0,20):
                    try:
                        newworker.step()
                        newworker.update_grad()
                    except:
                        break
                new_acc = newworker.get_accuracy(newworker.model)
                acc_improve = new_acc - old_acc
                eval_result.append(acc_improve)
            act = choose_merge(worker, eval_result, model_dict_list, merge_history.history, merge_history.pointer, choose_which)
            eval_all.append(eval_result)
            action.append(act)
        worker.step()
        worker.update_grad()
    if merge_history.time % 20 == 0:
        record_info(eval_all, action, choose_which)
        merge_history.add_history(eval_all)
        
def compute_loss_every_epoch(loss):
    sum_ = 0
    loss_every_epoch_list = list()
    loss1 = loss['choose_node']
    print(loss1)
    k = sum((x - y) for x, y in zip(loss1[0][0], loss1[0][1]))
    sum_ += k
    loss_every_epoch_list.append(k)
    loss2 = loss['neighbor_node']
    for i in range(len(loss2)):
        k = sum((x - y) for x, y in zip(loss2[i][0], loss2[i][1]))
        sum_ += k
        loss_every_epoch_list.append(k)
    return sum_, loss_every_epoch_list

def search_model(worker_list, P, args, probe_valid_loader):

    neighbor_worker = list()
    for i in range(args.size):
        p = P[i][args.choose_node]
        if p != 0 and i != args.choose_node:
            # print(f"neighbor node: {i}")
            neighbor_worker.append(worker_list[i])
 
    examine_list = list()
    for worker in neighbor_worker:
        search_list = list()
        result = dict()
        result['self rank'] = worker.rank
        worker.eval(probe_valid_loader, 1)
        for i in range(args.size):
            p = P[worker.rank][i]
            if p != 0 and i != worker.rank:
                search_list.append(worker_list[i])
        for examiner in search_list:
      
            dot_ = sum(torch.sum(g1 * g2) for g1, g2 in zip(worker.grads_after_choosebatch, examiner.grads_train)).item()
            
            result[f'neighbor:rank{examiner.rank}'] = dot_
        result[f'bad node:'] = args.choose_node  
        examine_list.append(result)
    return examine_list
    
def eval_across_workers(worker_list, P, args, probe_valid_loader):
    # first get the gradients of every worker on valid data
    # compute estimation for every worker
    # record estimation of all in a list
    estimation_list_every_node = list()
    record = dict()
    for worker in worker_list:
        worker.eval(probe_valid_loader, 3)
    for (idx, worker) in enumerate(worker_list):
        if worker.rank == args.choose_node:
            neighbor_worker = list()
            neighbor_weight = list()
            for i in range(args.size):
                # if P[i][idx] != 0 and i != idx:
                if P[i][idx] != 0:
                    neighbor_weight.append(P[i][idx])
                    neighbor_worker.append(worker_list[i])
                
            estimation, dot1, dot2, dotlist, self_influence, firstnei_influence = compute_estimation(worker, neighbor_weight, neighbor_worker)
            estimation_list_every_node.append(estimation.item())
            record[f'node {idx}'] = dotlist
        else:
            continue
    return estimation_list_every_node, record, self_influence, firstnei_influence

def eval_secondnei(worker_list, P, args, probe_valid_loader, second_neighbor):
    for worker in worker_list:
        worker.eval(probe_valid_loader, 5)
    secondnei_influencedict = dict()
    for firstnei, secondnei_set in second_neighbor.items():
        firstnei_Weight = P[firstnei][args.choose_node]
        neidict = dict()
        for secondnei, weight in secondnei_set.items():
            secondnei_influence = compute_grad_and_hvp(worker_list[firstnei].firstnei_params, 
                                                       worker_list[firstnei].firstnei_grads, 
                                                       worker_list[secondnei].secondnei_grads_after_merge, 
                                                       worker_list[args.choose_node].grads_train,
                                                       worker_list[firstnei].current_lr)
            secondnei_influence = (-1) * secondnei_influence * weight * firstnei_Weight
           
            neidict[f'node{secondnei}'] = secondnei_influence.item()
        secondnei_influencedict[f'node{firstnei}'] = neidict
    return secondnei_influencedict

def eval_thirdnei(worker_list, P, args, probe_valid_loader, second_neighbor):
    for worker in worker_list:
        worker.eval(probe_valid_loader, 6)
    thirdnei_influencedict = dict()
    for firstnei, secondnei_set in second_neighbor.items():
        firstnei_Weight = P[firstnei][args.choose_node]
        secondneidict = dict()
        for secondnei, thirdnei_set in secondnei_set.items():
            secondnei_Weight = P[secondnei][firstnei]
            neidict = dict()
            for thirdnei, weight in thirdnei_set.items():
                thirdnei_influence = thirdnei_compute_grad_and_hvp(model1_params=worker_list[secondnei].secondnei_params, 
                                                                   model2_params=worker_list[firstnei].firstnei_params, 
                                                                   grad1=worker_list[secondnei].secondnei_grads, 
                                                                   grad2=worker_list[firstnei].firstnei_grads, 
                                                                   grad3=worker_list[thirdnei].thirdnei_grads_after_merge, 
                                                                   grad4=worker_list[args.choose_node].grads_train,
                                                                   current_lr1=worker_list[firstnei].current_lr,
                                                                   current_lr2=worker_list[secondnei].current_lr)
                thirdnei_influence = (-1) * thirdnei_influence * weight * firstnei_Weight * secondnei_Weight
               
                neidict[f'node{thirdnei}'] = thirdnei_influence.item()
            secondneidict[f'node{secondnei}'] = neidict
        thirdnei_influencedict[f'node{firstnei}'] = secondneidict 
    return thirdnei_influencedict

def compute_grad_and_hvp(model1_params, grad1, grad2, grad3, current_lr):

    grad1_flat = torch.cat([g.reshape(-1) for g in grad1])
   

    grad2_flat = torch.cat([g.reshape(-1) for g in grad2])
    
    with torch.autograd.set_detect_anomaly(True):
    
        hvp = torch.autograd.grad(grad1_flat, model1_params, grad_outputs=grad2_flat, retain_graph=True)
    
   
    hvp_flat = torch.cat([h.reshape(-1) for h in hvp])
    
   
    ihvp_flat = grad2_flat - hvp_flat * current_lr
    
  
    grad3_flat = torch.cat([g.reshape(-1) for g in grad3])
    
   
    result = torch.dot(grad3_flat, ihvp_flat)
    
    return result

def thirdnei_compute_grad_and_hvp(model1_params, model2_params, grad1, grad2, grad3, grad4, current_lr1, current_lr2):
  
    grad1_flat = torch.cat([g.reshape(-1) for g in grad1])
    grad2_flat = torch.cat([g.reshape(-1) for g in grad2])
    grad3_flat = torch.cat([g.reshape(-1) for g in grad3])
    grad4_flat = torch.cat([g.reshape(-1) for g in grad4])

    #  (I - Hessian2) × grad4
    with torch.autograd.set_detect_anomaly(True):
        #  Hessian2 × grad4
        hvp2 = torch.autograd.grad(grad2_flat, model2_params, grad_outputs=grad4_flat, retain_graph=True)
        hvp2_flat = torch.cat([h.reshape(-1) for h in hvp2])
        #  (I - Hessian2) × grad4 = grad4 - (Hessian2 × grad4)
        ihvp2_flat = grad4_flat - hvp2_flat * current_lr1

    #  (I - Hessian1) × (I - Hessian2) × grad4
    with torch.autograd.set_detect_anomaly(True):
        #  Hessian1 × (I - Hessian2) × grad4
        hvp1 = torch.autograd.grad(grad1_flat, model1_params, grad_outputs=ihvp2_flat, retain_graph=True)
        hvp1_flat = torch.cat([h.reshape(-1) for h in hvp1])
        #  (I - Hessian1) × (I - Hessian2) × grad4 = (I - Hessian2) × grad4 - (Hessian1 × (I - Hessian2) × grad4)
        ihvp1_flat = ihvp2_flat - hvp1_flat * current_lr2

    #  grad3^T × (I - Hessian1) × (I - Hessian2) × grad4
    result = torch.dot(grad3_flat, ihvp1_flat)

    return result

def get_specific_batch(train_dataloaders, args):
    specific_loader = train_dataloaders[args.choose_node]
    for idx, batch in enumerate(specific_loader):
        if idx == args.choose_batch:
            return batch
        
import os
import copy
import json
import torch
import socket
import datetime
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from our_datasets import load_dataset
from networks import load_model, load_valuemodel
from workers.worker_vision import Worker_Vision
from utils.scheduler import Warmup_MultiStepLR
from utils.utils import (
    
    add_identity,
    generate_P,
    save_model,
    evaluate_and_log,
    update_center_model,
    update_dsgd,
    update_dqn_chooseone,
    update_csgd,
    update_heuristic,
    Merge_History,
    update_heuristic_2,
    update_heuristic_3,
    update_dqn_chooseone_debug,
    update_dqn_chooseone_debug_2,
    merge_model,
    evaluate_last,
    merge_without_update,
    compute_loss_every_node,
    read_json_file,
    calculate_estimation_and_loss
)
from utils.random import set_seed
from easydict import EasyDict
import wandb
from utils.dirichlet import (
    dirichlet_split_noniid,
    create_dataloaders,
    create_simple_preference,
    create_IID_preference,
    dirichlet_split
)
from torchvision.datasets import CIFAR10
import numpy as np
from fire import Fire
from tqdm import trange
import sys
# torch.set_num_threads(4)


def main(
    dataset_path="datasets",
    dataset_name="cifar10",  # cifar10_test
    image_size=56,
    batch_size=512,
    n_swap=None,
    mode="dqn_chooseone",
    shuffle="fixed",
    size=10,
    port=29500,
    backend="gloo",
    model="ResNet18_M",
    pretrained=1,
    lr=0.1,
    wd=0.0,
    gamma=0.1,
    momentum=0.0,
    warmup_step=0,
    epochs=5,
    milestones=[2400, 4800],
    seed=666,
    device="cuda:0",
    amp=False,
    sample=0,
    n_components=0,
    nonIID=False,
    dirichlet=False,
    project_name="decentralized",
    alpha=0.8,
    state_size=144,
    valuemodel_hiddensize=320,
    merge_step=1,
    choose_which=0,
    choose_batch=10,
    node_datasize=6000,
    choose_node=0,
    train_to_end=False,
    adam=False,
):

    args = EasyDict(locals().copy())
  
    set_seed(seed)
    dir_path = os.path.dirname(__file__)
    args = add_identity(args, dir_path)


    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    dir_path = os.path.dirname(__file__)
    args = add_identity(args, dir_path)

    # check nfs dataset path

    log_id = (
        datetime.datetime.now().strftime("%b%d_%H:%M:%S")
        + "_"
        + socket.gethostname()
        + "_"
        + args.identity
    )
    writer = SummaryWriter(log_dir=os.path.join(args.runs_data_dir, log_id))

    
    print('image size',args.image_size)
    probe_train_loader, probe_valid_loader, _, classes = load_dataset(
        root=args.dataset_path,
        name=args.dataset_name,
        image_size=args.image_size,
        train_batch_size=args.batch_size,
        valid_batch_size=args.batch_size,
        return_dataloader=True,
        debug=True
    )

    train_set, valid_set, _, nb_class = load_dataset(
        dataset_path, dataset_name, image_size, return_dataloader=False
    )
    if nonIID:
        if dirichlet:
            all_class_weights = dirichlet_split(args.size, nb_class, alpha)
        else:
            all_class_weights = create_simple_preference(
                args.size, nb_class, important_prob=0.8
            )
    else:
        all_class_weights = create_IID_preference(args.size, nb_class)
    train_dataloaders = create_dataloaders(
        train_set, args.size, args.node_datasize, args.batch_size, all_class_weights, nb_class)

    valid_dataloaders = create_dataloaders(valid_set, args.size, args.node_datasize
                                           , args.batch_size, all_class_weights, nb_class )
    worker_list = []
    trainloader_length_list = []
    
    for rank in range(args.size):
        train_loader = train_dataloaders[rank]
        trainloader_length_list.append(len(train_loader))
        model = load_model(args.model, nb_class, pretrained=args.pretrained, args=args, nb_class= nb_class).to(
            args.device
        )
        if adam:
            optimizer = Adam(
                model.parameters(), 
                lr=args.lr, 
                weight_decay=args.wd
            )
        else:
            optimizer = SGD(
            model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum
        )
       
        scheduler = Warmup_MultiStepLR(
            optimizer,
            warmup_step=args.warmup_step,
            milestones=args.milestones,
            gamma=args.gamma,
        )
    
        worker = Worker_Vision(
                model, rank, optimizer, scheduler, train_loader, args.device, args.choose_node, args.choose_batch, args.size,
                args.train_to_end
            )
        worker_list.append(worker)
        # model_list.append(model)

    P = generate_P(args.mode, args.size)

    steps = epochs * trainloader_length_list[0]
    sum_estimation = 0
    estimation_list = []
    dot_list = []
    valid_acc_list = []
    train_acc_list = []
    for iteration in trange(steps):
      
        current_epoch = iteration // (
            sum(trainloader_length_list) / len(trainloader_length_list)
        )

        for i in range(0, args.size):
            if iteration % trainloader_length_list[i] == 0:
                worker_list[i].update_iter()
                worker_list[i].current_batch_index = -1

        
       # dsgd
        update_dsgd(worker_list, P, args, probe_valid_loader)
        merge_without_update(worker_list, P, args, probe_valid_loader)
        merge_model(worker_list, P)
        
        if worker_list[0].current_batch_index == worker_list[0].choose_batch and worker_list[0].train_to_end == False:
     
            train_acc, train_loss, valid_acc, acc_train, valid_loss, estimation, dot1, dot2 = evaluate_and_log(  
                probe_train_loader,
                probe_valid_loader,
                iteration,
                current_epoch,
                writer,
                args,
                wandb,
                mode,
                worker_list,
                train_dataloaders,
                valid_dataloaders,
                P
                
            )
            dot_list.append([dot1, dot2, dot1+dot2])
            estimation = estimation.item()
            sum_estimation += estimation
            estimation_list.append(estimation)
            print('estimation: ', estimation)
            valid_acc_list.append(valid_acc)    
            train_acc_list.append(train_acc)
    
    # loss list include the end loss of neighbor node and self node
    # loss every epoch include every epoch loss of neighbor node and self node
    loss_list, loss_every_epoch, valid_acc = evaluate_last(worker_list, P, args, probe_valid_loader)
    if args.train_to_end == False:
        loss_estimation, loss_estimation_every_epoch_list = compute_loss_every_node(loss_every_epoch)
    else:
        loss_estimation = 0
        loss_estimation_every_epoch_list = []
    writer.close()
    print('sum_estimation: ', sum_estimation)
    print('loss list: ', loss_list)
    print("Ending")
    
    return loss_list, estimation_list, sum_estimation, dot_list, loss_every_epoch, loss_estimation, loss_estimation_every_epoch_list, valid_acc, valid_acc_list, train_acc_list


if __name__ == "__main__":

    # 定义参数字典
    params = {
        "dataset_path": "./data",
        "dataset_name": sys.argv[6],
        "image_size": int(sys.argv[12]),
        "batch_size": int(sys.argv[4]),
        "n_swap": None,
        "mode": sys.argv[7],
        "shuffle": "fixed",
        "size": int(sys.argv[5]),
        "port": 29500,
        "backend": "gloo",
        "model": sys.argv[10],
        "pretrained": int(sys.argv[13]),
        "lr": 0.1,
        "wd": 0.0,
        "gamma": 0.1,
        "momentum": 0.0,
        "warmup_step": 0,
        "epochs": int(sys.argv[8]),
        "milestones": [2400, 4800],
        "seed": 222,
        "device": f"cuda:{int(sys.argv[9])}",
        "amp": False,
        "nonIID": False if sys.argv[11] == 'false' else True,
        "project_name": "decentralized",
        "dirichlet": True,
        "node_datasize": int(sys.argv[3]),
        "choose_node": int(sys.argv[1]),
        "choose_batch":int(sys.argv[2]),
        "train_to_end": True,
        "adam": False if sys.argv[14] == 'false' else True,
    }

    params['train_to_end'] = False
    loss_list_not_to_end, estimation_list, sum_estimation, dot_list, loss_every_epoch, loss_estimation, loss_estimation_every_node_list, valid_acc, valid_acc_list, train_acc_list = main(**params)
   
    data = {
       
        "loss_list_not_to_end": loss_list_not_to_end,
        "estimation_list": estimation_list,
        "dot_list": dot_list,
  
        "loss_every_epoch": loss_every_epoch,
        "loss_estimation": loss_estimation,
        "loss_estimation_list": loss_estimation_every_node_list,
        "sum_estimation": sum_estimation,
        'valid_acc': valid_acc,
        'valid_acc_list': valid_acc_list,
        
    }
    
    save_dir = f"./loss_record/{sys.argv[6]}_epochs{sys.argv[8]}_data{sys.argv[3]}_batchsize{sys.argv[4]}_mode{sys.argv[7]}_size{sys.argv[5]}_{sys.argv[10]}_noniid{sys.argv[11]}_pretrained{sys.argv[13]}_adam{sys.argv[14]}/"
    os.makedirs(save_dir, exist_ok=True)

    save_json_path = os.path.join(save_dir, f"node:{sys.argv[1]}_batch:{sys.argv[2]}.json")
    with open(save_json_path, 'w') as f:
        json.dump(data, f, indent=4)
    read_json_file(save_json_path)
    calculate_estimation_and_loss(save_json_path)
    print("finish")
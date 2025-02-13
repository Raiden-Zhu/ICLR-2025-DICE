import os
import copy
import json
import torch
import socket
import datetime
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from our_datasets import load_dataset
from networks import load_model, load_valuemodel
from workers.worker_vision import Worker_Vision, Worker_Vision_AMP, DQNAgent
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
    compute_loss_every_epoch,
    search_model,
    eval_across_workers,
    get_specific_batch,
    get_secondneighbor,
    eval_secondnei,
    get_thirdneighbor,
    eval_thirdnei
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
    choose_epoch=0
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
    specific_batch = get_specific_batch(train_dataloaders, args)
    for rank in range(args.size):

        # train_loader, _, _, classes = load_dataset(
        #     root=args.dataset_path,
        #     name=args.dataset_name,
        #     image_size=args.image_size,
        #     train_batch_size=args.batch_size,
        #     distribute=True,
        #     rank=rank,
        #     split=split,
        #     seed=args.seed,
        # )
        train_loader = train_dataloaders[rank]
        trainloader_length_list.append(len(train_loader))
        model = load_model(args.model, nb_class, pretrained=args.pretrained, args=args, nb_class= nb_class).to(
            args.device
        )

        optimizer = SGD(
            model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum
        )
        # scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        scheduler = Warmup_MultiStepLR(
            optimizer,
            warmup_step=args.warmup_step,
            milestones=args.milestones,
            gamma=args.gamma,
        )
        # worker是训练器
        if args.amp:
            worker = Worker_Vision_AMP(
                model, rank, optimizer, scheduler, train_loader, args.device
            )
        else:
            
            worker = Worker_Vision(
                model, rank, optimizer, scheduler, train_loader, args.device, args.choose_node, args.choose_batch, args.size,
                args.train_to_end, args.choose_epoch, specific_batch
            )
        worker_list.append(worker)
        # model_list.append(model)

    P = generate_P(args.mode, args.size)

    steps = epochs * trainloader_length_list[0]
    sum_estimation = 0
    estimation_list = []
    dot_list = []
    for iteration in trange(steps):
        # 这里的epoch是平均epoch
        current_epoch = iteration // (
            sum(trainloader_length_list) / len(trainloader_length_list)
        )

        # if iteration % len(train_loader) == 0:
        # for worker in worker_list:
        # worker.update_iter()

        # 对每个trainloader检测是否遍历完，如果遍历完 则对应的worker 更新trainloader
        for i in range(0, args.size):
            if iteration % trainloader_length_list[i] == 0:
                worker_list[i].update_iter()
                worker_list[i].current_batch_index = -1

        
       # dsgd
        update_dsgd(worker_list, P, args, probe_valid_loader)
        # merge_without_update(worker_list, P, args, probe_valid_loader)
        
        merge_model(worker_list, P)
        
        if args.choose_batch == worker_list[0].current_batch_index and args.choose_epoch == worker_list[0].now_epoch:
            estimation, record, self_influence, firstnei_influence =  eval_across_workers(worker_list, P, args, probe_valid_loader)
            estimation_list.append(estimation)
            train_acc, train_loss, valid_acc, valid_loss, _, dot1, dot2, trainacc_list, validacc_list, self_influence, firstnei_influence = evaluate_and_log(  
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
            second_neighbor = get_secondneighbor(choose_node, P)
        if args.choose_batch == worker_list[0].current_batch_index and args.choose_epoch + 1 == worker_list[0].now_epoch:
            secondnei_influencedict =  eval_secondnei(worker_list, P, args, probe_valid_loader, second_neighbor)
            third_neighbor = get_thirdneighbor(choose_node, P)
            
        if args.choose_batch == worker_list[0].current_batch_index and args.choose_epoch + 2 == worker_list[0].now_epoch:
            thirdnei_influencedict =  eval_thirdnei(worker_list, P, args, probe_valid_loader, third_neighbor)
            break
    # estimation list include several list , each list has the estimation of each worker at specific epoch
    return estimation_list, trainacc_list, validacc_list, record, self_influence, firstnei_influence, secondnei_influencedict, thirdnei_influencedict


if __name__ == "__main__":

    # 定义参数字典
    params = {
        "dataset_path": "/mnt/csp/mmvision/home/lwh/FLASH-RL/data",
        "dataset_name": sys.argv[6],
        "image_size": 28,
        "batch_size": int(sys.argv[4]),
        "n_swap": None,
        "mode": sys.argv[7],
        "shuffle": "fixed",
        "size": int(sys.argv[5]),
        "port": 29500,
        "backend": "gloo",
        "model": sys.argv[10],
        "pretrained": 0,
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
        "train_to_end": False,
        "choose_epoch": int(sys.argv[12])
    }

    estimation_list, trainacc_list, validacc_list, record, self_influence, firstnei_influence, secondnei_influencedict, thirdnei_influencedict = main(**params)
    result_dict = {}
    # for small_list in estimation_list:
    # # 遍历小列表中的每个值
    #     for index, value in enumerate(small_list):
    #         # 如果索引不在字典中，初始化为0
    #         if index not in result_dict:
    #             result_dict[index] = 0
    #         # 累加值到字典中
    #         result_dict[index] += value
    # result_dict['estimation_list'] = estimation_list
    result_dict['self_influence'] = self_influence
    result_dict['secondnei_influencedict'] = secondnei_influencedict
    result_dict['firstnei_influence'] = firstnei_influence
    result_dict['thirdnei_influencedict'] = thirdnei_influencedict
    # result_dict['train_acc'] = trainacc_list
    # result_dict['valid_acc'] = validacc_list
    # result_dict['record'] = record

            
        
    os.makedirs(f"/mnt/csp/mmvision/home/lwh/DLS_2/loss_record/{sys.argv[6]}_epochs{sys.argv[8]}_data{sys.argv[3]}_batchsize{sys.argv[4]}_mode{sys.argv[7]}_size{sys.argv[5]}_{sys.argv[10]}_noniid{sys.argv[11]}_chooseepoch{sys.argv[12]}/", exist_ok=True)
    with open(f"/mnt/csp/mmvision/home/lwh/DLS_2/loss_record/{sys.argv[6]}_epochs{sys.argv[8]}_data{sys.argv[3]}_batchsize{sys.argv[4]}_mode{sys.argv[7]}_size{sys.argv[5]}_{sys.argv[10]}_noniid{sys.argv[11]}_chooseepoch{sys.argv[12]}/node:{sys.argv[1]}_batch:{sys.argv[2]}.json", 'w') as f:
        
        json.dump(result_dict, f, indent=4)
    
    '''
    params['train_to_end'] = False
    loss_list_not_to_end, estimation_list, sum_estimation, dot_list, loss_every_epoch, loss_estimation, loss_estimation_list = main(**params)
    groundtruth = sum((a - b) for a, b in zip(loss_list_not_to_end, loss_list_to_end))
    data = {
        "loss_list_to_end": loss_list_to_end,
        "loss_list_not_to_end": loss_list_not_to_end,
        "estimation_list": estimation_list,
        "dot_list": dot_list,
        "sum_estimation": sum_estimation,
        "groundtruth": groundtruth,
        "loss_every_epoch": loss_every_epoch,
        "loss_estimation": loss_estimation,
        "loss_estimation_list": loss_estimation_list
    }
    
    os.makedirs(f"/mnt/csp/mmvision/home/lwh/DLS_1/loss_record/{sys.argv[6]}_epochs{sys.argv[8]}_data{sys.argv[3]}_batchsize{sys.argv[4]}_mode{sys.argv[7]}_size{sys.argv[5]}_{sys.argv[10]}_noniid{sys.argv[11]}/", exist_ok=True)
    with open(f"/mnt/csp/mmvision/home/lwh/DLS_1/loss_record/{sys.argv[6]}_epochs{sys.argv[8]}_data{sys.argv[3]}_batchsize{sys.argv[4]}_mode{sys.argv[7]}_size{sys.argv[5]}_{sys.argv[10]}_noniid{sys.argv[11]}/node:{sys.argv[1]}_batch:{sys.argv[2]}.json", 'w') as f:
        
        json.dump(data, f, indent=4)
        '''
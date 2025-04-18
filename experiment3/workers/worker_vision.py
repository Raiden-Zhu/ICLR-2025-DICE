import copy
import torch
import torch.nn as nn
from replay_buffer import ReplayBuffer
import torch.nn.functional as F
from .feature import pca_weights
import torch.optim as optim
import numpy as np

criterion = nn.CrossEntropyLoss()


class Worker_Vision:
    def __init__(self, model, rank, optimizer, scheduler, train_loader, device, choose_node, choose_batch, size,
                 train_to_end, choose_epoch, specific_batch):
        self.model = model
        self.rank = rank
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.size = size
        self.train_loader = train_loader
        # self.train_loader_iter = train_loader.__iter__()
        self.device = device
        self.choose_node = choose_node
        self.choose_batch = choose_batch
        self.current_batch_index = -1
        self.grads_after_choosebatch = []
        self.grads_train = []
        self.grads_after_merge = []
        self.grads_before_choosebatch = []
        self.total_loss = 0
        self.train_to_end = train_to_end
        self.loss_mode0 = list()
        self.loss_mode1 = list()
        self.loss_mode2 = list()
        self.loss_mode3 = list()
        self.choose_epoch = choose_epoch
        self.now_epoch = -1
        self.specific_batch = specific_batch
        
    def update_iter(self):
        self.train_loader_iter = self.train_loader.__iter__()

    def step(self, probe_valid_loader):
        self.model.train()
        try:
            if self.current_batch_index == -1:
                self.now_epoch += 1
            batch = self.train_loader_iter.__next__()
            self.current_batch_index += 1
        except StopIteration:
            print("迭代结束")
        if self.current_batch_index == self.choose_batch and  self.train_to_end == False:
            self.eval(probe_valid_loader, 1)
            self.data, self.target = self.specific_batch[0].to(self.device), self.specific_batch[1].to(self.device)
            output = self.model(self.data)
            
            loss = criterion(output, self.target)
            self.optimizer.zero_grad()
            loss.backward()
            params1 = list(self.model.parameters())
            self.grads_train = [p.grad for p in params1]
            self.current_lr = self.scheduler.get_last_lr()[0]
        elif self.current_batch_index == self.choose_batch + 1 and self.train_to_end == False:
           
            self.data, self.target = self.specific_batch[0].to(self.device), self.specific_batch[1].to(self.device)
            output = self.model(self.data)
            loss = criterion(output, self.target)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            temp_model = copy.deepcopy(self.model)
            temp_model.load_state_dict(self.model.state_dict())
            temp_output = temp_model(self.data)
            temp_loss = criterion(temp_output, self.target)
            temp_loss.backward(retain_graph=True)
            self.firstnei_params = list(temp_model.parameters())
            self.firstnei_grads = torch.autograd.grad(temp_loss, temp_model.parameters(), create_graph=True)
            self.current_lr = self.scheduler.get_last_lr()[0]

        elif self.current_batch_index == self.choose_batch + 2 and self.train_to_end == False:
           
            self.data, self.target = self.specific_batch[0].to(self.device), self.specific_batch[1].to(self.device)
            output = self.model(self.data)
            loss = criterion(output, self.target)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)

            temp_model = copy.deepcopy(self.model)
            temp_model.load_state_dict(self.model.state_dict())
            temp_output = temp_model(self.data)
            temp_loss = criterion(temp_output, self.target)
            temp_loss.backward(retain_graph=True)
            self.secondnei_params = list(temp_model.parameters())
            self.secondnei_grads = torch.autograd.grad(temp_loss, temp_model.parameters(), create_graph=True)
            self.current_lr = self.scheduler.get_last_lr()[0]
        else:
            self.data, self.target = batch[0].to(self.device), batch[1].to(self.device)

            output = self.model(self.data)
            loss = criterion(output, self.target)
            self.optimizer.zero_grad()
            loss.backward()
    
    def eval(self, valid_loader, loss_mode):
        # loss mode is 0 ，compute z' theta t + 1/2 
        # loss mode is 1 , compute z' theta t ，计算梯度
        # loss mode is 2 ， compute z ' theta k/j t+1 
     
        # loss mode is 4， compute loss at end
        
        # loss mode is 3， compute z ' theta k t + 1
        total_loss, total_correct, total, step = 0, 0, 0, 0
        total_loss_sum = torch.tensor(0.0, device=self.device)
        for batch in valid_loader:
            step += 1
            data, target = batch[0].to(self.device), batch[1].to(self.device)
            output = self.model(data)
            p = torch.softmax(output, dim=1).argmax(1)
            total_correct += p.eq(target).sum().item()
            total += len(target)
            loss = criterion(output, target)
            total_loss += loss.item()
            total_loss_sum += loss

        total_valid_loss_sum = total_loss_sum / step
        if loss_mode == 0:
            self.loss_mode0.append(total_loss / step)
        elif loss_mode == 2:
            self.loss_mode2.append(total_loss / step)
        # mode为3 will not be used
        elif loss_mode == 3:
            self.loss_mode1.append(total_loss / step)
            self.optimizer.zero_grad()
            total_valid_loss_sum.backward()
            params1 = list(self.model.parameters())
            self.grads_after_merge = [p.grad for p in params1]
            self.optimizer.zero_grad()
        elif loss_mode == 4:
            self.total_loss = (total_loss / step)
        elif loss_mode == 1:
            self.loss_mode1.append(total_loss / step)
            self.optimizer.zero_grad()
            total_valid_loss_sum.backward()
            params1 = list(self.model.parameters())
            self.grads_before_choosebatch = [p.grad for p in params1]
            self.optimizer.zero_grad()
        elif loss_mode == 5:
        
            self.optimizer.zero_grad()
            total_valid_loss_sum.backward()
            params1 = list(self.model.parameters())
            self.secondnei_grads_after_merge = [p.grad for p in params1]
            self.optimizer.zero_grad()
        elif loss_mode == 6:
        
            self.optimizer.zero_grad()
            total_valid_loss_sum.backward()
            params1 = list(self.model.parameters())
            self.thirdnei_grads_after_merge = [p.grad for p in params1]
            self.optimizer.zero_grad()

    def refresh_bn(self):
        self.model.train()

        batch = self.train_loader_iter.__next__()
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        self.model(data)

    def step_csgd(self):
        self.model.train()

        batch = self.train_loader_iter.next()
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        output = self.model(data)
        loss = criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()

        grad_dict = {}
        for name, param in self.model.named_parameters():
            grad_dict[name] = param.grad.data

        return grad_dict

    def get_accuracy(self, model):
        model.eval()
        output = model(self.data)
        _, predicted = torch.max(output.data, 1)
        total_samples = self.target.size(0)
        total_correct = (predicted == self.target).sum().item()
        accuracy = total_correct / total_samples
        return accuracy

    def update_grad(self):
        self.optimizer.step()
        self.scheduler.step()

    def scheduler_step(self):
        self.scheduler.step()

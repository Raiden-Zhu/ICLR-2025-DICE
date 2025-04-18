import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

criterion = nn.CrossEntropyLoss()


class Worker_Vision:
    def __init__(self, model, rank, optimizer, scheduler, train_loader, device, choose_node, choose_batch, size,
                 train_to_end, choose_epoch, noise_feature=False):
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
        self.noise_feature = noise_feature
        

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
            print("iteration stop")
        if self.now_epoch == self.choose_epoch and self.current_batch_index == self.choose_batch and self.rank == self.choose_node and self.train_to_end == False:
            # self.eval(probe_valid_loader, 1)
           
            if self.noise_feature == False:
                self.statedict_before_batch = copy.deepcopy(self.model.state_dict())
                self.data, self.target = batch[0].to(self.device), batch[1].to(self.device)
                perm = torch.randperm(self.target.size(0))
                # shuffle target lables
                self.target = self.target[perm]
                output = self.model(self.data)
                if not isinstance(output, torch.Tensor):
                    output = output.logits
                loss = criterion(output, self.target)
                self.optimizer.zero_grad()
                loss.backward()
                params1 = list(self.model.parameters())
                self.grads_train = [p.grad for p in params1 if p.grad is not None]
            else:
                self.statedict_before_batch = copy.deepcopy(self.model.state_dict())
                self.data, self.target = batch[0].to(self.device), batch[1].to(self.device)
                
                noise_std = 100 
                noise = torch.randn_like(self.data) * noise_std
                self.data = self.data + noise
              
                output = self.model(self.data)
                if not isinstance(output, torch.Tensor):
                    output = output.logits
                loss = criterion(output, self.target)
                self.optimizer.zero_grad()
                loss.backward()
                params1 = list(self.model.parameters())
                self.grads_train = [p.grad for p in params1 if p.grad is not None]
        elif self.now_epoch == self.choose_epoch and self.current_batch_index == self.choose_batch and self.rank != self.choose_node and self.train_to_end == False:
            self.data, self.target = batch[0].to(self.device), batch[1].to(self.device)
            output = self.model(self.data)
            if not isinstance(output, torch.Tensor):
                output = output.logits
            loss = criterion(output, self.target)
            self.optimizer.zero_grad()
            loss.backward()
            params1 = list(self.model.parameters())
            self.grads_train = [p.grad for p in params1 if p.grad is not None]
        else:
            self.data, self.target = batch[0].to(self.device), batch[1].to(self.device)
            # print('shape', self.data.shape, self.target.shape)
            output = self.model(self.data)
            if not isinstance(output, torch.Tensor):
                output = output.logits
            loss = criterion(output, self.target)
            self.optimizer.zero_grad()
            loss.backward()
    
    def eval(self, valid_loader, loss_mode):
        # loss mode  0 ， z' theta t + 1/2 
        # loss mode  1 ,  z' theta t ，
        # loss mode 2 ， z ' theta k/j t+1 
     
        # loss mode 4，  loss at end
        total_loss, total_correct, total, step = 0, 0, 0, 0
        total_loss_sum = torch.tensor(0.0, device=self.device)
        for batch in valid_loader:
            step += 1
            if step == 20:
                break
            data, target = batch[0].to(self.device), batch[1].to(self.device)
            output = self.model(data)
            if not isinstance(output, torch.Tensor):
                output = output.logits
            p = torch.softmax(output, dim=1).argmax(1)
            total_correct += p.eq(target).sum().item()
            total += len(target)
            loss = criterion(output, target)
            total_loss += loss.item()
            total_loss_sum += loss

        # print('loss mode', loss_mode)
        # print(total_loss)
        total_valid_loss_sum = total_loss_sum / step
        if loss_mode == 0:
            self.loss_mode0.append(total_loss / step)
        elif loss_mode == 2:
            self.loss_mode2.append(total_loss / step)
            # mode3 not used
        elif loss_mode == 3:
            self.loss_mode3.append(total_loss / step)
        elif loss_mode == 4:
            self.total_loss = (total_loss / step)
        elif loss_mode == 1:
            self.loss_mode1.append(total_loss / step)
            self.optimizer.zero_grad()
            total_valid_loss_sum.backward()
            params1 = list(self.model.parameters())
            self.grads_after_choosebatch = [p.grad for p in params1 if p.grad is not None]
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
        if not isinstance(output, torch.Tensor):
            output = output.logits
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
        if not isinstance(output, torch.Tensor):
            output = output.logits
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



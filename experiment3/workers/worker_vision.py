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

    # 需要计算出choose node 对 choose batch 的 loss的梯度
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
            #  先计算自身节点和valid data的梯度
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
            # params1 = list(self.model.parameters())
            # self.firstnei_params = copy.deepcopy(list(self.model.parameters()))
            temp_model = copy.deepcopy(self.model)
            temp_model.load_state_dict(self.model.state_dict())
            temp_output = temp_model(self.data)
            temp_loss = criterion(temp_output, self.target)
            temp_loss.backward(retain_graph=True)
            self.firstnei_params = list(temp_model.parameters())
            self.firstnei_grads = torch.autograd.grad(temp_loss, temp_model.parameters(), create_graph=True)
            self.current_lr = self.scheduler.get_last_lr()[0]
            # grads = torch.autograd.grad(loss, params1, create_graph=True)

            # # 初始化 Hessian 矩阵
            # hessian = []
           
            # # 计算 Hessian 矩阵
            # for grad in grads:
            #     print(1)
            #     grad_flat = grad.view(-1)  # 将梯度展平为一维向量
            #     hessian_row = []
            #     for g in grad_flat:
            #         # 对每个梯度元素计算二阶导数
            #         hessian_element = torch.autograd.grad(g, params1, retain_graph=True)
            #         hessian_row.append(hessian_element)
            #     hessian.append(hessian_row)

            # # 将 Hessian 矩阵转换为张量
            # hessian = torch.tensor(hessian)
            # self.firstnei_hessian = torch.eye(hessian.shape[1]) - self.scheduler.get_last_lr()[0] * hessian
            
            # del grads, grad_flat, hessian_row, hessian_element
            # torch.cuda.empty_cache() 
        elif self.current_batch_index == self.choose_batch + 2 and self.train_to_end == False:
           
            self.data, self.target = self.specific_batch[0].to(self.device), self.specific_batch[1].to(self.device)
            output = self.model(self.data)
            loss = criterion(output, self.target)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # params1 = list(self.model.parameters())
            # self.firstnei_params = copy.deepcopy(list(self.model.parameters()))
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
            # print('shape', self.data.shape, self.target.shape)
            output = self.model(self.data)
            loss = criterion(output, self.target)
            self.optimizer.zero_grad()
            loss.backward()
    
    def eval(self, valid_loader, loss_mode):
        # loss mode 为 0 ，计算 z' theta t + 1/2 
        # loss mode 为 1 , 计算 z' theta t ，计算梯度
        # loss mode 为2 ， 计算z ' theta k/j t+1 
     
        # loss mode 为4， 计算 loss at end
        
        # loss mode 为 3， 计算 z ' theta k t + 1
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

        # print('loss mode', loss_mode)
        # print(total_loss)
        total_valid_loss_sum = total_loss_sum / step
        if loss_mode == 0:
            self.loss_mode0.append(total_loss / step)
        elif loss_mode == 2:
            self.loss_mode2.append(total_loss / step)
            # mode为3不会被激活
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
        # loss = criterion(output, target)
        # self.optimizer.zero_grad()
        # loss.backward()

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


# 传入一个定义好的network
class DQNAgent(Worker_Vision):
    """DQN Agent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
    """

    def __init__(
        self,
        model,
        value_model,
        rank,
        optimizer,
        scheduler,
        train_loader,
        args,
        wandb,
        max_epsilon: float = 0.2,
        min_epsilon: float = 0.05,
        gamma: float = 0.99,
        memory_size: int = 10000,
        batch_size: int = 10,
        target_update: int = 10,
        epsilon_decay: float = 1 / 2000,
        seed: int = 6666,
    ):
        super().__init__(model, rank, optimizer, scheduler, train_loader, args.device)

        self.state_size = args.state_size
        self.memory = ReplayBuffer(self.state_size, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        self.clients_number = args.size  # clients总数
        # pac 系数
        self.n_components = args.n_components
        # device: cpu / gpu
        self.device = args.device
        # print(self.device)
        self.sample = args.sample
        self.wandb = wandb
        # networks: dqn, dqn_target
        # self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn = value_model.to(self.device)
        # self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = value_model.to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # dqn网络的optimizer
        self.dqn_optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

        self.update_cnt = 0
        
        self.last_action = torch.zeros(1)

    def feature(self, n_components, model):
        weights_pca = pca_weights(n_components=n_components, weights=model.state_dict())
        # weights_pca = torch.from_numpy(weights_pca)
        return weights_pca

    def select_action_sample(self):
        # 增加采样数量
        if self.sample < 0:
            print("invalid sample")
        else:
            # 假设 self.dqn 是一个 PyTorch 模型，self.state 是输入状态
            # 获取网络输出
            output = self.dqn(self.state.to(self.device))  # output shape [size]
            num_samples = self.sample
            # 使用 multinomial 函数根据输出概率随机选择下标
            # 注意：multinomial 函数返回的是下标，而不是概率值
            # replacement=False 表示不重复选择
            # dim=1 表示在输出张量的最后一个维度上进行选择
            selected_indices = torch.multinomial(
                output.softmax(dim=-1), num_samples, replacement=False
            )
            # selected_indices shape [sample]
            # 使用选择的下标来获取对应的输出值 selected_indices 和 output shape 相同 都是一维
            selected_outputs = output.gather(0, selected_indices)

            self.transition_sample = list()
            for i in range(0, num_samples):
                self.transition_sample.append([self.state, selected_indices[i].item()])
                merge_model = self.act(
                    self.model, selected_indices[i], self.worker_list_model
                )
                old_accuracy = self.get_accuracy(self.model)
                new_accuracy = self.get_accuracy(merge_model)
                reward = new_accuracy - old_accuracy
                next_state = self.feature(self.n_components, merge_model)
                action_record = torch.zeros(self.clients_number)
                action_record[selected_indices[i]] = 1
                next_state = torch.cat((next_state, action_record), dim=0)
                done = 0
                if not self.is_test:
                    self.transition_sample[i] += [reward, next_state, done]
                self.memory.store(*self.transition_sample[i])
            # 现在 selected_indices 包含了选择的下标，selected_outputs 包含了对应的输出值
            # self.store_buffer_sample(*self.transition_sample[i])

    # 这个函数暂时用不上
    """  
    def store_buffer_sample(self):
        for i in range(0, len(self.transition_sample)):
            # 这里的三个参数的计算到后面再优化
            done = 0
            next_state = self.state
            reward = new_acc - old_acc
            if not self.is_test:
                .transition_sample[i] += [reward, next_state, done]
                self.memory.store(*self.transition_sample[i])
    """

    def select_action(self) -> int:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > torch.rand(1).item():
            # selected_action = self.env.action_space.sample()
            action_space = torch.tensor([i for i in range(0, self.clients_number)])
            selected_action = action_space[
                torch.randint(low=0, high=len(action_space), size=(1,))
            ].item()
            # entropy = torch.log(self.clients_number)
        else:
            logits = self.dqn(self.state.to(self.device))
            # softmax logits
            normalized_logits = F.softmax(logits, dim=-1)
            entropy = -torch.sum(normalized_logits * torch.log(normalized_logits))
            selected_action = logits.argmax().item()
            # selected_action = selected_action.detach().cpu().numpy()
            
            self.wandb.log({"entropy": entropy})
        
        self.last_action = torch.zeros(self.clients_number)
        self.last_action[selected_action] = 1
        
        # 在这里先存好状态和动作
        if not self.is_test:
            self.transition = [self.state, selected_action]

    
        
        return selected_action

    def store_buffer(
        self,
        old_acc,
        new_acc,
        amplify="exp",
    ):
        """Take an action and return the response of the env."""
        # next_state, reward, terminated, truncated, _ = self.env.step(action)
        # 这里设定next state 和state 相同 ， done设置为False，reward 用准确率的变化来计算
        done = 0
        next_state = self.feature(self.n_components, self.model)
        next_state = torch.cat((next_state, self.last_action), dim=0)
        if amplify == "exp":
            reward = np.exp(new_acc) - np.exp(old_acc)
        elif amplify == "linear" or amplify is None:
            reward = new_acc - old_acc

        self.wandb.log({"reward": reward})
        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

    # 更新策略网络
    def update_dqn(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)

        self.dqn_optimizer.zero_grad()
        loss.backward()
        self.dqn_optimizer.step()

        return loss.item()

    # 做出行动
    def act(self, model, action, worker_list):
        models = copy.deepcopy(model)
        for name, param in models.named_parameters():
            choose_worker = worker_list[action]
            param.data += choose_worker.state_dict()[name].data
            param.data /= 2
        return models

    def get_workerlist(self, worker_list):
        self.worker_list_model = worker_list

    # 这个方法用于在每个step里面模型融合
    def step_mergemodel(self, worker_list):
        action = self.select_action()
        self.model.load_state_dict(self.act(self.model, action, worker_list).state_dict())
    
    def step_mergemodel_random(self, worker_list):
        # get a random number from 0 to len(worker_list)
        num = torch.randint(low=0, high=len(worker_list), size=(1,))
        model2 = copy.deepcopy(self.model)
        self.model = self.act(self.model, num.item(), worker_list)
        # self.model.load_state_dict(self.act(self.model, num.item(), worker_list).state_dict())
        # print('model the same:', self.compare_state_dicts(self.model.state_dict(), model2.state_dict()))

    def train_step_dqn(self):
        # action = self.select_action(self.state)
        # next_state, reward, done = self.step(action)
        # next_state = self.state

        self.state = self.feature(self.n_components, self.model)
        if len(self.last_action) == 1:
            self.last_action = torch.zeros(self.clients_number)
        self.state = torch.cat((self.state, self.last_action), dim=0)
        # 在选择action并作出action前先判断是否要sample
        if self.sample > 0:
            self.select_action_sample()

        # 这一步的作用是选择action并作出action
        self.step_mergemodel(self.worker_list_model)

        # 思考done的含义？
        # if episode ends
        # if done:
        # state, _ = self.env.reset(seed=self.seed)
        # scores.append(score)
        # score = 0
        # 在这里更新策略函数
        # if training is ready
        if len(self.memory) >= self.batch_size:
            loss = self.update_dqn()
            # losses.append(loss)
            self.update_cnt += 1

            # linearly decrease epsilon
            self.epsilon = max(
                self.min_epsilon,
                self.epsilon
                - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay,
            )
            # epsilons.append(self.epsilon)

            # if hard update is needed
            if self.update_cnt % self.target_update == 0:
                self._target_hard_update()
    
    def compare_state_dicts(self, state_dict1, state_dict2):
        # 检查两个字典的键是否相同
        if state_dict1.keys() != state_dict2.keys():
            print('State dicts have different keys!')
            return False
        
        # 逐个比较字典中的键值对
        for key in state_dict1:
            if not torch.allclose(state_dict1[key], state_dict2[key]):
                print(f'State dict keys not match! {key}')
                return False
        
        return True

    '''
    def train(self, num_frames: int):
        """Train the agent."""
        self.is_test = False
        
        state, _ = self.env.reset(seed=self.seed)
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # if episode ends
            # if done:
                # state, _ = self.env.reset(seed=self.seed)
                # scores.append(score)
                # score = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1
                
                # linearly decrease epsilon
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
                epsilons.append(self.epsilon)
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # plotting
            # if frame_idx % plotting_interval == 0:
                # self._plot(frame_idx, scores, losses, epsilons)
                
        # self.env.close()
        '''

    def test(self, video_folder: str) -> None:
        """Test the agent."""
        self.is_test = True

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        # print("score: ", score)
        self.env.close()

        # reset
        # self.env = naive_env

    def _compute_dqn_loss(self, samples) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        action = action.long()
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())


from torch.cuda.amp.grad_scaler import GradScaler

scaler = GradScaler()


class Worker_Vision_AMP:
    def __init__(self, model, rank, optimizer, scheduler, train_loader, device):
        self.model = model
        self.rank = rank
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        # self.train_loader_iter = train_loader.__iter__()
        self.device = device

    def update_iter(self):
        self.train_loader_iter = self.train_loader.__iter__()

    def step(self):
        self.model.train()

        batch = self.train_loader_iter.next()
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            output = self.model(data)
            loss = criterion(output, target)
        self.optimizer.zero_grad()
        scaler.scale(loss).backward()
        # loss.backward()

    def step_csgd(self):
        self.model.train()

        batch = self.train_loader_iter.next()
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            output = self.model(data)
            loss = criterion(output, target)
        self.optimizer.zero_grad()
        scaler.scale(loss).backward()
        # loss.backward()

        grad_dict = {}
        for name, param in self.model.named_parameters():
            grad_dict[name] = param.grad.data

        return grad_dict

    def update_grad(self):
        # self.optimizer.step()
        scaler.step(self.optimizer)
        scaler.update()
        self.scheduler.step()

    def scheduler_step(self):
        self.scheduler.step()

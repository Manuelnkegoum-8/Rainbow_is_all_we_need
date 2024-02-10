import torch
from torch import nn
import torch.nn.init as init
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import random,math
import torch.nn.functional as F
from .Noisy import NoisyLayer
from common.buffer import *

class AtariNet(nn.Module):
    def __init__(self,env):
        super(AtariNet,self).__init__()
        self.input_size = env.observation_space.shape
        self.channels = self.input_size[0]
        self.convnet = nn.Sequential(
            nn.Conv2d(self.channels,32, kernel_size=8,stride=4),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self,x):
        x = self.convnet(x)
        return x


class  BaseNetwork(nn.Module):
    def __init__(self,env,hidden):
        super(BaseNetwork,self).__init__()
        self.out_dim = env.action_space.n
        self.input_size = env.observation_space.shape
        self.channels = self.input_size[0]
        self.model = AtariNet(env)
        self.flatten_size = 3136
        self.values_fc1 = NoisyLayer(self.flatten_size,hidden)
        self.values_fc2 = NoisyLayer(hidden,1)

        self.advantages_fc1 = NoisyLayer(self.flatten_size,hidden)
        self.advantages_fc2 = NoisyLayer(hidden,env.action_space.n)
    
    def forward(self,x):
        feat = self.model(x)
        out1 = F.relu(self.values_fc1(feat))
        vals = self.values_fc2(out1)
        out2 = F.relu(self.advantages_fc1(feat))
        adv = self.advantages_fc2(out2)
        q = vals+adv - adv.mean(dim=1, keepdim=True)
        dist = F.softmax(q,dim=-1)
        return dist

    def reset_noise(self):
        self.advantages_fc1.reset_noise()
        self.advantages_fc2.reset_noise()
        self.values_fc1.reset_noise()
        self.values_fc2.reset_noise() 



class RainbowAgent2:
    def __init__(self,env,hidden,gamma=0.99,tau=5e-3,n_steps=3,weights=None,device=None,alpha=0.5,size=1000000,eps=1e-7):
        super(RainbowAgent2,self).__init__()
        self.device = device
        self.policy_net = BaseNetwork(env,hidden)

        if weights is not None:
            self.policy_net.load_state_dict(torch.load(weights,map_location='cpu'))
    
        self.target_net = BaseNetwork(env,hidden)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.policy_net = self.policy_net.to(device)
        self.target_net = self.target_net.to(device)
        self.criterion = nn.SmoothL1Loss(reduction="none")
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.gamma = gamma
        self.tau = tau
        self.n_steps = n_steps
        self.replay_buffer = PrioritizedReplayBuffer(alpha=alpha,
                                               size=size,
                                               eps=eps,
                                               n_steps=n_steps,
                                               gamma=gamma)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                 lr=6.25e-5,
                                 eps=1.5e-4)
        self.scaler =  GradScaler(enabled=True)
    
    @torch.no_grad
    def compute_tdtarget(self,new_states,rewards,dones):
        self.target_net.reset_noise()
        tmp = self.policy_net(new_states)
        max_idx = tmp.argmax(dim=1,keepdim=True)
        targets_values = self.target_net(new_states) # (bs,num_actions)
        max_targets_values = torch.gather(targets_values,dim=1,index=max_idx)
        targets = rewards + (1-dones)*max_targets_values*(self.gamma**self.n_steps)   
        return targets

    def update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
    
    def select_action(self,state):
        self.policy_net.eval()
        with torch.no_grad():
            with autocast(enabled=True):
                state = prep_observation_for_qnet(torch.Tensor(state).unsqueeze(0))
                q_values = self.policy_net(state)
                acts = torch.argmax(q_values,dim=1)[0]
                action =  acts.detach().cpu().numpy()
        self.policy_net.train()
        return action
    
    def optimize(self,batch_size,beta):
        batchs,info_buffer = self.replay_buffer.sample(batch_size,beta) # sample random transitions in replay buffer
        indices = info_buffer[0]
        weights = info_buffer[1].reshape(-1, 1)
        weights = torch.from_numpy(weights).cuda()
        states, new_states, actions, rewards ,dones = batchs
        actions = actions
        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1)
        self.optimizer.zero_grad()
        with autocast(enabled=True):
            q_values = self.policy_net(states) # (bs,num_actions)
            # we want to select the value according to the specified actions
            inputs = torch.gather(q_values,dim=1,index=actions.unsqueeze(-1)) # select among rows so dim = 1
            td_targets = self.compute_tdtarget(new_states,rewards,dones)
            elementwise_loss = self.criterion(td_targets,inputs)
            loss = torch.mean(elementwise_loss*weights)
            loss_for_prior = (inputs-td_targets).abs().detach().cpu().numpy()
            self.replay_buffer.update_priority(indices, loss_for_prior) #updtae priorities
            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),10)
            self.scaler.step(self.optimizer)
            self.scaler.update()

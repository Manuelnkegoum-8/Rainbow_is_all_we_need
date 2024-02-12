import torch
from torch import nn
import torch.nn.init as init
import numpy as np
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
    def __init__(self,env,hidden,atom_size,v_min,v_max):
        super(BaseNetwork,self).__init__()
        self.atom_size = atom_size
        self.v_max = v_max
        self.v_min = v_min
        self.out_dim = env.action_space.n
        self.input_size = env.observation_space.shape
        self.channels = self.input_size[0]
        self.model = AtariNet(env)
        self.flatten_size = 3136
        self.values_fc1 = NoisyLayer(self.flatten_size,hidden)
        self.values_fc2 = NoisyLayer(hidden,atom_size)

        self.advantages_fc1 = NoisyLayer(self.flatten_size,hidden)
        self.advantages_fc2 = NoisyLayer(hidden,atom_size*env.action_space.n)
    
    def forward(self,x):
        x = self.model(x)
        v = self.values_fc2(F.relu(self.values_fc1(x)))  # Value stream
        a = self.advantages_fc2(F.relu(self.advantages_fc1(x)))  # Advantage stream
        v, a = v.view(-1, 1, self.atom_size), a.view(-1, self.out_dim, self.atom_size)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams
        dist = F.softmax(q,dim=-1)
        return dist

    def reset_noise(self):
        self.advantages_fc1.reset_noise()
        self.advantages_fc2.reset_noise()
        self.values_fc1.reset_noise()
        self.values_fc2.reset_noise() 



class RainbowAgent:
    def __init__(self,env,hidden,gamma=0.99,tau=5e-3,v_min=0,v_max=200,
                 atom_size=51,n_steps=3,weights=None,device=None,alpha=0.5,size=1000000,eps=1e-7):
        super(RainbowAgent,self).__init__()
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.device = device
        self.policy_net = BaseNetwork(env,hidden,atom_size,v_min,v_max)

        if weights is not None:
            self.policy_net.load_state_dict(torch.load(weights,map_location='cpu'))
    
        self.target_net = BaseNetwork(env,hidden,atom_size,v_min,v_max)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.policy_net = self.policy_net.to(device)
        self.target_net = self.target_net.to(device)

        for param in self.target_net.parameters():
            param.requires_grad = False

        self.gamma = gamma
        self.tau = tau
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(device)
        self.n_steps = n_steps
        self.replay_buffer = PrioritizedReplayBuffer(alpha=alpha,
                                               size=size,
                                               eps=eps,
                                               n_steps=n_steps,
                                               gamma=gamma)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                 lr=6.25e-5,
                                 eps=1.5e-4)
    def compute_loss(self,states,actions,new_states,rewards,dones):
        batch_size = states.size(0)
        delta_z = (self.v_max - self.v_min) / (self.atom_size - 1)
        
        dist = self.policy_net(states)
        dist_ = dist[range(batch_size), actions]
        
        with torch.no_grad():
            self.target_net.reset_noise() 
            next_dist = self.policy_net(new_states)
            next_action = torch.sum(next_dist * next_dist, dim=2).argmax(dim=1)

            target_dist = self.target_net(new_states)
            next_dist   = target_dist[range(batch_size),next_action]

            targets = rewards + (1 - dones) * (self.gamma**self.n_steps) * self.support
            targets = targets.clamp(min=self.v_min, max=self.v_max)
            b = ((targets - self.v_min) / delta_z).float()
            l = b.floor().long()
            u = b.ceil().long()
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atom_size - 1)) * (l == u)] += 1
            offset = (
                        torch.linspace(
                            0, (batch_size - 1) * self.atom_size, batch_size
                        ).long()
                        .unsqueeze(1)
                        .expand(batch_size, self.atom_size)
                    ).to(self.device)
            proj_dist = torch.zeros(next_dist.size(),device = self.device)
            proj_dist.view(-1).index_add_(
                        0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
                    )
            proj_dist.view(-1).index_add_(
                        0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
                    )
        
        log_p = dist_.log()
        loss = -(proj_dist * log_p).sum(1)
        return loss

    def update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
    
    def select_action(self,state):
        with torch.no_grad():
            with autocast(enabled=True):
                state = prep_observation_for_qnet(torch.Tensor(state).unsqueeze(0))
                q_values = self.policy_net(state)
                acts = torch.sum(q_values * self.support, dim=2).argmax(dim=1)[0]
                action =  acts.detach().cpu().item()

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
        elementwise_loss = self.compute_loss(states,actions,new_states,rewards,dones)
        loss = torch.mean(elementwise_loss*weights)
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        self.replay_buffer.update_priority(indices, loss_for_prior) #updtae priorities
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),10)
        self.optimizer.step()


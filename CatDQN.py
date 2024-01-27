import torch
from torch import nn
import torch.nn.init as init
import numpy as np
import random,math
import torch.nn.functional as F

class BaseNetwork(nn.Module):
    def __init__(self,env,hidden,atom_size,v_min,v_max):
        super(BaseNetwork,self).__init__()
        self.atom_size = atom_size
        self.v_max = v_max
        self.v_min = v_min
        self.out_dim = env.action_space.n
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size)
        self.net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0],hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,atom_size*env.action_space.n)
            )
    def distribution(self,x):
        feat = self.net(x)
        q = feat.view(-1,self.out_dim,self.atom_size)
        dist = F.softmax(q,dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        return dist

    def forward(self,x):
        # X is a state/observation
        feat = self.distribution(x)
        q = torch.sum(feat * self.support, dim=2)
        return q #shape(bs,num_actions)


class CatAgent:
    def __init__(self,env,hidden,gamma=0.99,tau=5e-3,per=False,v_min=0,v_max=200,atom_size=51):
        super(CatAgent,self).__init__()
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.policy_net = BaseNetwork(env,hidden,atom_size,v_min,v_max)
        self.target_net = BaseNetwork(env,hidden,atom_size,v_min,v_max)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.gamma = gamma
        self.per = per
        self.tau = tau

    def compute_loss(self,states,actions,new_states,rewards,dones):
        """
        Compute the loss. implemented with the help of torch_rl
        """
        batch_size = states.size(0)
        delta_z = (self.v_max - self.v_min) / (self.atom_size - 1)
        support = self.target_net.support
        with torch.no_grad():
            next_action = self.target_net(new_states).argmax(dim=1)
            next_dist = self.target_net.distribution(new_states)
            next_dist = next_dist[range(batch_size), next_action]
            targets = rewards + (1 - dones) * self.gamma * support
            targets = targets.clamp(min=self.v_min, max=self.v_max)
            b = ((targets - self.v_min) / delta_z).float()
            l = b.floor().long()
            u = b.ceil().long()
            offset = (
                    torch.linspace(
                        0, (batch_size - 1) * self.atom_size, batch_size
                    ).long()
                    .unsqueeze(1)
                    .expand(batch_size, self.atom_size)
                )
            proj_dist = torch.zeros(next_dist.size())
            proj_dist.view(-1).index_add_(
                    0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
                )
            proj_dist.view(-1).index_add_(
                    0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
                )
        dist = self.policy_net.distribution(states)
        log_p = torch.log(dist[range(batch_size), actions])
        loss = -(proj_dist * log_p).sum(1)
        return loss.mean()


    def update(self):
        """
        Make a soft update of the parameters.
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
import torch
from torch import nn
import torch.nn.init as init
import numpy as np
import random,math
import torch.nn.functional as F
from Noisy import NoisyLayer

class AtariNet(nn.Module):
    def __init__(self,env,features = 128):
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
        x = self.convnet(x/255.)
        return x


class  BaseNetwork(nn.Module):
    def __init__(self,env,hidden,atom_size,v_min,v_max,atari,device):
        super(BaseNetwork,self).__init__()
        self.atom_size = atom_size
        self.v_max = v_max
        self.v_min = v_min
        self.out_dim = env.action_space.n
        self.input_size = env.observation_space.shape
        self.channels = self.input_size[0]
        
        if atari :
            self.model = AtariNet(env,features=hidden)
            self.flatten_size = self._infer_flat_size()
        else:
            self.flatten_size = hidden
            self.model = nn.Sequential(
                nn.Linear(env.observation_space.shape[0],self.flatten_size),
                nn.ReLU(),
            )
        self.values_fc1 = NoisyLayer(self.flatten_size,hidden)
        self.values_fc2 = NoisyLayer(hidden,atom_size)

        self.advantages_fc1 = NoisyLayer(self.flatten_size,hidden)
        self.advantages_fc2 = NoisyLayer(hidden,atom_size*env.action_space.n)
    
    def forward(self,x):
        feat = self.model(x)
        out1 = F.relu(self.values_fc1(feat))
        vals = self.values_fc2(out1).view(-1, 1, self.atom_size)
        out2 = F.relu(self.advantages_fc1(feat))
        adv = self.advantages_fc2(out2).view(-1, self.out_dim, self.atom_size)
        q = vals+adv - adv.mean(dim=1, keepdim=True)
        dist = F.softmax(q,dim=-1)
        return dist

    def reset_noise(self):
        self.advantages_fc1.reset_noise()
        self.advantages_fc2.reset_noise()
        self.values_fc1.reset_noise()
        self.values_fc2.reset_noise()     
    @torch.no_grad()
    def _infer_flat_size(self):
        output = self.model(torch.ones(1, *self.input_size))
        return int(np.prod(output.size()[1:]))
class RainbowAgent:
    def __init__(self,env,hidden,gamma=0.99,tau=5e-3,v_min=0,v_max=200,atom_size=51,n_steps=3,atari=False,device=None):
        super(RainbowAgent,self).__init__()
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.device = device
        self.policy_net = BaseNetwork(env,hidden,atom_size,v_min,v_max,atari,device).to(device)
        self.target_net = BaseNetwork(env,hidden,atom_size,v_min,v_max,atari,device).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        for param in self.target_net.parameters():
            param.requires_grad = False

        self.gamma = gamma
        self.tau = tau
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(device)
        self.n_steps = n_steps

    def compute_loss(self,states,actions,new_states,rewards,dones):
        batch_size = states.size(0)
        delta_z = (self.v_max - self.v_min) / (self.atom_size - 1)
        support = self.support.to(self.device)
        with torch.no_grad():
            next_dist = self.policy_net(new_states)
            next_action = torch.sum(next_dist * support, dim=2).argmax(dim=1)
            next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
            
            target_dist = self.target_net(new_states)
            next_dist   = target_dist.gather(1, next_action).squeeze(1)
            targets = rewards + (1 - dones) * (self.gamma**self.n_steps) * self.support
            targets = targets.clamp(min=self.v_min, max=self.v_max)
            b = ((targets - self.v_min) / delta_z).float()
            l = b.floor().long().to(self.device)
            u = b.ceil().long().to(self.device)
            offset = (
                        torch.linspace(
                            0, (batch_size - 1) * self.atom_size, batch_size
                        ).long()
                        .unsqueeze(1)
                        .expand(batch_size, self.atom_size)
                    ).to(self.device)
            proj_dist = torch.zeros(next_dist.size()).to(self.device)
            proj_dist.view(-1).index_add_(
                        0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
                    )
            proj_dist.view(-1).index_add_(
                        0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
                    )
        dist = self.policy_net(states)
        actions = actions.unsqueeze(1).expand(batch_size, 1, self.atom_size)
        dist = dist.gather(1, actions).squeeze(1)
        dist.data.clamp_(0.01, 0.99)
        log_p = dist.log()
        loss = -(proj_dist * log_p).sum(1)
        return loss

    def update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
    
    def select_action(self,state):
        self.policy_net.eval()
        with torch.no_grad():
            state = torch.Tensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            acts = torch.sum(q_values * self.support, dim=2).argmax(dim=1)[0]
            action =  acts.detach().cpu().item()
        self.policy_net.train()
        return action
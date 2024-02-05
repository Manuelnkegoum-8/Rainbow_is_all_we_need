import torch
from torch import nn
import torch.nn.init as init
import numpy as np
import random,math
import torch.nn.functional as F
from .Noisy import NoisyLayer

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
    def __init__(self,env,hidden,atari,device):
        super(BaseNetwork,self).__init__()
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
    @torch.no_grad()
    def _infer_flat_size(self):
        output = self.model(torch.ones(1, *self.input_size))
        return int(np.prod(output.size()[1:]))

class RainbowAgent2:
    def __init__(self,env,hidden,gamma=0.99,tau=5e-3,n_steps=3,atari=False,device=None):
        super(RainbowAgent2,self).__init__()
        self.criterion = nn.L1Loss(reduction="none")
        self.device = device
        self.policy_net = BaseNetwork(env,hidden,atari,device).to(device)
        self.target_net = BaseNetwork(env,hidden,atari,device).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        for param in self.target_net.parameters():
            param.requires_grad = False

        self.gamma = gamma
        self.tau = tau
        self.n_steps = n_steps

    def compute_loss(self,states,actions,new_states,rewards,dones):
        q_values = self.policy_net(states) # (bs,num_actions)
        # we want to select the value according to the specified actions
        inputs = torch.gather(q_values,dim=1,index=actions) # select among rows so dim = 1
        tmp = self.policy_net(new_states)
        max_idx = tmp.argmax(dim=1,keepdim=True)
        targets_values = self.target_net(new_states) # (bs,num_actions)
        max_targets_values = torch.gather(targets_values,dim=1,index=max_idx)
        targets = rewards + (1-dones)*max_targets_values*(self.gamma**self.n_steps)   
        #compute the loss
        loss = self.criterion(inputs, targets)
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
            if len(state.shape)>3 :
                state = torch.Tensor(state).to(self.device) # for vectorized envs
                q_values = self.policy_net(state)
                acts = torch.argmax(q_values,dim=1)[0]
                action =  acts.detach().cpu().numpy()
            else:
                state = torch.Tensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                acts = torch.argmax(q_values,dim=1)[0]
                action =  acts.detach().cpu().item()
        self.policy_net.train()
        return action

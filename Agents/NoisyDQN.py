import torch
from torch import nn
import torch.nn.init as init
import numpy as np
import random,math
import torch.nn.functional as F
from Noisy import NoisyLayer

class  DuelingNetwork(nn.Module):
    def __init__(self,env,hidden):
        super(DuelingNetwork,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(env.observation_space.shape[0],hidden),
            nn.ReLU(),
        )
        self.values_fc1 = NoisyLayer(hidden,hidden)
        self.values_fc2 = NoisyLayer(hidden,1)

        self.advantages_fc1 = NoisyLayer(hidden,hidden)
        self.advantages_fc2 = NoisyLayer(hidden,env.action_space.n)
 
    def forward(self,x):
        # X is a state/observation
        feat = self.model(x)
        out1 = F.relu(self.values_fc1(feat))
        vals = self.values_fc2(out1)
        out2 = F.relu(self.advantages_fc1(feat))
        adv = self.advantages_fc2(out2)
        q = vals+adv - adv.mean(dim=-1, keepdim=True)
        return q #shape(bs,num_actions)

    def reset_noise(self):
        self.advantages_fc1.reset_noise()
        self.advantages_fc2.reset_noise()
        self.values_fc1.reset_noise()
        self.values_fc2.reset_noise()

class  ClassicNetwork(nn.Module):
    def __init__(self,env,hidden):
        super(ClassicNetwork,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(env.observation_space.shape[0],hidden),
            nn.ReLU(),
        )
        self.fc1 = NoisyLayer(hidden,hidden)
        self.fc2 = NoisyLayer(hidden,env.action_space.n)
 
    def forward(self,x):
        # X is a state/observation
        feat = self.model(x)
        out1 = F.relu(self.fc1(feat))
        q = self.fc2(out1)
        return q #shape(bs,num_actions)

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()

class NoisyAgent:
    def __init__(self,env,hidden,gamma=0.99,tau=5e-3,double=False,dueling=False,per=False):
        super(NoisyAgent,self).__init__()
        if dueling:
            self.policy_net = DuelingNetwork(env,hidden)
            self.target_net = DuelingNetwork(env,hidden)
        else:
            self.policy_net = ClassicNetwork(env,hidden)
            self.target_net = ClassicNetwork(env,hidden)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.double = double
        self.gamma = gamma
        self.tau = tau
        self.criterion = nn.MSELoss() if not per else nn.L1Loss(reduction="none")

    def compute_loss(self,states,actions,new_states,rewards,dones):
        q_values = self.policy_net(states) # (bs,num_actions)
        # we want to select the value according to the specified actions
        inputs = torch.gather(q_values,dim=1,index=actions) # select among rows so dim = 1
        if self.double:
            tmp = self.policy_net(new_states)
            max_idx = tmp.argmax(dim=1,keepdim=True)
            targets_values = self.target_net(new_states) # (bs,num_actions)
            max_targets_values = torch.gather(targets_values,dim=1,index=max_idx)
        else:
            targets_values = self.target_net(new_states).detach() # (bs,num_actions)
            max_targets_values = targets_values.max(dim=1,keepdim=True)[0] # we keep batch_size with keeepdim
        targets = rewards + self.gamma*(1-dones)*max_targets_values   
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
            q_values = self.policy_net(torch.Tensor(state).unsqueeze(0))
            acts = torch.argmax(q_values,dim=1)[0]
            action =  acts.detach().cpu().item()
        self.policy_net.train()
        return action
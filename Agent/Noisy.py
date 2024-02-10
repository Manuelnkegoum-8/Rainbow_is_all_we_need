import torch
from torch import nn
import torch.nn.init as init
import numpy as np
import random,math
import torch.nn.functional as F
class NoisyLayer(nn.Module):
    """
    Base model for our dqn agent
    """
    def __init__(self,in_features,out_features):
        super(NoisyLayer,self).__init__()
        self.std_init = 0.5
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_std = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_std = nn.Parameter(torch.empty(out_features))
        self.register_buffer("noise_b", torch.empty(out_features))
        self.register_buffer(
            "noise_w", torch.empty(out_features, in_features)
        )
        self.reset_parameters()
        self.reset_noise()
        

    def f_noise(self,x):
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self): #need to be called at each step
        """Make new noise."""
        noise1 = self.f_noise(torch.randn(self.out_features))
        noise2 = self.f_noise(torch.randn(self.in_features))
        self.noise_w.copy_(noise1.outer(noise2))
        self.noise_b.copy_(noise1)

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_std.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_std.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def forward(self,x):
        return F.linear(x, self.weight_mu + self.weight_std*self.noise_w,self.bias_mu + self.bias_std*self.noise_b)
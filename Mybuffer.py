from torchrl.data import PrioritizedReplayBuffer,ListStorage
from collections import namedtuple, deque
import random,math

class MyPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    """
    To updtae the value of beta during training we need to subclass the 
    PrioritizedREplayBuffer class and add a update beta method
    """
    def update_beta(self, new_beta):
        self._sampler._beta = new_beta

class Replay_buffer(object):
    """
    To store episodes
    """
    def __init__(self,size=10000):
        super(Replay_buffer,self).__init__()
        self.memory = deque([], maxlen=size)

    def add(self,transition):
        """
        Transition : (state,action,reward,new_state,done)
        """
        return self.memory.append(transition)
    def sample(self,batch_size):
        """
        Sample random minibatch of transitions
        """
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)
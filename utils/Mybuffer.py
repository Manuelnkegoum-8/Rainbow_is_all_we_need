#from torchrl.data import PrioritizedReplayBuffer,ListStorage
from PrioritizedReplaybuffer import PrioritizedReplayBuffer
from collections import namedtuple, deque
import random,math,torch
import numpy as np
class MyPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    def n_step_sample(self,batch_size,n_steps,gamma,beta):
        batchs,info_buffer = self.sample(batch_size,beta) # sample random transitions in replay buffer
        indices = info_buffer['index']
        for j,i in enumerate(indices):
            sum_reward = 0
            states_look_ahead = self[i].next_state
            done_look_ahead = self[i].done
            for n in range(n_steps):
                if len(self) > i+n:
                    # compute the n-th reward
                    sum_reward += (gamma**n) * self[i+n].reward
                    if self[i+n].done:
                        states_look_ahead = torch.tensor(np.array(self[i+n].next_state)).float()
                        done_look_ahead = True
                        break
                    else:
                        states_look_ahead = torch.tensor(np.array(self[i+n].next_state)).float()
                        done_look_ahead = False
            batchs.reward[j] = sum_reward
            batchs.next_state[j] = states_look_ahead
            batchs.done[j] = done_look_ahead
        return batchs,info_buffer


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
from torchrl.data import PrioritizedReplayBuffer,ListStorage
from collections import namedtuple, deque
import random,math,torch

class MyPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    """
    To updtae the value of beta during training we need to subclass the 
    PrioritizedREplayBuffer class and add a update beta method
    """
    """def __init__(self, n_steps,gamma, *args, **kwargs):
        super(MyPrioritizedReplayBuffer, self).__init__(*args, **kwargs)
        self.n_steps = n_steps
        self.gamma = gamma"""

    def update_beta(self, new_beta):
        self._sampler._beta = new_beta
    
    def n_step_sample(self,batch_size,n_steps,gamma):
        batchs,info_buffer = self.sample(batch_size,return_info=True) # sample random transitions in replay buffer
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
                        states_look_ahead = torch.tensor(self[i+n].next_state).float()
                        done_look_ahead = True
                        break
                    else:
                        states_look_ahead = torch.tensor(self[i+n].next_state).float()
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
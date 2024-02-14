from collections import deque
import random
from math import sqrt
import numpy as np
import torch
from gymnasium.wrappers import LazyFrames


def prep_observation_for_qnet(tensor):
    """ Tranfer the tensor the gpu and reshape it into (batch, frame_stack*channels, y, x) """
    assert len(tensor.shape) == 5, tensor.shape # (batch,  frame_stack,y, x,channels)
    tensor = tensor.cuda().permute(0,1,4,2,3)
    # .cuda() needs to be before this ^ so that the tensor is made contiguous on the gpu
    tensor = tensor.reshape((tensor.shape[0], tensor.shape[1]*tensor.shape[2], *tensor.shape[3:]))
    return tensor.to(dtype= torch.float32) / 255 


class PrioritizedReplayBuffer:
    """ based on https://nn.labml.ai/rl/dqn, supports n-step bootstrapping and parallel environments,
    removed alpha hyperparameter like google/dopamine
    """

    def __init__(self, eps: float, size: int, gamma: float, n_steps: int, alpha : float):
        self.capacity = size  # must be a power of two
        self.gamma = gamma
        self.n_step = n_steps
        self.n_step_buffers = deque(maxlen=self.n_step + 1)
        self.alpha = alpha
        self.eps = eps
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]

        self.max_priority = 1.0  # initial priority of new transitions

        self.data = [None for _ in range(self.capacity)]  # cyclical buffer for transitions
        self.next_idx = 0  # next write location
        self.size = 0  # number of buffer elements


    def add(self, *transition):
        self.n_step_buffers.append(transition)

        if len(self.n_step_buffers) == self.n_step + 1 and not self.n_step_buffers[0][3]:  # n-step transition can't start with terminal state
            state = self.n_step_buffers[0][0]
            action = self.n_step_buffers[0][1]
            next_state = self.n_step_buffers[self.n_step][0]
            done = self.n_step_buffers[self.n_step][3]
            reward = self.n_step_buffers[0][2]
            for k in range(1, self.n_step):
                reward += self.n_step_buffers[k][2] * self.gamma ** k
                if self.n_step_buffers[k][3]:
                    done = True
                    break

            assert isinstance(state, LazyFrames)
            assert isinstance(next_state, LazyFrames)

            idx = self.next_idx
            self.data[idx] = (state,  action, reward, next_state,done)
            self.next_idx = (idx + 1) % self.capacity
            self.size = min(self.capacity, self.size + 1)

            self._set_priority_min(idx, self.max_priority**self.alpha)
            self._set_priority_sum(idx, self.max_priority**self.alpha)

    def _set_priority_min(self, idx, priority_alpha):
        idx += self.capacity
        self.priority_min[idx] = priority_alpha
        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):
        idx += self.capacity
        self.priority_sum[idx] = priority
        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):
        return self.priority_sum[1]

    def _min(self):
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        """ find the largest i such that the sum of the leaves from 1 to i is <= prefix sum"""

        idx = 1
        while idx < self.capacity:
            if self.priority_sum[idx * 2] > prefix_sum:
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1
        return idx - self.capacity

    def sample(self, batch_size: int, beta: float) -> tuple:
        weights = np.zeros(shape=batch_size, dtype=np.float32)
        indices = np.zeros(shape=batch_size, dtype=np.int32)

        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            indices[i] = idx

        prob_min = self._min() / self._sum()
        max_weight = (prob_min * self.size) ** (-beta)

        for i in range(batch_size):
            idx = indices[i]
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            weight = (prob * self.size) ** (-beta)
            weights[i] = weight / max_weight

        samples = []
        for i in indices:
            samples.append(self.data[i])
        
        return self.prepare_samples(samples),(indices,weights)

    def prepare_samples(self, batch):
        state,  action, reward,next_state, done = zip(*batch)
        state = list(map(lambda x: torch.from_numpy(x.__array__()), state))
        next_state = list(map(lambda x: torch.from_numpy(x.__array__()), next_state))
        action = list(map(lambda x:  torch.tensor(x,dtype=torch.long), action))
        reward = list(map(lambda x:  torch.tensor(x,dtype=torch.float32), reward))
        done = list(map(lambda x: torch.tensor(x,dtype=torch.float32), done))
        state, next_state, action, reward, done = map(torch.stack, [state, next_state, action, reward, done])
        return prep_observation_for_qnet(state), prep_observation_for_qnet(next_state), \
               action.cuda(), reward.cuda(), done.cuda()

    def update_priority(self, indexes, priorities):
        for idx, priority in zip(indexes, priorities):
            priority+=self.eps
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = priority**self.alpha
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    @property
    def is_full(self):
        return self.capacity == self.size
    
    def __len__(self):
        return self.size

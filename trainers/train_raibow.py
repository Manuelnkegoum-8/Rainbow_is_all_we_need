import torch
import argparse
from torch import nn
from utils.wrapper import wrap_deepmind
from tqdm import tqdm
from collections import namedtuple
from itertools import count
import torch.nn.init as init
import numpy as np
from torchrl.data import ListStorage
import random, math
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from  utils.Mybuffer import MyPrioritizedReplayBuffer
import gymnasium as gym
from Agents.Rainbow import RainbowAgent

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

def optimize(agent, replay_buffer, optimizer, batch_size, device, n_steps, gamma):
    if len(replay_buffer) < batch_size:
        return False
    batchs, info_buffer = replay_buffer.n_step_sample(batch_size, n_steps, gamma)
    indices = info_buffer['index']
    
    new_states = batchs.next_state.float().to(device)
    states = batchs.state.float().to(device)
    actions = batchs.action.long().unsqueeze(-1).to(device)
    rewards = batchs.reward.float().unsqueeze(-1).to(device)
    dones = batchs.done.float().unsqueeze(-1).to(device)
    
    elementwise_loss = agent.compute_loss(states, actions, new_states, rewards, dones)
    loss = torch.mean(elementwise_loss)
    loss_for_prior = elementwise_loss.detach().cpu().numpy()
    new_priorities = loss_for_prior
    replay_buffer.update_priority(indices, new_priorities)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    agent.policy_net.reset_noise()
    agent.target_net.reset_noise()
    return True

def train(agent, num_episodes, env, buffer, batch_size, device, optimizer, frequency, beta, gamma, n_steps):
    steps_done = 0.
    writer = SummaryWriter(envname)
    history = {"duration": [], "reward": []}
    agent.policy_net.train()
    
    for i in tqdm(range(1, num_episodes + 1)):
        state, _ = env.reset()
        total_reward = 0.
        for t in count():
            action = agent.select_action(state)
            steps_done += 1
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transition = Transition(np.array(state), action, reward, np.array(new_state), done)
            buffer.add(transition)
            state = new_state
            total_reward += reward
            
            if optimize(agent, buffer, optimizer, batch_size, device, n_steps, gamma):
                if steps_done % frequency == 0:
                    agent.update()
                    torch.save(agent.policy_net.state_dict(), 'policy.pt')
                    torch.save(agent.target_net.state_dict(), 'target.pt')
                
            if done:
                writer.add_scalar('/Reward/Rainbow', total_reward, i)
                history["duration"].append(t + 1)
                history["reward"].append(total_reward)
                break
        
        beta_ = min(1.0, beta + (1.0 - beta) * (i / num_episodes * 2))
        buffer.update_beta(beta_)
        
        if i % 20 == 0:
            print(f"Episode: {i}\t reward: {total_reward}")
    
    return history

def evaluate(agent, env, num_evals, device):
    history = {"reward": []}
    writer = SummaryWriter(envname)
    agent.policy_net.eval()
    for k in range(num_evals):
        state, _ = env.reset()
        total_reward = 0
        while True:
            action = agent.select_action(state, evaluate=True)
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = new_state
            total_reward += reward
            if done:
                writer.add_scalar(f'/test/Reward/Rainbow', total_reward, k)
                history['reward'].append(total_reward)
                break
    return history

if __name__ == '__main__':
    """parser = argparse.ArgumentParser()
    args = parser.parse_args()"""

    envname = "ALE/MsPacman-v5"
    env = gym.make(envname, render_mode='rgb_array')
    env = wrap_deepmind(env, episode_life=True, clip_rewards=False, frame_stack=True, scale=True)
    
    # Parameters
    alpha = 0.5
    hidden = 128
    batch_size = 32
    atom_size = 51
    vmin = -10.
    vmax = 10.
    atari = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gamma = 0.99
    n_steps = 3
    TAU = 5e-3
    LR = 1e-3
    num_episodes = 1000
    beta = 0.4
    frequency = 1
    prior_eps = 1e-6

    agent = RainbowAgent(env, hidden, gamma, TAU, vmin, vmax, atom_size, n_steps, atari, device)
    replay_buffer = MyPrioritizedReplayBuffer(alpha=alpha, beta=beta, eps=prior_eps, storage=ListStorage(10000))
    optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=LR)

    print("[INFO] Training started for model: Rainbow")
    history = train(agent, num_episodes, env, replay_buffer, batch_size, device, optimizer, frequency, beta, gamma, n_steps)
    eval_history = evaluate(agent, env, 5, device)
    
    plt.subplot(1, 2, 1)
    plt.plot(history["duration"])
    plt.subplot(1, 2, 2)
    plt.plot(history["reward"])
    plt.show()

import torch
import argparse
from torch import nn
from wrapper import wrap_deepmind,make_atari
from helpers import optimize,custom_collate_fn
from tqdm import tqdm
from collections import namedtuple,deque
import torch.nn.init as init
import numpy as np
from torchrl.data import ListStorage
import random,math
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from Mybuffer import MyPrioritizedReplayBuffer
import gymnasium as gym
from Rainbow import RainbowAgent

Max_steps_done = int(5e6)
min_samples = int(8e4)
train_freq = 4
target_freq = int(32e3)

Transition = namedtuple('Transition',
                        ('state', 'action',  'reward','next_state','done'))

def train(agent,optimizer,env,buffer,batch_size,beta,n_steps,gamma,device):
    steps_done = 0.
    writer = SummaryWriter(envname)
    agent.policy_net.train()
    done = True
    scores = deque([],maxlen=100)
    total_reward = 0.
    for steps_done in tqdm(range(Max_steps_done)):
        if done:
            print("Frame:= {0}\t reward:= {1}".format(steps_done,total_reward))
            scores.append(total_reward)
            if len(scores)==100:
                writer.add_scalar('/train/avg_reward/Rainbow',np.mean(scores), steps_done)
            state,_ = env.reset()
            done = False
            total_reward = 0.

        state_ = np.array(state)
        action = agent.select_action(state_)
        new_state, reward, terminated, truncated , info = env.step(action)
        done = terminated or truncated
        transition = Transition(state,action,reward,new_state,done) # store transition in buffer
        buffer.add(transition)
        state = new_state
        total_reward+=reward
        if steps_done > min_samples:
            if steps_done%train_freq==0:
                boolean = optimize(agent,buffer,optimizer,batch_size,n_steps,gamma,device)
                torch.save(agent.policy_net.state_dict(),'policy.pt')
            if steps_done%target_freq==0:
                agent.update()
            beta_ = min(1.0,beta + (1.0 - beta) * (steps_done / Max_steps_done))
            buffer.update_beta(beta)

def evaluate(agent,env,num_evals,device):
    writer = SummaryWriter(envname)
    agent.policy_net.eval()
    for k in range(num_evals):
        state,_ = env.reset()
        i = 0
        while True:
            state = np.array(state)
            action = agent.select_action(state)
            new_state, reward, terminated, truncated , info = env.step(action)
            done = terminated or truncated
            state = new_state
            i+=reward
            if done:
                writer.add_scalar('/test/Reward/Rainbow',i, k)
                history['reward'].append(i)
                break


if __name__=='__main__':
    """envname = "CartPole-v1"
    env = gym.make(envname,max_episode_steps=200)"""
    envname = "SpaceInvadersNoFrameskip-v4"
    env = make_atari(envname,'rgb_array',max_episode_steps=18000)
    env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=False)
    alpha = 0.5
    beta = 0.4
    hidden = 512
    prior_eps=1e-7
    batch_size = 32
    atom_size = 51
    n_steps = 3
    vmin = -10
    vmax = 10
    LR = 6.25e-5
    TAU = 5e-3
    gamma = 0.99
    atari = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = RainbowAgent(env,hidden,gamma,TAU,vmin,vmax,atom_size,n_steps,atari,device)

    replay_buffer = MyPrioritizedReplayBuffer(alpha=alpha,
                                            beta=beta,
                                            eps=prior_eps,
                                            storage=ListStorage(int(1e5)),
                                            collate_fn=custom_collate_fn)
    optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=LR,eps=1.5e-4)

    torch.autograd.set_detect_anomaly(True)
    print("[INFO] Training started for modelRainbow")

    train(agent,optimizer,env,replay_buffer,batch_size,beta,n_steps,gamma,device)
    evaluate(agent,env,5,device)
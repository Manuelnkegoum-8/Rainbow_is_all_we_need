import gymnasium as gym
from Agent.Rainbow import RainbowAgent
import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from common.wrapper import wrap_deepmind,make_atari
from gymnasium.wrappers.monitoring import video_recorder

TAU = 5e-3
gamma = 0.99
def evaluate(agent,env,num_evals,device):
    writer = SummaryWriter(f'logs/{envname}')
    agent.policy_net.eval()
    for k in tqdm(range(num_evals),ncols=80):
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
                break

if __name__=='__main__':
    envname = "SpaceInvadersNoFrameskip-v4"
    env = make_atari(envname,render_mode='rgb_array',max_episode_steps=108000)
    env.metadata['render_fps'] = 20
    env = wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, scale=False)
    hidden = 512
    atom_size = 51
    vmin = -10.
    vmax = 10.
    n_steps = 3
    atari = True
    epsilon = 0.0001
    device = torch.device('cpu')
    weights = 'weights.pt'
    agent = RainbowAgent(env,hidden,gamma,TAU,vmin,vmax,atom_size,n_steps,weights,device)

    evaluate(agent,env,100,device)
    state, _ = env.reset()
    vid = video_recorder.VideoRecorder(env, path="Rainbow_{}.mp4".format(envname))
    rew = 0.
    done = False
    while not done:
        frame = env.render()
        env.render()
        vid.capture_frame()
        state = np.array(state)
        action = agent.select_action(state)
        new_state, reward, terminated, truncated , info = env.step(action)
        done = terminated or truncated
        state = new_state
        rew+=reward
    vid.close()
    env.close()
    print(rew)




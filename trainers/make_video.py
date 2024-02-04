import gymnasium as gym
from Agents.Rainbow import RainbowAgent
import cv2
import numpy as np
import torch
from utils.wrapper import wrap_deepmind,make_atari
from gymnasium.wrappers.monitoring import video_recorder

parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--game', type=str, default='SpaceInvadersNoFrameskip-v4', help='ATARI game use no Frame skip version!!')

parser.add_argument('--max_episode_length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length (0 to disable)')
parser.add_argument('--hidden_size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V_min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V_max', type=float, default=10, metavar='V', help='Maximum of value distribution support')

parser.add_argument('--multi_step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--reward_clip', type=bool, default=True, metavar='VALUE', help='Reward clipping (0 to disable)')




if __name__=='__main__':
    envname = args.game
    env = make_atari(envname,'rgb_array',args.max_episode_length)
    env = wrap_deepmind(env,
                         episode_life=True, clip_rewards=args.reward_clip,
                         frame_stack=True, scale=False) # scaling must be set to false to save memory 
    hidden = args.hidden_size
    atom_size = args.atoms
    n_steps = args.multi_step
    vmin = args.V_min
    vmax = args.V_max
    TAU = 5e-3
    gamma = args.discount
    atari = True
    agent = RainbowAgent(env,hidden,gamma,TAU,vmin,vmax,atom_size,n_steps,atari)
    agent.policy_net.load_state_dict(torch.load('weights/policy.pt',map_location='cpu'))
    state, _ = env.reset()
    vid = video_recorder.VideoRecorder(env, path="Rainbow_{}.mp4".format(envname))
    rew = 0.
    done = False
    while not done:
        frame = env.render()
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




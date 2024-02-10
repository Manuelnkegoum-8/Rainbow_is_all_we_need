import torch
import argparse
from colorama import Fore, Style
from wrapper import wrap_deepmind,make_atari
from tqdm import trange
from collections import deque
import numpy as np
import random,math,time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from utils.PrioritizedReplaybuffer import PrioritizedReplayBuffer
import gymnasium as gym
from Agents.Rainbow import RainbowAgent
from Agents.Rainbow2 import RainbowAgent2
rainbow_colors = [Fore.RED, Fore.YELLOW, Fore.GREEN, Fore.CYAN, Fore.BLUE, Fore.MAGENTA]

def printf(text='='*80):
    for char in text:
        color = rainbow_colors[0]
        print(color + char, end="", flush=True)
        time.sleep(0.001)
        rainbow_colors.append(rainbow_colors.pop(0))
    print('\n'+Style.RESET_ALL)
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--game', type=str, default='SpaceInvadersNoFrameskip-v4', help='ATARI game use no Frame skip version!!')
parser.add_argument('--steps', type=int, default=int(5e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
parser.add_argument('--max_episode_length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length (0 to disable)')
parser.add_argument('--hidden_size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V_min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V_max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--memory_capacity', type=int, default=int(1e5), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--train_frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority_exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority_weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi_step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target_update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward_clip', type=bool, default=True, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--lr', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam_eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch_size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--min_samples', type=int, default=int(2e4), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluation_episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--variant', type=str, default='simple', metavar='N', help='varaint of rainbow used')

args = parser.parse_args()
print(' ' * 35,end='')
printf('Parameters')
printf()
for k, v in vars(args).items():
  print(Fore.GREEN+k + ': '+Style.RESET_ALL + str(v))
printf()
random.seed(args.seed)
torch.manual_seed(random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(random.randint(1, 10000))
  torch.backends.cudnn.enabled = False
else:
  args.device = torch.device('cpu')

envname = args.game
Max_steps_done = args.steps
min_samples = args.min_samples
train_freq = args.train_frequency
target_freq = args.target_update

def train(agent,env,batch_size,beta):
    steps_done = 0.
    writer = SummaryWriter(f'logs/{envname}')
    agent.policy_net.train()
    done = True
    scores = deque([],maxlen=10)
    total_reward = 0.
    for steps_done in trange(1,Max_steps_done+1,ncols=80):

        if done:
            print("\nFrame:= {0}\t reward:= {1}".format(steps_done,total_reward))
            scores.append(total_reward)
            if len(scores)==10:
                writer.add_scalar('/train/avg_reward/Rainbow',np.mean(scores), steps_done)
            state,_ = env.reset()
            done = False
            total_reward = 0.

        if steps_done%train_freq==0: # reset the noisy-nets noise in the policy
            agent.policy_net.reset_noise()
  
        state_ = np.array(state)
        action = agent.select_action(state_)
        new_state, reward, terminated, truncated , info = env.step(action)
        done = terminated or truncated
        # store transition in buffer
        agent.replay_buffer.add(state,action,reward,done)
        state = new_state
        total_reward+=reward
        if steps_done > min_samples:
            beta_ = min(1.0,beta + (1.0 - beta) * (steps_done / Max_steps_done))
            if steps_done%train_freq==0:
                agent.optimize(batch_size,beta_)
            if steps_done%target_freq==0:
                agent.update()
            if steps_done%20000==0:
                torch.save(agent.policy_net.state_dict(),'weights.pt')


if __name__=='__main__':
    
    env = make_atari(envname,'rgb_array',args.max_episode_length)
    env = wrap_deepmind(env,
                         episode_life=True, clip_rewards=args.reward_clip,
                         frame_stack=True, scale=False) # scaling must be set to false to save memory 
    alpha = args.priority_exponent
    beta = args.priority_weight
    hidden = args.hidden_size
    prior_eps = 1e-7
    batch_size = args.batch_size
    atom_size = args.atoms
    n_steps = args.multi_step
    vmin = args.V_min
    vmax = args.V_max
    LR = args.lr
    TAU = 5e-3
    gamma = args.discount
    atari = True
    device = args.device
    if args.variant=='simple':
        agent = RainbowAgent(env,hidden,gamma,TAU,vmin,vmax,atom_size,n_steps,None,device,alpha,args.memory_capacity,prior_eps)
    else:
        agent = RainbowAgent2(env,hidden,gamma,TAU,n_steps,None,device,alpha,args.memory_capacity,prior_eps)
    
    printf()
    print("[INFO] Training started for Rainbow 🌈 (see logs for details)")
    printf()
    torch.autograd.set_detect_anomaly(True)
    train(agent,env,batch_size,beta)
    printf()
    print("[INFO] End training")
    printf()
    print("[INFO] Evaluation started (see logs for details)")
    evaluate(agent,env,args.evaluation_episodes,device)
    printf()
    print("[INFO] DONE")
    print(Style.RESET_ALL)


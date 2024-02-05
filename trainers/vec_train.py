import torch
import argparse
from colorama import Fore, Style
from stable_baselines3.common.vec_env import SubprocVecEnv,VecFrameStack,VecMonitor
from utils.wrapper import wrap_deepmind,make_atari
from .helpers import optimize
from tqdm import tqdm
from collections import namedtuple,deque
import numpy as np
import random,math,time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from utils.Mybuffer import MyPrioritizedReplayBuffer
import gymnasium as gym
from Agents.Rainbow import RainbowAgent

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
parser.add_argument('--memory_capacity', type=int, default=int(2e4), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--train_frequency', type=int, default=1, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority_exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority_weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi_step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target_update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward_clip', type=bool, default=True, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--lr', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam_eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch_size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--min_samples', type=int, default=int(1e4), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluation_episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')


args = parser.parse_args()
envname = args.game
Max_steps_done = args.steps
min_samples = args.min_samples
train_freq = args.train_frequency
target_freq = args.target_update

Transition = namedtuple('Transition',
                        ('state', 'action',  'reward','next_state','done'))

def train(agent,optimizer,env,buffer,batch_size,beta,n_steps,gamma,device):
    steps_done = 0.
    writer = SummaryWriter(f'logs/{envname}')
    agent.policy_net.train()
    done = True
    scores = deque([],maxlen=100)
    states = env.reset()
    for steps_done in tqdm(range(Max_steps_done),ncols=80):
        if len(scores)==100:
            writer.add_scalar('/train/avg_reward/Rainbow',np.mean(scores), steps_done)
        states_ = np.array(states)
        actions = agent.select_action(states_)
        new_states, rewards, dones, infos = env.step(actions)
        for obs,new_state,action,reward,done,info in zip(states,new_states,actions,rewards, dones,infos):
            transition = Transition(obs,action,reward,new_state,done) # store transition in buffer
            buffer.add(transition)
            if done:
                rew = info['episode']['r']
                print("\nFrame:= {0}\t total_reward:= {1}".format(steps_done,rew))
                scores.append(rew)
        states = new_states
        if steps_done > min_samples:
            beta_ = min(1.0,beta + (1.0 - beta) * (steps_done / Max_steps_done*2))
            if steps_done%train_freq==0:
                boolean = optimize(agent,buffer,optimizer,batch_size,n_steps,gamma,beta_,device)
                torch.save(agent.policy_net.state_dict(),'rerehe.pt')
            if steps_done%target_freq==0:
                agent.update()

def evaluate(agent,env,num_evals,device):
    writer = SummaryWriter(f'logs/{envname}')
    agent.policy_net.eval()
    for k in tqdm(range(num_evals),ncols=80):
        state = env.reset()
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
    #env = make_atari(envname,'rgb_array',args.max_episode_length)

    print("[INFO] Training started for Rainbow 🌈 (see logs for details)")
    printf()
    make_env = lambda : wrap_deepmind(make_atari(envname,'rgb_array',args.max_episode_length),
                         episode_life=True, clip_rewards=args.reward_clip,
                         frame_stack=False, scale=False) # scaling must be set to false to save memory
    env = SubprocVecEnv([lambda: make_env() for _ in range(4)])
    env = VecFrameStack(env,4)
    env = VecMonitor(env)
    s = env.reset()
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
    state = env.reset()
    agent = RainbowAgent(env,hidden,gamma,TAU,vmin,vmax,atom_size,n_steps,atari,device)

    replay_buffer = MyPrioritizedReplayBuffer(alpha=alpha,
                                               size=args.memory_capacity,
                                               eps=prior_eps)
    optimizer = torch.optim.Adam(agent.policy_net.parameters(),
                                 lr=args.lr,
                                 eps=args.adam_eps)
    
    
    with Profile() as profile:
        train(agent,optimizer,env,replay_buffer,batch_size,beta,n_steps,gamma,device)
        (
        Stats(profile)
         .strip_dirs()
         .sort_stats(SortKey.CUMULATIVE)
         .print_stats()
     )
    printf()
    print("[INFO] End training")
    printf()
    print("[INFO] Evaluation started (see logs for details)")
    #evaluate(agent,env,args.evaluation_episodes,device)
    printf()
    print("[INFO] DONE")
    print(Style.RESET_ALL)
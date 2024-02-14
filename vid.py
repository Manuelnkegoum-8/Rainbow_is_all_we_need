import gymnasium as gym
from Agent.Rainbow import RainbowAgent
from colorama import Fore, Style
import cv2,argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from common.wrapper import wrap_deepmind,make_atari
from gymnasium.wrappers.monitoring import video_recorder

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
parser.add_argument('--game', type=str, default='SpaceInvaders', help='ATARI game to use')
parser.add_argument('--weights', type=str, default='weights.pt', help='weigts path')
parser.add_argument('--max_episode_length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length (0 to disable)')
parser.add_argument('--hidden_size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V_min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V_max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--multi_step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--evaluation_episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--variant', type=str, default='simple', metavar='N', help='varaint of rainbow used')

args = parser.parse_args()
random.seed(args.seed)
torch.manual_seed(random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(random.randint(1, 10000))
  torch.backends.cudnn.enabled = False
else:
  args.device = torch.device('cpu')
  
  
envname = args.game+'NoFrameskip-v4'
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
    env = make_atari(envname,render_mode='rgb_array',max_episode_steps=108000)
    env.metadata['render_fps'] = 20
    env = wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, scale=False)
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
    device = args.device
    weights =  args.weights
    if args.variant=='simple':
        agent = RainbowAgent(env,hidden,gamma,TAU,vmin,vmax,atom_size,n_steps,args.weights,device,alpha,args.memory_capacity,prior_eps)
    else:
        agent = RainbowAgent2(env,hidden,gamma,TAU,n_steps,args.weights,device,alpha,args.memory_capacity,prior_eps)
    
    printf()
    print("[INFO]Succesfully loaded weights.Evaluation started for Rainbow 🌈 (see logs for details)")
    printf()
    evaluate(agent,env,100,device)
    state, _ = env.reset()
    vid = video_recorder.VideoRecorder(env, path="Rainbow_{}.mp4".format(envname))
    rew = 0.
    done = False
    print("[INFO]Recording ...")
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




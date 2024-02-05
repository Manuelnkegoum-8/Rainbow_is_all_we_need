import numpy as np
import torch


def optimize(agent,replay_buffer,optimizer,batch_size,n_steps,gamma,beta,device):

    batchs,info_buffer = replay_buffer.sample(batch_size,beta) # sample random transitions in replay buffer
    indices = info_buffer['index']
    weights = torch.FloatTensor(
                info_buffer['_weight'].reshape(-1, 1)
            ).to(device)
    # create batch of states, new_states, rewards ,actions
    new_states = batchs.next_state.to(device)
    states = batchs.state.to(device)
    actions = batchs.action.unsqueeze(-1).to(device)
    rewards = batchs.reward.unsqueeze(-1).to(device)
    dones = batchs.done.unsqueeze(-1).to(device)
    elementwise_loss = agent.compute_loss(states,actions,new_states,rewards,dones)
    loss = torch.mean(elementwise_loss*weights)
    loss_for_prior = elementwise_loss.detach().cpu().numpy()
    new_priorities = loss_for_prior
    replay_buffer.update_priority(indices, new_priorities)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(),10)
    optimizer.step()
    agent.policy_net.reset_noise()
    agent.target_net.reset_noise()

def optimize2(agent,replay_buffer,optimizer,batch_size,n_steps,gamma,beta,device):
    batchs,info_buffer = replay_buffer.sample(batch_size,beta) # sample random transitions in replay buffer
    #batchs,info_buffer = replay_buffer.n_step_sample(batch_size,n_steps,gamma,beta) # sample random transitions in replay buffer
    indices = info_buffer['index']
    weights = torch.FloatTensor(
                info_buffer['_weight'].reshape(-1, 1)
            ).to(device)
    # create batch of states, new_states, rewards ,actions
    new_states = batchs.next_state.to(device)
    states = batchs.state.to(device)
    actions = batchs.action.unsqueeze(-1).to(device)
    rewards = batchs.reward.unsqueeze(-1).to(device)
    dones = batchs.done.unsqueeze(-1).to(device)
    elementwise_loss = agent.compute_loss(states,actions,new_states,rewards,dones)
    loss = torch.mean(weights*(elementwise_loss)**2)
    loss_for_prior = elementwise_loss.detach().cpu().numpy()
    new_priorities = loss_for_prior
    replay_buffer.update_priority(indices, new_priorities)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(),10)
    optimizer.step()
    agent.policy_net.reset_noise()
    agent.target_net.reset_noise()

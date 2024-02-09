import numpy as np
import torch


def optimize(agent,replay_buffer,optimizer,batch_size,n_steps,gamma,beta,device):
    batchs,info_buffer = replay_buffer.sample(batch_size,beta) # sample random transitions in replay buffer
    indices = info_buffer['index']
    weights = torch.FloatTensor(
                info_buffer['_weight'].reshape(-1, 1)
            ).to(device)
    states, new_states, actions, rewards ,dones = batchs
    new_states = new_states.to(device)
    states = states.to(device)
    actions = actions.unsqueeze(-1).to(device)
    rewards = rewards.unsqueeze(-1).to(device)
    dones = dones.unsqueeze(-1).to(device)

    elementwise_loss = agent.compute_loss(states,actions,new_states,rewards,dones)
    loss = torch.mean(elementwise_loss*weights)

    loss_for_prior = elementwise_loss.detach().cpu().numpy()
    replay_buffer.update_priority(indices, loss_for_prior) #updtae priorities
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(),10)
    optimizer.step()

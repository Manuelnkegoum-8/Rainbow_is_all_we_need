import numpy as np
import torch

class CustomBatch:
    def __init__(self, states, actions, rewards,next_states, dones):
        self.state = states
        self.next_state = next_states
        self.action = actions
        self.reward = rewards
        self.done = dones

def custom_collate_fn(batch):
    states = []
    next_states = []
    actions = []
    rewards = []
    dones = []

    for item in batch:
        state, action, reward,next_state, done = item
        states.append(torch.tensor(np.array(state), dtype=torch.float32))
        next_states.append(torch.tensor(np.array(next_state), dtype=torch.float32))
        actions.append(torch.tensor(action, dtype=torch.int64))
        rewards.append(torch.tensor(reward, dtype=torch.float32))
        dones.append(torch.tensor(done, dtype=torch.float32))

    # Stack all lists to create batched tensors
    batched_states = torch.stack(states)
    batched_next_states = torch.stack(next_states)
    batched_actions = torch.stack(actions)
    batched_rewards = torch.stack(rewards)
    batched_dones = torch.stack(dones)
    return CustomBatch(batched_states,batched_actions, batched_rewards,batched_next_states, batched_dones)


def optimize(agent,replay_buffer,optimizer,batch_size,n_steps,gamma,device):

    batchs,info_buffer = replay_buffer.n_step_sample(batch_size,n_steps,gamma) # sample random transitions in replay buffer
    indices = info_buffer['index']
    weights = torch.FloatTensor(
                info_buffer["_weight"].reshape(-1, 1)
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
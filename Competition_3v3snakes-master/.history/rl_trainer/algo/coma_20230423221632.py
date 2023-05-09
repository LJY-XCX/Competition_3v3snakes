import fnmatch
import os
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from replay_buffer import ReplayBuffer
from common import soft_update, hard_update, device
from algo.network import Actor, Critic

class COMA:

    def __init__(self, obs_dim, act_dim, num_agent, args):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = device
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.model_episode = args.model_episode
        self.eps = args.epsilon
        self.decay_speed = args.epsilon_speed
        self.output_activation = args.output_activation

        # Initialise actor network and critic network with ξ and θ
        self.actor = Actor(obs_dim, act_dim, num_agent, args, self.output_activation).to(self.device)
        self.critic = Critic(obs_dim, act_dim, num_agent, args).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim, num_agent, args).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)
        hard_update(self.critic, self.critic_target)

        # Initialise replay buffer R
        self.replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size)

        self.c_loss = None
        self.a_loss = None

    def choose_action(self, obs, evaluation=False):

        p = np.random.random()
        if p > self.eps or evaluation:
            obs = torch.Tensor([obs]).to(self.device)
            action = self.actor(obs).cpu().detach().numpy()[0]
        else:
            action = self.random_action()

        self.eps *= self.decay_speed
        return action
    
    def random_action(self):
        if self.output_activation == 'tanh':
            return np.random.uniform(low=-1, high=1, size=(self.num_agent, self.act_dim))
        return np.random.uniform(low=0, high=1, size=(self.num_agent, self.act_dim))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.get_batches()

        state_batch = torch.Tensor(state_batch).reshape(self.batch_size, self.num_agent, -1).to(self.device)
        action_batch = torch.Tensor(action_batch).reshape(self.batch_size, self.num_agent, -1).to(self.device)
        reward_batch = torch.Tensor(reward_batch).reshape(self.batch_size, self.num_agent, 1).to(self.device)
        next_state_batch = torch.Tensor(next_state_batch).reshape(self.batch_size, self.num_agent, -1).to(self.device)
        done_batch = torch.Tensor(done_batch).reshape(self.batch_size, self.num_agent, 1).to(self.device)
        
        # Compute target value for each agent in each transition using the Bi-RNN
        with torch.no_grad():
            target_next_actions = self.actor(next_state_batch)
            target_next_q = self.critic_target(next_state_batch, target_next_actions)
            q_hat = reward_batch + self.gamma * target_next_q * (1 - done_batch)

        # CHANGED: Update the critic network
        main_q = self.critic(state_batch, action_batch)
        loss_critic = torch.nn.MSELoss()(q_hat, main_q)

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        # ADDED: Calculate baseline
        mac_out = self.actor(state_batch)
        mac_out = mac_out / mac_out.sum(dim=-1, keepdim=True)
        pi = mac_out.view(-1, self.act_dim)
        baseline = (pi * main_q.view(-1, 1)).sum(-1).detach()

        # Calculate policy grad with mask
        q_taken = main_q.view(-1, self.act_dim).gather(dim=1, index=action_batch.view(-1, 1)).squeeze(1)
        pi_taken = pi.gather(dim=1, index=action_batch.view(-1, 1)).squeeze(1)
        log_pi_taken = torch.log(pi_taken)
        mask = (1 - done_batch.view(-1)).detach()

        advantages = (q_taken - baseline).detach()
        loss_actor = - ((advantages * log_pi_taken) * mask).sum() / mask.sum()

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()

        self.c_loss = loss_critic.item()
        self.a_loss = loss_actor.item()

        # Update the target networks
        soft_update(self.critic, self.critic_target, self.tau)

        return self.c_loss, self.a_loss
    
    
    def _compute_targets(self, next_state_batch, reward_batch, done_batch):
        # Compute targets for the critic
        # You can modify this function to compute targets using your global actor network

        # Obtain actions from the global actor network for the next_state_batch
        next_actions = self.actor(next_state_batch)

        # Obtain Q-values for the next state and actions using the target critic network
        target_next_q = self.critic_target(next_state_batch, next_actions)

        # Compute the targets using the Bellman equation
        q_hat = reward_batch + self.gamma * target_next_q * (1 - done_batch)

        return q_hat
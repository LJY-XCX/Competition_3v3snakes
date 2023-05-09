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

        # Calculate targets for the critic
        with torch.no_grad():
            q_hat = self._compute_targets(next_state_batch, reward_batch, done_batch)

        # Update the critic networks
        main_q = self.critic(state_batch, action_batch)
        loss_critic = torch.nn.MSELoss()(q_hat, main_q)

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        # Sample joint actions for each agent
        mac_out = self.actor(state_batch)

        # Calculate Q values for all possible actions for each agent
        q_vals = self.critic(state_batch, mac_out)

        # Compute the counterfactual baseline
        pi = mac_out.view(-1, self.n_actions)
        baseline = (pi * q_vals).sum(-1).detach()

        # Compute the COMA loss
        q_taken = torch.gather(q_vals, dim=1, index=action_batch.view(-1, 1)).squeeze(1)
        pi_taken = torch.gather(pi, dim=1, index=action_batch.view(-1, 1)).squeeze(1)
        log_pi_taken = torch.log(pi_taken)
        advantages = (q_taken - baseline).detach()
        mask = torch.ones_like(advantages)  # Modify this mask as required
        coma_loss = -((advantages * log_pi_taken) * mask).sum() / mask.sum()

        # Update the global actor networks
        self.actor_optimizer.zero_grad()
        coma_loss.backward()
        clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()

        self.c_loss = loss_critic.item()
        self.a_loss = coma_loss.item()

        # Update the target networks
        soft_update(self.critic, self.critic_target, self.tau)

        return self.c_loss, self.a_loss

    def baseline(self, state, action):
        with torch.no_grad():
            q = self.critic(state[np.newaxis, ...], action[np.newaxis, ...]).squeeze()
        policy = self.actor(state, action)
        counter_actions = action.unsqueeza(0).unsqueeze(0).repeat(self.num_agent, self.act_dim, 1)
        for i in range(self.num_agent):
            counter_actions[i, :, i] = torch.arange(counter_actions.shape[1])
        
        with torch.no_grad():
            counter_q = self.critic(state, counter_actions)
        counter_q = counter_q.reshape(self.num_agent, -1)

        baseline = q - torch.matmul(policy, counter_q)

        return baseline, policy
    
    
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
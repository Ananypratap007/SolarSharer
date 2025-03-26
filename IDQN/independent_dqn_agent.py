 # independent_dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # state, next_state: 1D np.array (obs_dim)
        # action: int (discrete action index)
        # reward, done: float
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class IndependentDQNAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        lr=1e-3,
        gamma=0.99,
        tau=0.01,       # for target network soft update
        buffer_capacity=5000,
        batch_size=32,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=5000,
        device="cpu"
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device

        # Epsilon for exploration
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0
        self.eps = eps_start

        # Q-Network & Target Q-Network
        self.q_net = QNetwork(obs_dim, act_dim).to(device)
        self.target_q_net = QNetwork(obs_dim, act_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, obs):
        """
        obs: 1D np.array of shape (obs_dim,) for THIS agent only.
        Returns a single discrete action index (0..act_dim-1).
        """
        self.steps_done += 1
        # Update epsilon
        self.eps = self.eps_end + (self.eps_start - self.eps_end) * \
                   np.exp(-1.0 * self.steps_done / self.eps_decay)

        if random.random() < self.eps:
            return random.randint(0, self.act_dim - 1)
        else:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                q_values = self.q_net(obs_t)  # shape [1, act_dim]
                action_idx = q_values.argmax(dim=1).item()
            return action_idx

    def update(self):
        """
        Sample from buffer, do one step of DQN update.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        (state, action, reward, next_state, done) = self.replay_buffer.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        # Current Q
        q_values = self.q_net(state)  # [batch_size, act_dim]
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)  # [batch_size]

        # Target Q
        with torch.no_grad():
            next_q_values = self.target_q_net(next_state)  # [batch_size, act_dim]
            next_q_max = next_q_values.max(dim=1)[0]       # [batch_size]

        q_target = reward + self.gamma * next_q_max * (1 - done)

        loss = nn.MSELoss()(q_value, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update
        for target_param, local_param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def push_transition(self, s, a, r, s_next, d):
        self.replay_buffer.push(s, a, r, s_next, d)

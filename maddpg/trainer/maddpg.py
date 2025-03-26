import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from maddpg.trainer.replay_buffer import ReplayBuffer
import torch.nn.utils as nn_utils

def mlp_block(input_dim, hidden_dim=64, output_dim=64):
    """Helper for building a simple MLP block."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU()
    )

class Actor(nn.Module):
    """Actor network: maps local state -> local action."""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # outputs actions in [-1, 1]
        )

    def forward(self, state):
        return self.net(state)

class SharedCritic(nn.Module):
    def __init__(self, global_state_dim, global_action_dim, hidden_dim=128, num_agents=1):
        super().__init__()
        self.common = nn.Sequential(
            nn.Linear(global_state_dim + global_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.Q_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_agents)])
        
    def forward(self, global_state, global_action):
        # global_state:  (B, global_state_dim)
        # global_action: (B, global_action_dim)
        x = torch.cat([global_state, global_action], dim=1)
        h = self.common(x)
        # Compute each agent's Q-value => (B, num_agents)
        Q_values = [head(h) for head in self.Q_heads]  # list of (B,1)
        Q_values = torch.cat(Q_values, dim=1)          # (B, num_agents)
        return Q_values

class Agent:
    """
    An individual agent. 
    """
    def __init__(self, 
                 local_state_dim,     
                 action_dim,          
                 lr_actor=1e-3,
                 gamma=0.95,
                 tau=0.01,
                 device=torch.device("cpu")):  # pass in device from outside

        self.local_state_dim = local_state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device  # store the device

        self.actor = Actor(local_state_dim, action_dim)
        self.target_actor = Actor(local_state_dim, action_dim)
        self.actor.to(self.device)
        self.target_actor.to(self.device)
        self._update_actor_targets(tau=1.0)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)

    def _update_actor_targets(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def select_action(self, local_state, noise=0.1):
        """
        Select a single continuous action in [-1,1], then map to [0,1].
        """
        with torch.no_grad():
            s = torch.FloatTensor(local_state).unsqueeze(0).to(self.device)  # (1, local_state_dim)
            a = self.actor(s).cpu().numpy().flatten()                        # (action_dim,)

        # Add noise
        a += noise * np.random.randn(self.action_dim)
        a = np.clip(a, -1.0, 1.0)
        # Map [-1,1] -> [0,1] if your env expects [0,1]
        a = (a + 1.0) / 2.0
        return a

    def forward_actor(self, local_state):
        return self.actor(local_state)

class MADDPG:
   
    def __init__(self,
                 num_agents,
                 local_state_dim,
                 action_dim,
                 gamma=0.95,
                 tau=0.01,
                 lr_actor=1e-3,
                 lr_critic=1e-3,
                 buffer_size=100000):


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_agents = num_agents
        self.local_state_dim = local_state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

     
        global_state_dim = num_agents * local_state_dim
        global_action_dim = num_agents * action_dim


        self.agents = []
        for _ in range(num_agents):
            agent = Agent(local_state_dim, 
                          action_dim, 
                          lr_actor=lr_actor, 
                          gamma=gamma, 
                          tau=tau, 
                          device=self.device)
            self.agents.append(agent)

        self.shared_critic = SharedCritic(global_state_dim, 
                                          global_action_dim, 
                                          hidden_dim=128, 
                                          num_agents=num_agents)
        self.target_shared_critic = SharedCritic(global_state_dim, 
                                                 global_action_dim, 
                                                 hidden_dim=128, 
                                                 num_agents=num_agents)

    
        self.shared_critic.to(self.device)
        self.target_shared_critic.to(self.device)

   
        self._update_shared_critic_targets(tau=1.0)

 
        self.shared_critic_optim = optim.Adam(self.shared_critic.parameters(), lr=lr_critic)


        self.memory = ReplayBuffer(max_size=buffer_size)

        self.init_noise = 0.3  
        self.final_noise = 0.01
        self.noise_episodes = 300 
        self.current_noise = self.init_noise
        self.current_episode = 0

    def _update_shared_critic_targets(self, tau=None):
        
        if tau is None:
            tau = self.tau
        for tp, p in zip(self.target_shared_critic.parameters(), self.shared_critic.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    def update_noise(self):
        fraction = min(float(self.current_episode) / self.noise_episodes, 1.0)
        self.current_noise = self.init_noise + fraction * (self.final_noise - self.init_noise)

    def select_actions(self, states):
        actions = []
        for i, agent in enumerate(self.agents):
            action_i = agent.select_action(states[i], noise=self.current_noise)
            actions.append(action_i)
        return np.array(actions)

    def store_transition(self, states, actions, rewards, next_states, done):
        self.memory.add(states, actions, rewards, next_states, done)

    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        device = self.device 
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

       
        states_t      = torch.FloatTensor(states).to(device)       # (B, N, sdim)
        actions_t     = torch.FloatTensor(actions).to(device)      # (B, N, adim)
        rewards_t     = torch.FloatTensor(rewards).to(device)      # (B, N)
        next_states_t = torch.FloatTensor(next_states).to(device)  # (B, N, sdim)

        if dones.ndim == 1:
            dones_t = torch.FloatTensor(dones).to(device).unsqueeze(-1)  # (B,1)
        else:
            dones_t = torch.FloatTensor(dones).to(device)               # (B,N) if per-agent


        global_states      = states_t.view(batch_size, -1)       # (B, N*sdim)
        global_actions     = actions_t.view(batch_size, -1)      # (B, N*adim)
        global_next_states = next_states_t.view(batch_size, -1)

      
        with torch.no_grad():
            target_actions = []
            for i, agent in enumerate(self.agents):
                next_local_state = next_states_t[:, i, :]  # (B, sdim)
                next_act_i = agent.target_actor(next_local_state)
                target_actions.append(next_act_i)
            target_actions = torch.stack(target_actions, dim=1)        # (B, N, adim)
            global_next_actions = target_actions.view(batch_size, -1)  # (B, N*adim)

            # Get next Q-values for all agents => (B, N)
            next_Q_values = self.target_shared_critic(global_next_states, global_next_actions)

      
        Q_values = self.shared_critic(global_states, global_actions)

        critic_loss = 0.0
        for i in range(self.num_agents):
            if dones_t.shape == (batch_size, 1):
                done_i = dones_t  # single env-level done => (B,1)
            else:
                done_i = dones_t[:, i].unsqueeze(-1)  # per-agent done => (B,1)

            next_Q_i = next_Q_values[:, i].unsqueeze(-1)  # (B,1)
            y_i = rewards_t[:, i].unsqueeze(-1) + self.gamma * (1.0 - done_i) * next_Q_i
            Q_i = Q_values[:, i].unsqueeze(-1)            # (B,1)

            critic_loss += nn.MSELoss()(Q_i, y_i)

        # Optionally average across all agents
        critic_loss = critic_loss / self.num_agents

        self.shared_critic_optim.zero_grad()
        critic_loss.backward()
        nn_utils.clip_grad_norm_(self.shared_critic.parameters(), max_norm=1.0)
        self.shared_critic_optim.step()

        for i, agent in enumerate(self.agents):
            # Re-build actions with agent i's action as a variable, others fixed
            new_actions = []
            for j, other_agent in enumerate(self.agents):
                if j == i:
                    local_state_j = states_t[:, j, :]  # (B, sdim)
                    new_act_j = other_agent.actor(local_state_j)
                else:
                
                    new_act_j = actions_t[:, j, :]
                new_actions.append(new_act_j)

            new_actions = torch.stack(new_actions, dim=1)  # (B, N, adim)
            global_new_actions = new_actions.view(batch_size, -1)

          
            Q_values_new = self.shared_critic(global_states, global_new_actions)
     
            actor_loss = -Q_values_new[:, i].mean()

            agent.actor_optim.zero_grad()
            actor_loss.backward()
            nn_utils.clip_grad_norm_(agent.actor.parameters(), max_norm=1.0)
            agent.actor_optim.step()

            
            agent._update_actor_targets()

      
        self._update_shared_critic_targets()

    def on_episode_end(self):
        self.current_episode += 1
        self.update_noise()

    def __len__(self):
        return len(self.memory)

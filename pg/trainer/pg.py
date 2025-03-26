import numpy as np
import torch

class PGActorCritic(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=128, init_std=0.1):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.init_std = init_std
        self.base_net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        )
        self.actor_head = torch.nn.Linear(hidden_size, act_dim)
        self.critic_head = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        f = self.base_net(x)
        logits = self.actor_head(f)
        value = self.critic_head(f).squeeze(-1)
        return logits, value

    def get_action(self, obs):
        logits, value = self(obs)
        mean = torch.sigmoid(logits)
        std = self.init_std
        dist = torch.distributions.Normal(mean, std)
        a = dist.sample()
        a_clamped = torch.clamp(a, 0.0, 1.0)
        log_prob = dist.log_prob(a_clamped).sum(dim=-1)
        return a_clamped.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

    def get_log_prob_value(self, obs, actions):
        logits, values = self(obs)
        mean = torch.sigmoid(logits)
        dist = torch.distributions.Normal(mean, self.init_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        return log_probs, values


class PGRolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def store(self, obs, action, log_prob, value, reward, done):
        # obs and actions can be stored as arrays (no grad needed).
        # log_prob and value must remain Tensors to allow grad backprop.
        self.obs.append(obs)       # shape: (obs_dim,)
        self.actions.append(action)  # shape: (act_dim,)
        self.log_probs.append(log_prob)  # shape: scalar or [1]
        self.values.append(value)        # shape: scalar or [1]
        self.rewards.append(reward)      # python float
        self.dones.append(done)          # python bool

    def as_tensors(self, device):
        # Convert obs, actions, rewards, dones via np.array -> torch.FloatTensor
        obs_t = torch.FloatTensor(np.array(self.obs)).to(device)
        act_t = torch.FloatTensor(np.array(self.actions)).to(device)
        rew_t = torch.FloatTensor(np.array(self.rewards)).to(device)
        done_t = torch.BoolTensor(np.array(self.dones)).to(device)

        # For log_probs and values, stack them (each item is already a Tensor).
        lp_t = torch.stack(self.log_probs).to(device)   # shape: [T]
        val_t = torch.stack(self.values).to(device)     # shape: [T]

        return obs_t, act_t, lp_t, val_t, rew_t, done_t

    def clear(self):
        self.__init__()


def compute_returns(rewards, dones, gamma=0.99, device=torch.device("cpu")):
    r = 0.0
    returns = []
    for reward, done in zip(reversed(rewards), reversed(dones)):
        r = reward + gamma * r * (1.0 - float(done))
        returns.append(r)
    returns.reverse()
    return torch.FloatTensor(returns).to(device)


def policy_gradient_update(
    policy,
    optimizer,
    obs,
    actions,
    log_probs_old,
    values_old,
    rewards,
    dones,
    gamma=0.99,
    device=torch.device("cpu")
):
    # Compute discounted returns
    returns = compute_returns(rewards, dones, gamma, device)
    # Compute advantage (returns - baseline)
    advantages = returns - values_old
    # Policy gradient loss
    policy_loss = -(log_probs_old * advantages).mean()
    # Value function loss
    value_loss = torch.nn.functional.mse_loss(values_old, returns)
    # Combine
    loss = policy_loss + 0.5 * value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def flatten_obs(obs_list):
    return np.concatenate(obs_list, axis=0)

def combine_rewards(rewards_list):
    return float(np.mean(rewards_list))

import numpy as np
import torch

class PPOActorCritic(torch.nn.Module):
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


class PPORolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def store(self, obs, action, log_prob, value, reward, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def as_tensors(self, device):
        obs_t = torch.FloatTensor(np.array(self.obs)).to(device)
        act_t = torch.FloatTensor(np.array(self.actions)).to(device)
        lp_t = torch.FloatTensor(np.array(self.log_probs)).to(device)
        val_t = torch.FloatTensor(np.array(self.values)).to(device)
        rew_t = torch.FloatTensor(np.array(self.rewards)).to(device)
        done_t = torch.BoolTensor(np.array(self.dones)).to(device)
        return obs_t, act_t, lp_t, val_t, rew_t, done_t

    def clear(self):
        self.__init__()


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = torch.zeros(T, dtype=torch.float32, device=rewards.device)
    ret = torch.zeros(T, dtype=torch.float32, device=rewards.device)
    gae = 0.0
    nv = 0.0
    for t in reversed(range(T)):
        nxt = 1.0 - float(dones[t])
        if t == T - 1:
            d = rewards[t] + gamma * nv * nxt - values[t]
        else:
            d = rewards[t] + gamma * values[t+1] * nxt - values[t]
        gae = d + gamma * lam * nxt * gae
        adv[t] = gae
        ret[t] = adv[t] + values[t]
        if dones[t]:
            gae = 0.0
            nv = 0.0
        else:
            nv = values[t]
    return adv, ret


def ppo_update(
    policy,
    optimizer,
    obs,
    actions,
    log_probs_old,
    values_old,
    rewards,
    dones,
    gamma=0.99,
    lam=0.95,
    clip_epsilon=0.2,
    ppo_epochs=10,
    batch_size=32,
    device=torch.device("cpu")
):
    advantages, returns = compute_gae(rewards, values_old, dones, gamma, lam)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    ds = len(obs)
    idxs = np.arange(ds)
    for _ in range(ppo_epochs):
        np.random.shuffle(idxs)
        s = 0
        while s < ds:
            e = s + batch_size
            b_idx = idxs[s:e]
            s = e
            b_obs = obs[b_idx]
            b_act = actions[b_idx]
            b_lp_old = log_probs_old[b_idx]
            b_adv = advantages[b_idx]
            b_ret = returns[b_idx]
            lp_new, val_new = policy.get_log_prob_value(b_obs, b_act)
            ratio = torch.exp(lp_new - b_lp_old)
            o1 = ratio * b_adv
            o2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * b_adv
            p_loss = -torch.min(o1, o2).mean()
            v_loss = torch.nn.functional.mse_loss(val_new, b_ret)
            loss = p_loss + 0.5 * v_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def flatten_obs(obs_list):
    return np.concatenate(obs_list, axis=0)

def combine_rewards(rewards_list):
    return float(np.mean(rewards_list))

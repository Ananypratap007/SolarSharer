# grid_search.py
import itertools
import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solar_sharer_env import SolarSharer
from maddpg.trainer.maddpg import MADDPG

def run_single_experiment(alpha, beta, gamma, delta, 
                          episodes=10, max_steps=48, noise=0.1):
    """
    Creates a new environment with the specified hyperparameters,
    runs short training, and returns the average final reward across episodes
    as a performance metric.
    """

    # 1) Create environment (adjust paths/params if needed)
    env = SolarSharer(
        data_path="data/filtered_pivoted_austin.csv",
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        max_grid_price=0.2112,  # match your main training code
        time_freq="15T",       # ...
        agg_method="mean"      # ...
    )

    # 2) Infer dimensions from env
    num_agents = env.num_agents
    local_state_dim = env.observation_space.shape[1]  # e.g. 6
    action_dim      = env.action_space.shape[1]       # e.g. 4

    # 3) Initialize MADDPG
    maddpg = MADDPG(
        num_agents=num_agents,
        local_state_dim=local_state_dim,
        action_dim=action_dim,
        gamma=0.95,
        tau=0.01,
        lr_actor=1e-3,
        lr_critic=1e-3,
        buffer_size=50_000  # smaller buffer for quicker tests
    )

    total_reward_across_episodes = 0.0

    # 4) Short training loop
    for ep in range(episodes):
        obs = env.reset()
        ep_reward = np.zeros(num_agents)

        for _ in range(max_steps):
            # Optionally pass in a noise_scale to add exploration noise
            actions = maddpg.select_actions(obs)
            
            # Environment step
            next_obs, rewards, done, info = env.step(actions)
            
            # Store transitions
            maddpg.store_transition(obs, actions, rewards, next_obs, done)
            
            # Accumulate reward
            ep_reward += rewards
            
            # Train MADDPG
            maddpg.train(batch_size=128)

            # Move to next state
            obs = next_obs

            if done:
                break

        # sum agent rewards => single scalar for ep
        total_reward_across_episodes += np.sum(ep_reward)
    
    # 5) final score = average reward across all episodes
    average_score = total_reward_across_episodes / episodes
    return average_score

def main():
    # Define your search ranges
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    beta_values  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    gamma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    delta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = []  # store (alpha, beta, gamma, delta, score)
    best_score = -1e9
    best_params = None

    # Cartesian product
    for alpha, beta, gamma, delta in itertools.product(alpha_values, 
                                                       beta_values, 
                                                       gamma_values, 
                                                       delta_values):
        score = run_single_experiment(
            alpha, 
            beta, 
            gamma, 
            delta,
            episodes=10,   # short number of episodes
            max_steps=48,  # half-day steps
            noise=0.1
        )

        print(f"Combo (W1={alpha}, W2={beta}, W3={gamma}, W4={delta}) => Score={score:.3f}")

        results.append((alpha, beta, gamma, delta, score))

        if score > best_score:
            best_score = score
            best_params = (alpha, beta, gamma, delta)

    print(f"\nBest Params: {best_params}, Best Score: {best_score:.3f}")

    
    results_arr = np.array(results)  # shape (N, 5): alpha, beta, gamma, delta, score

    alphas =  results_arr[:, 0]
    betas  =  results_arr[:, 1]
    gammas =  results_arr[:, 2]
    deltas =  results_arr[:, 3]
    scores =  results_arr[:, 4]

    plt.figure(figsize=(12, 8))

    # Subplot 1: alpha vs. score
    plt.subplot(2, 2, 1)
    plt.scatter(alphas, scores, alpha=0.7)
    plt.xlabel("Alpha")
    plt.ylabel("Score")
    plt.title("Alpha vs. Score")

    # Subplot 2: beta vs. score
    plt.subplot(2, 2, 2)
    plt.scatter(betas, scores, alpha=0.7)
    plt.xlabel("Beta")
    plt.ylabel("Score")
    plt.title("Beta vs. Score")

    # Subplot 3: gamma vs. score
    plt.subplot(2, 2, 3)
    plt.scatter(gammas, scores, alpha=0.7)
    plt.xlabel("Gamma")
    plt.ylabel("Score")
    plt.title("Gamma vs. Score")

    # Subplot 4: delta vs. score
    plt.subplot(2, 2, 4)
    plt.scatter(deltas, scores, alpha=0.7)
    plt.xlabel("Delta")
    plt.ylabel("Score")
    plt.title("Delta vs. Score")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

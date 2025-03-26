# train_dqn.py (modified to match your existing MADDPG/PPO structure)
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import time

from independent_dqn_agent import IndependentDQNAgent
from actions import ACTION_MAP  # Our discrete action map

# Ensure Python can find your local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solar_sharer_env import SolarSharer

def main():
    
    num_episodes = 1000        # e.g. 200 days
    steps_per_day = 96
    checkpoint_interval = 50

    # For logging
    episode_rewards = []
    agent_rewards_log = []
    best_mean_reward = -1e9
    best_model_path = "dqn_best.pth"

    training_start_time = time.time()
    episode_durations = []
    total_steps_global = 0
    episode_log_data = []

   
    env = SolarSharer(
        data_path="data/filtered_pivoted_austin.csv",
        alpha=0.3, beta=0.5, gamma=0.5, delta=0.4,
        max_grid_price=0.2112
    )
    num_agents = env.num_agents
    obs_dim = 6
    act_dim = 4

  
    agents = []
    for i in range(num_agents):
        agent = IndependentDQNAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            lr=1e-3,
            gamma=0.99,
            tau=0.01,
            buffer_capacity=1000,
            batch_size=128,
            eps_start=1.0,
            eps_end=0.05,
            eps_decay=5000,
            device="cpu"
        )
        agents.append(agent)

    
    for episode in range(1, num_episodes + 1):
        episode_start_time = time.time()

        
        day_idx = (episode - 1) % 10
        env.day_index = day_idx

        obs_list = env.reset()
        done = False
        step_count = 0
        total_reward_agents = np.zeros(num_agents, dtype=np.float32)

      
        day_logs = []

        while not done:
            chosen_actions_idx = np.zeros(num_agents, dtype=int)
            actions_array = np.zeros((num_agents, 4), dtype=np.float32)

            # Each agent picks an action
            for i in range(num_agents):
                a_idx = agents[i].select_action(obs_list[i])
                chosen_actions_idx[i] = a_idx
                actions_array[i] = ACTION_MAP[a_idx]

            # Step environment
            next_obs_list, rewards, done, info = env.step(actions_array)

            # Log day-level P2P info for each step
            day_logs.append({
                "step": step_count,
                "grid_import_no_p2p": info["grid_import_no_p2p"],
                "grid_import_with_p2p": info["grid_import_with_p2p"],
                "p2p_buy": info["p2p_buy"],
                "p2p_sell": info["p2p_sell"]
            })

            # Store transitions & update
            for i in range(num_agents):
                agents[i].push_transition(
                    s=obs_list[i],
                    a=chosen_actions_idx[i],
                    r=rewards[i],
                    s_next=next_obs_list[i],
                    d=float(done)
                )
                agents[i].update()

            obs_list = next_obs_list
            total_reward_agents += rewards
            step_count += 1
            total_steps_global += 1

            if step_count >= steps_per_day:
                done = True

        # Summarize day
        mean_ep_reward = float(np.mean(total_reward_agents))
        episode_rewards.append(mean_ep_reward)
        agent_rewards_log.append(total_reward_agents.copy())

        # Summarize P2P steps
        steps_data = []
        for entry in day_logs:
            step_idx = entry["step"]
            p2p_buy_array  = entry["p2p_buy"]
            p2p_sell_array = entry["p2p_sell"]
            grid_no_p2p_array   = entry["grid_import_no_p2p"]
            grid_with_p2p_array = entry["grid_import_with_p2p"]

            steps_data.append({
                "step": step_idx,
                "p2p_buy_sum":  float(np.sum(p2p_buy_array)),
                "p2p_sell_sum": float(np.sum(p2p_sell_array)),
                "grid_import_no_p2p_sum":   float(np.sum(grid_no_p2p_array)),
                "grid_import_with_p2p_sum": float(np.sum(grid_with_p2p_array))
            })

        day_df = pd.DataFrame(steps_data)
        steps_summed = day_df.groupby("step")[["p2p_buy_sum", "p2p_sell_sum"]].sum(numeric_only=True)
        valid_steps = steps_summed[(steps_summed["p2p_buy_sum"] + steps_summed["p2p_sell_sum"]) > 1].index
        day_df_trades_only = day_df[day_df["step"].isin(valid_steps)]

        p2p_day_import_no   = day_df_trades_only["grid_import_no_p2p_sum"].sum()
        p2p_day_import_with = day_df_trades_only["grid_import_with_p2p_sum"].sum()

        if p2p_day_import_no > 1e-9:
            day_savings = (p2p_day_import_no - p2p_day_import_with) / p2p_day_import_no
        else:
            day_savings = 0.0

        num_p2p_active_steps = len(valid_steps)

        print(f"Episode {episode}/{num_episodes} (Day {day_idx}) | Steps: {step_count} "
              f"| Mean Reward: {mean_ep_reward:.3f} "
              f"| P2P Savings: {day_savings:.3f} "
              f"(over {num_p2p_active_steps} P2P-active steps)")
        for i, rew_val in enumerate(total_reward_agents):
            print(f"  Agent {i} Reward: {rew_val:.3f}")

        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        episode_durations.append(episode_duration)

        # Save best model
        if mean_ep_reward > best_mean_reward:
            best_mean_reward = mean_ep_reward
            torch.save(agents, best_model_path)
            print(f"  New best mean reward: {best_mean_reward:.3f} | Model saved to {best_model_path}")

        # Checkpoint
        if (checkpoint_interval > 0) and (episode % checkpoint_interval == 0):
            ckpt_name = f"dqn_checkpoint_{episode}.pth"
            torch.save(agents, ckpt_name)
            print(f"Checkpoint saved: {ckpt_name}")

        # Record data in our per-episode log
        episode_log_data.append({
            "Episode":          episode,
            "Steps":            step_count,
            "Mean_Reward":      mean_ep_reward,
            "P2P_Savings":      float(day_savings),
            "P2P_Active_Steps": num_p2p_active_steps,
            "Episode_Duration": episode_duration
        })

   
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    # Save final agent models
    save_dir = "checkpoints_idqn"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    final_model_path = os.path.join(save_dir, "dqn_trained.pth")
    torch.save(agents, final_model_path)
    print(f"\nTraining complete. Final model saved as {final_model_path}")
    print(f"Best model during training saved as {best_model_path}")

    # Convert agent_rewards_log from list-of-arrays to array-of-arrays
    # shape: [num_episodes, num_agents]
    agent_rewards_log = np.array(agent_rewards_log)
    np.save("agent_rewards_dqn.npy", agent_rewards_log)

    # Calculate mean reward across agents each episode
    mean_rewards = np.mean(agent_rewards_log, axis=1)
    np.save("mean_rewards_dqn.npy", mean_rewards)

    # Plot
    plot_dir = "mean_reward"
    os.makedirs(plot_dir, exist_ok=True)

    existing_files = os.listdir(plot_dir)
    fig_numbers = [int(f.split("fig")[1].split(".png")[0])
                   for f in existing_files
                   if f.startswith("fig") and f.endswith(".png")]
    next_fig_number = max(fig_numbers, default=0) + 1

    plt.figure()
    plt.plot(mean_rewards, label="Mean reward per episode")
    plt.xlabel("Episodes")
    plt.ylabel("Mean Reward (All Agents)")
    plt.title("Independent DQN Training on SolarSharer")
    plt.legend()

    plot_path = os.path.join(plot_dir, f"fig{next_fig_number}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

    # Timing summaries
    episode_durations_np = np.array(episode_durations)
    avg_ep_time = np.mean(episode_durations_np)
    std_ep_time = np.std(episode_durations_np)

    print("\n=== Timing Benchmarks ===")
    print(f"Total Training Time: {total_training_time:.2f} seconds")
    print(f"Average Time per Episode: {avg_ep_time:.2f} seconds ± {std_ep_time:.2f}")
    print(f"Total Episodes: {num_episodes}")
    print(f"Total Steps (Environment Steps): {total_steps_global}")
    steps_per_second = total_steps_global / total_training_time if total_training_time > 0 else 0
    print(f"Steps per Second (overall): {steps_per_second:.2f}")

    # Save episode-level logs
    df_logs = pd.DataFrame(episode_log_data)
    df_logs.to_csv("episode_log_dqn.csv", index=False)
    print("\nEpisode-level training logs saved to 'episode_log_dqn.csv'.")

if __name__ == "__main__":
    main()

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solar_sharer_env import SolarSharer
from maddpg.trainer.maddpg import MADDPG

def main():
    num_episodes = 100  # e.g. 200 days, 300 days
    batch_size = 512
    checkpoint_interval = 50

    env = SolarSharer(
        data_path="/Users/ananygupta/Desktop/solar_trader_maddpg/data/filtered_pivoted_austin.csv",
        alpha=0.3, beta=0.5, gamma=0.5, delta=0.4,
        max_grid_price=0.2112,
        time_freq="15T",   # Example: 30-minute steps or turn on the system every 30 minutes, 15 minutes or 1 hour
        agg_method="mean"  # or "sum"
    )

   
    max_steps = env.num_steps  

    # dims from the env
    num_agents = env.num_agents
    local_state_dim = env.observation_space.shape[1]  # 6 for observation space
    action_dim = env.action_space.shape[1]            # 4 for action space

    # Create the MADDPG
    maddpg = MADDPG(
        num_agents=num_agents,
        local_state_dim=local_state_dim,
        action_dim=action_dim,
        gamma=0.95,
        tau=0.01,
        lr_actor=1e-3,
        lr_critic=1e-3,
        buffer_size=100000
    )

  
    episode_rewards = []
    agent_rewards_log = [[] for _ in range(num_agents)]
    best_mean_reward = -1e9
    best_model_path = "maddpg_best.pth"

    
    training_start_time = time.time()
    episode_durations = []
    total_steps_global = 0
    episode_log_data = []

    # Training loop
    for episode in range(1, num_episodes + 1):
        episode_start_time = time.time()

        obs = env.reset()
        total_reward = np.zeros(num_agents, dtype=np.float32)

        done = False
        step_count = 0

        day_logs = []

        while not done:
            actions = maddpg.select_actions(obs)
            next_obs, rewards, done, info = env.step(actions)

            maddpg.store_transition(obs, actions, rewards, next_obs, done)
            maddpg.train(batch_size=batch_size)

            total_reward += rewards
            obs = next_obs
            step_count += 1
            total_steps_global += 1

            day_logs.append({
                "step": step_count - 1,
                "grid_import_no_p2p": info["grid_import_no_p2p"],
                "grid_import_with_p2p": info["grid_import_with_p2p"],
                "p2p_buy": info["p2p_buy"],
                "p2p_sell": info["p2p_sell"]
            })

    
            if step_count >= max_steps:
                break

 
        mean_ep_reward = np.mean(total_reward)
        episode_rewards.append(mean_ep_reward)
        for i in range(num_agents):
            agent_rewards_log[i].append(total_reward[i])

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
        valid_steps = steps_summed[(steps_summed["p2p_buy_sum"] + steps_summed["p2p_sell_sum"]) > 0.1].index
        day_df_trades_only = day_df[day_df["step"].isin(valid_steps)]

        p2p_day_import_no = day_df_trades_only["grid_import_no_p2p_sum"].sum()
        p2p_day_import_with = day_df_trades_only["grid_import_with_p2p_sum"].sum()

        if p2p_day_import_no > 1e-9:
            day_savings = (p2p_day_import_no - p2p_day_import_with) / p2p_day_import_no
        else:
            day_savings = 0.0

        num_p2p_active_steps = len(valid_steps)
        print(f"Episode {episode}/{num_episodes} "
              f"| Steps: {step_count} "
              f"| Mean Reward: {mean_ep_reward:.3f} "
              f"| P2P Savings: {day_savings:.3f} "
              f"(over {num_p2p_active_steps} P2P-active steps)")

        for i, rew_val in enumerate(total_reward):
            print(f"  Agent {i} Reward: {rew_val:.3f}")

        maddpg.on_episode_end()

        # Save best model
        if mean_ep_reward > best_mean_reward:
            best_mean_reward = mean_ep_reward
            torch.save(maddpg, best_model_path)
            print(f"  New best mean reward: {best_mean_reward:.3f} | Model saved to {best_model_path}")

        # Checkpoints
        if episode % checkpoint_interval == 0:
            ckpt_name = f"checkpoint_{episode}.pth"
            torch.save(maddpg, ckpt_name)
            print(f"Checkpoint saved: {ckpt_name}")

        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        episode_durations.append(episode_duration)

        # Record data in our per-episode log
        episode_log_data.append({
            "Episode":          episode,
            "Steps":            step_count,
            "Mean_Reward":      float(mean_ep_reward),
            "P2P_Savings":      float(day_savings),
            "P2P_Active_Steps": num_p2p_active_steps,
            "Episode_Duration": episode_duration
        })

    # End of all training
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    np.save("agent_rewards.npy", np.array(agent_rewards_log))
    np.save("mean_rewards.npy",  np.array(episode_rewards))

    # Create a folder to save the mean-reward plot if it doesn't exist
    plot_dir = "mean_reward"
    os.makedirs(plot_dir, exist_ok=True)

    existing_files = os.listdir(plot_dir)
    fig_numbers = [int(f.split("fig")[1].split(".png")[0]) 
                   for f in existing_files 
                   if f.startswith("fig") and f.endswith(".png")]
    next_fig_number = max(fig_numbers, default=0) + 1

    # Plot the mean reward over episodes
    plt.figure()
    plt.plot(episode_rewards, label="Mean reward per episode")
    plt.xlabel("Episodes")
    plt.ylabel("Mean Reward (All Agents)")
    plt.title("MADDPG Training on SolarSharer")
    plt.legend()

    plot_path = os.path.join(plot_dir, f"fig{next_fig_number}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved to {plot_path}")

    final_model_path = "maddpg_trained.pth"
    torch.save(maddpg, final_model_path)
    print("\nTraining complete.")
    print(f"Final model saved as {final_model_path}")
    print(f"Best model during training saved as {best_model_path}")

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

    df_logs = pd.DataFrame(episode_log_data)
    df_logs.to_csv("episode_log.csv", index=False)
    print("\nEpisode-level training logs saved to 'episode_log.csv'.")


    individual_agents_dir = os.path.join(plot_dir, "individual_agents_reward")
    os.makedirs(individual_agents_dir, exist_ok=True)

    for i in range(num_agents):
        # Save CSV
        agent_csv_path = os.path.join(individual_agents_dir, f"agent_{i}_rewards.csv")
        pd.DataFrame({"Reward": agent_rewards_log[i]}).to_csv(agent_csv_path, index=False)

        # Save plot
        plt.figure()
        plt.plot(agent_rewards_log[i], label=f"Agent {i} Reward")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title(f"MADDPG Training - Agent {i}")
        plt.legend()

        agent_plot_path = os.path.join(individual_agents_dir, f"agent_{i}_fig{next_fig_number}.png")
        plt.savefig(agent_plot_path)
        plt.close()

        print(f"Agent {i} reward plot saved to {agent_plot_path}, CSV saved to {agent_csv_path}")

 
    all_agents_rewards_array = np.array(agent_rewards_log)  # shape: [num_agents, num_episodes]

    all_agents_rewards_array = all_agents_rewards_array.T    # shape: [num_episodes, num_agents]


    columns = [f"Agent_{i}_Reward" for i in range(num_agents)]
    df_all_agents_rewards = pd.DataFrame(all_agents_rewards_array, columns=columns)
    df_all_agents_rewards.insert(0, "Episode", range(1, num_episodes + 1))

    #save that CSV
    df_all_agents_rewards.to_csv("all_agents_rewards.csv", index=False)
    print("Saved per-episode rewards for all agents to 'all_agents_rewards.csv'.")

if __name__ == "__main__":
    main()

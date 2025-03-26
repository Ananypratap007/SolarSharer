import os
import sys
import numpy as np
import torch
if torch.cuda.is_available():
    print("GPU is available. Training will run on CUDA.")
else:
    print("No GPU detected. Training will run on CPU.")

import matplotlib.pyplot as plt
import pandas as pd
import time

# Ensure local modules are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solar_sharer_env import SolarSharer


from pg.trainer.pg import (
    PGActorCritic,
    PGRolloutBuffer,
    policy_gradient_update,
    flatten_obs,
    combine_rewards
)

def main():
    num_episodes = 1000      # Match MADDPG for fair comparison
    max_steps = 96          # 96 steps per day
    checkpoint_interval = 1000
    gamma = 0.95            # Matches MADDPG’s discount factor
    lr = 1e-3               

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    env = SolarSharer(
        data_path="data/filtered_pivoted_austin.csv",
        alpha=300, beta=600, gamma=500, delta=300,
        max_grid_price=0.2112
    )

    num_agents = env.num_agents
    local_state_dim = env.observation_space.shape[1]  # 6
    action_dim = env.action_space.shape[1]            #  4

   
    obs_dim = num_agents * local_state_dim
    act_dim = num_agents * action_dim

    policy = PGActorCritic(obs_dim, act_dim, hidden_size=128, init_std=0.1).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)


    episode_rewards = []
    agent_rewards_log = [[] for _ in range(num_agents)]
    best_mean_reward = -1e9
    best_model_path = "pg_best.pth"

    training_start_time = time.time()
    episode_durations = []
    total_steps_global = 0
    episode_log_data = []

    for episode in range(1, num_episodes + 1):
        episode_start_time = time.time()

        obs_list = env.reset()
        obs_flat = flatten_obs(obs_list)
        total_reward = np.zeros(num_agents, dtype=np.float32)
        done = False
        step_count = 0


        rollout_buffer = PGRolloutBuffer()
        day_logs = []

        while not done:
            
            obs_t = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)
            action_out, log_prob_out, value_out = policy.get_action(obs_t)

  
            action_np = action_out.cpu().numpy()
            action_2d = action_np.reshape(num_agents, action_dim)


            next_obs_list, rewards_list, done, info = env.step(action_2d)
            total_reward += rewards_list

            
            reward_scalar = combine_rewards(rewards_list)

            # Store the transition
            rollout_buffer.store(
                obs_flat,
                action_np,
                log_prob_out,
                value_out,
                reward_scalar,
                done
            )

            # For logging P2P details
            day_logs.append({
                "step": step_count,
                "grid_import_no_p2p": info["grid_import_no_p2p"],
                "grid_import_with_p2p": info["grid_import_with_p2p"],
                "p2p_buy": info["p2p_buy"],
                "p2p_sell": info["p2p_sell"]
            })

            # Move on
            obs_list = next_obs_list
            if not done:
                obs_flat = flatten_obs(obs_list)

            step_count += 1
            total_steps_global += 1

            # End episode early if max_steps is reached
            if step_count >= max_steps:
                break

        # End of episode stats
        mean_ep_reward = float(np.mean(total_reward))
        episode_rewards.append(mean_ep_reward)
        for i in range(num_agents):
            agent_rewards_log[i].append(total_reward[i])

        # Summarize P2P steps 
        steps_data = []
        for entry in day_logs:
            step_idx = entry["step"]
            steps_data.append({
                "step": step_idx,
                "p2p_buy_sum":  float(np.sum(entry["p2p_buy"])),
                "p2p_sell_sum": float(np.sum(entry["p2p_sell"])),
                "grid_import_no_p2p_sum":   float(np.sum(entry["grid_import_no_p2p"])),
                "grid_import_with_p2p_sum": float(np.sum(entry["grid_import_with_p2p"]))
            })

        day_df = pd.DataFrame(steps_data)
        steps_summed = day_df.groupby("step")[["p2p_buy_sum", "p2p_sell_sum"]].sum(numeric_only=True)
        valid_steps = steps_summed[(steps_summed["p2p_buy_sum"] + steps_summed["p2p_sell_sum"]) > 0.2].index
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

       
        obs_t, act_t, lp_t, val_t, rew_t, done_t = rollout_buffer.as_tensors(device)
        policy_gradient_update(
            policy,
            optimizer,
            obs=obs_t,
            actions=act_t,
            log_probs_old=lp_t,
            values_old=val_t,
            rewards=rew_t,
            dones=done_t,
            gamma=gamma,
            device=device
        )
      

        # Save best model
        if mean_ep_reward > best_mean_reward:
            best_mean_reward = mean_ep_reward
            torch.save(policy.state_dict(), best_model_path)
            print(f"  New best mean reward: {best_mean_reward:.3f} | Model saved to {best_model_path}")

        # Periodic checkpoints
        if checkpoint_interval > 0 and (episode % checkpoint_interval == 0):
            ckpt_name = f"pg_checkpoint_{episode}.pth"
            torch.save(policy.state_dict(), ckpt_name)
            print(f"Checkpoint saved: {ckpt_name}")

        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        episode_durations.append(episode_duration)

        episode_log_data.append({
            "Episode":          episode,
            "Steps":            step_count,
            "Mean_Reward":      mean_ep_reward,
            "P2P_Savings":      float(day_savings),
            "P2P_Active_Steps": num_p2p_active_steps,
            "Episode_Duration": episode_duration
        })

    # End of training
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    # Save logs
    agent_rewards_log = np.array(agent_rewards_log)
    np.save("agent_rewards_pg.npy", agent_rewards_log)
    np.save("mean_rewards_pg.npy",  np.array(episode_rewards))

    # Create plot dir if needed
    plot_dir = "mean_reward"
    os.makedirs(plot_dir, exist_ok=True)
    existing_files = os.listdir(plot_dir)
    fig_numbers = [
        int(f.split("fig")[1].split(".png")[0])
        for f in existing_files
        if f.startswith("fig") and f.endswith(".png")
    ]
    next_fig_number = max(fig_numbers, default=0) + 1

    # Plot
    plt.figure()
    plt.plot(episode_rewards, label="Mean reward per episode")
    plt.xlabel("Episodes")
    plt.ylabel("Mean Reward (All Agents)")
    plt.title("Policy Gradient Training on SolarSharer")
    plt.legend()

    plot_path = os.path.join(plot_dir, f"fig{next_fig_number}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

    final_model_path = "pg_trained.pth"
    torch.save(policy.state_dict(), final_model_path)
    print("\nTraining complete.")
    print(f"Final PG model saved as {final_model_path}")
    print(f"Best PG model during training saved as {best_model_path}")

    # Timing benchmarks
    episode_durations_np = np.array(episode_durations)
    avg_ep_time = np.mean(episode_durations_np)
    std_ep_time = np.std(episode_durations_np)

    print("\n=== Timing Benchmarks ===")
    print(f"Total Training Time: {total_training_time:.2f} seconds")
    print(f"Average Time per Episode: {avg_ep_time:.2f} seconds ± {std_ep_time:.2f}")
    print(f"Total Episodes: {num_episodes}")
    print(f"Total Steps (Environment Steps): {total_steps_global}")
    steps_per_second = (
        total_steps_global / total_training_time if total_training_time > 0 else 0
    )
    print(f"Steps per Second (overall): {steps_per_second:.2f}")

    df_logs = pd.DataFrame(episode_log_data)
    df_logs.to_csv("episode_log_pg.csv", index=False)
    print("\nEpisode-level training logs saved to 'episode_log_pg.csv'.")


if __name__ == "__main__":
    main()

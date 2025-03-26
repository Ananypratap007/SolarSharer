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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solar_sharer_env import SolarSharer


from ppo.trainer.ppo import (
    PPOActorCritic,
    PPORolloutBuffer,
    ppo_update,
    flatten_obs,
    combine_rewards
)

def main():

    
    num_episodes = 1000       
    max_steps = 96            # each day has 96 steps
    batch_size = 256
    checkpoint_interval = 1000

    # For PPO
    gamma = 0.99
    lam   = 0.95
    clip_epsilon = 0.2
    ppo_epochs = 10
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = SolarSharer(
        data_path="data/filtered_pivoted_austin.csv",
        alpha=0.3, beta=0.5, gamma=0.5, delta=0.4,
        max_grid_price=0.2112
    )
    num_agents = env.num_agents
    local_state_dim = env.observation_space.shape[1]  # 6
    action_dim = env.action_space.shape[1]            # 4

    obs_dim = num_agents * local_state_dim
    act_dim = num_agents * action_dim


    policy = PPOActorCritic(obs_dim, act_dim, hidden_size=128, init_std=0.1).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    episode_rewards = []                
    agent_rewards_log = [[] for _ in range(num_agents)]  
    best_mean_reward = -1e9
    best_model_path = "ppo_best.pth"

    training_start_time = time.time()
    episode_durations = []
    total_steps_global = 0
    episode_log_data = []

    for episode in range(1, num_episodes + 1):
        episode_start_time = time.time()

        obs_list = env.reset()  # shape: (num_agents, 6)
        obs_flat = flatten_obs(obs_list)
        total_reward = np.zeros(num_agents, dtype=np.float32)
        done = False
        step_count = 0


        rollout_buffer = PPORolloutBuffer()


        day_logs = []

        while not done:
            obs_t = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)  # [1, obs_dim]

       
            with torch.no_grad():
                action_out, log_prob_out, value_out = policy.get_action(obs_t)
                # action_out shape [act_dim]

            # Reshape for env
            action_np = action_out.cpu().numpy()
            action_2d = action_np.reshape(num_agents, action_dim)

            # Step environment
            next_obs_list, rewards_list, done, info = env.step(action_2d)

            # Collect agent rewards
            total_reward += rewards_list

            # Convert multi-agent reward to single scalar for PPO
            reward_scalar = combine_rewards(rewards_list)

            # Store in rollout buffer
            rollout_buffer.store(
                obs_flat,
                action_np,
                log_prob_out.item(),
                value_out.item(),
                reward_scalar,
                done
            )

        
            day_logs.append({
                "step": step_count,
                "grid_import_no_p2p": info["grid_import_no_p2p"],
                "grid_import_with_p2p": info["grid_import_with_p2p"],
                "p2p_buy": info["p2p_buy"],
                "p2p_sell": info["p2p_sell"]
            })

            obs_list = next_obs_list
            if not done:
                obs_flat = flatten_obs(obs_list)

            step_count += 1
            total_steps_global += 1

            if step_count >= max_steps:
                break  # end of day

     
        mean_ep_reward = float(np.mean(total_reward))
        episode_rewards.append(mean_ep_reward)
        for i in range(num_agents):
            agent_rewards_log[i].append(total_reward[i])

        # Summarize day-level P2P
        steps_data = []
        for entry in day_logs:
            step_idx = entry["step"]
            p2p_buy_arr  = entry["p2p_buy"]
            p2p_sell_arr = entry["p2p_sell"]
            g_no_p2p_arr = entry["grid_import_no_p2p"]
            g_with_p2p_arr = entry["grid_import_with_p2p"]
            steps_data.append({
                "step": step_idx,
                "p2p_buy_sum":  float(np.sum(p2p_buy_arr)),
                "p2p_sell_sum": float(np.sum(p2p_sell_arr)),
                "grid_import_no_p2p_sum":   float(np.sum(g_no_p2p_arr)),
                "grid_import_with_p2p_sum": float(np.sum(g_with_p2p_arr))
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
        print(f"Episode {episode}/{num_episodes} "
              f"| Steps: {step_count} "
              f"| Mean Reward: {mean_ep_reward:.3f} "
              f"| P2P Savings: {day_savings:.3f} "
              f"(over {num_p2p_active_steps} P2P-active steps)")

        for i, rew_val in enumerate(total_reward):
            print(f"  Agent {i} Reward: {rew_val:.3f}")

    
        obs_t, act_t, lp_t, val_t, rew_t, done_t = rollout_buffer.as_tensors(device)
        ppo_update(
            policy,
            optimizer,
            obs=obs_t,
            actions=act_t,
            log_probs_old=lp_t,
            values_old=val_t,
            rewards=rew_t,
            dones=done_t,
            gamma=gamma,
            lam=lam,
            clip_epsilon=clip_epsilon,
            ppo_epochs=ppo_epochs,
            batch_size=batch_size,
            device=device
        )


   
        if mean_ep_reward > best_mean_reward:
            best_mean_reward = mean_ep_reward
            torch.save(policy.state_dict(), best_model_path)
            print(f"  New best mean reward: {best_mean_reward:.3f} | Model saved to {best_model_path}")


        if episode % checkpoint_interval == 0:
            ckpt_name = f"ppo_checkpoint_{episode}.pth"
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


    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    # Save logs
    agent_rewards_log = np.array(agent_rewards_log)  # shape [num_agents, num_episodes]
    np.save("agent_rewards_ppo.npy", agent_rewards_log)
    np.save("mean_rewards_ppo.npy",  np.array(episode_rewards))

    # Create folder for plots
    plot_dir = "mean_reward"
    os.makedirs(plot_dir, exist_ok=True)

    existing_files = os.listdir(plot_dir)
    fig_numbers = [
        int(f.split("fig")[1].split(".png")[0])
        for f in existing_files
        if f.startswith("fig") and f.endswith(".png")
    ]
    next_fig_number = max(fig_numbers, default=0) + 1

    plt.figure()
    plt.plot(episode_rewards, label="Mean reward per episode")
    plt.xlabel("Episodes")
    plt.ylabel("Mean Reward (All Agents)")
    plt.title("PPO Training on SolarSharer")
    plt.legend()

    plot_path = os.path.join(plot_dir, f"fig{next_fig_number}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

    final_model_path = "ppo_trained.pth"
    torch.save(policy.state_dict(), final_model_path)
    print("\nTraining complete.")
    print(f"Final PPO model saved as {final_model_path}")
    print(f"Best PPO model during training saved as {best_model_path}")

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
    df_logs.to_csv("episode_log_ppo.csv", index=False)
    print("\nEpisode-level training logs saved to 'episode_log_ppo.csv'.")


if __name__ == "__main__":
    main()

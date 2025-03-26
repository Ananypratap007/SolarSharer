import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solar_sharer_env import SolarSharer
from independent_dqn_agent import IndependentDQNAgent
from actions import ACTION_MAP 


def compute_jains_fairness(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    if np.all(values == 0):
        return 1.0
    numerator = (values.sum()) ** 2
    denominator = len(values) * (values**2).sum()
    return numerator / denominator


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DQN Evaluation will run on device: {device}")

  
    agent_checkpoints_dir = "checkpoints_idqn"  
    data_path             = "data/filtered_pivoted_austin.csv"

    alpha, beta     = 0.4, 3.0
    gamma_, delta   = 3.0, 2.0
    max_grid_price  = 0.2112
    days_to_evaluate = 30  

    # This folder will store all logs and plots for DQN evaluation
    output_folder = "dqn_evaluation_current"
    os.makedirs(output_folder, exist_ok=True)

    # 1) Create environment
    env = SolarSharer(
        data_path=data_path,
        alpha=alpha,
        beta=beta,
        gamma=gamma_,
        delta=delta,
        max_grid_price=max_grid_price
    )
    house_ids = env.house_ids
    num_agents = env.num_agents

    
    obs_dim = 6
    act_dim = 4

    # 2) Load each agent from disk
    agents = []
    for i in range(num_agents):
        agent = IndependentDQNAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            lr=1e-3,
            gamma=0.99,
            tau=0.01,
            buffer_capacity=5000,
            batch_size=32,
            eps_start=0.0, 
            eps_end=0.0,
            eps_decay=1.0,
            device=str(device)
        )
        # Load Q-network checkpoint
        model_path = os.path.join(agent_checkpoints_dir, f"agent_{i}_dqn.pth")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Could not find checkpoint for Agent {i}: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        agent.q_net.load_state_dict(state_dict)
        agent.q_net.eval()
        agent.eps = 0.0 
        agents.append(agent)

    evaluation_start = time.time()

    all_logs = []         # step-level logs (for all days)
    daily_summaries = []  # day-level aggregated stats
    step_timing_records = []

    total_no_p2p_import_sum   = 0.0
    total_with_p2p_import_sum = 0.0

    # Evaluate day by day
    days_to_run = min(days_to_evaluate, env.total_days)
    for day_idx in range(days_to_run):
        env.day_index = day_idx
        obs_list = env.reset()

        done = False
        step_count = 0
        day_logs = []
        max_steps = 96  # each day has 96 steps

        while not done:
            step_start_time = time.time()

            # Greedy action per agent
            chosen_actions = np.zeros((num_agents, act_dim), dtype=np.float32)
            for i in range(num_agents):
                obs_i = obs_list[i]
                obs_t = torch.FloatTensor(obs_i).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_vals = agents[i].q_net(obs_t)  # shape [1, act_dim]
                    best_action_idx = q_vals.argmax(dim=1).item()
                chosen_actions[i] = ACTION_MAP[best_action_idx]

            next_obs_list, rewards, done, info = env.step(chosen_actions)

            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            print(f"[DQN Eval | Day {day_idx+1}, Step {step_count}] Step time: {step_duration:.6f} s")

            step_timing_records.append({
                "day":        day_idx + 1,
                "step":       step_count,
                "step_time_s": step_duration
            })

            # Log step-level data for each house
            for i, hid in enumerate(house_ids):
                grid_import_no   = float(info["grid_import_no_p2p"][i])
                grid_import_with = float(info["grid_import_with_p2p"][i])
                grid_export      = float(info["grid_export"][i])
                p2p_buy          = float(info["p2p_buy"][i])
                p2p_sell         = float(info["p2p_sell"][i])
                cost_i           = float(info["costs"][i])
                reward_i         = float(rewards[i])

                # For reference
                house_demand = float(env.demands[hid][step_count])
                house_solar  = float(env.solars[hid][step_count])
                grid_price_now = env.get_grid_price(step_count)
                baseline_cost  = grid_import_no * grid_price_now

                day_logs.append({
                    "day":               day_idx + 1,
                    "step":              step_count,
                    "house":             hid,
                    "grid_import_no_p2p":   grid_import_no,
                    "grid_import_with_p2p": grid_import_with,
                    "grid_export":          grid_export,
                    "p2p_buy":             p2p_buy,
                    "p2p_sell":            p2p_sell,
                    "actual_cost":         cost_i,
                    "reward":              reward_i,
                    "total_demand":        house_demand,
                    "total_solar":         house_solar,
                    "baseline_cost":       baseline_cost
                })

            obs_list = next_obs_list
            step_count += 1
            if step_count >= max_steps:
                break

        # Summarize day
        day_df = pd.DataFrame(day_logs)

    
        steps_summed = day_df.groupby("step")[["p2p_buy", "p2p_sell"]].sum(numeric_only=True)
        valid_steps  = steps_summed[(steps_summed["p2p_buy"] + steps_summed["p2p_sell"]) > 3].index
        day_df_trades_only = day_df[day_df["step"].isin(valid_steps)]

        grouped_by_step_trades_only = day_df_trades_only.groupby("step").sum(numeric_only=True)
        day_no_p2p   = grouped_by_step_trades_only["grid_import_no_p2p"].sum()
        day_with_p2p = grouped_by_step_trades_only["grid_import_with_p2p"].sum()

        total_no_p2p_import_sum   += day_no_p2p
        total_with_p2p_import_sum += day_with_p2p

        if day_no_p2p > 1e-9:
            day_savings_ratio = (day_no_p2p - day_with_p2p) / day_no_p2p
        else:
            day_savings_ratio = 0.0
        
        grouped_by_step  = day_df.groupby("step").sum(numeric_only=True)
        grouped_by_house = day_df.groupby("house").sum(numeric_only=True)

        total_demand_day   = grouped_by_step["total_demand"].sum()   if not grouped_by_step.empty  else 0.0
        total_solar_day    = grouped_by_step["total_solar"].sum()    if not grouped_by_step.empty  else 0.0
        baseline_cost_day  = grouped_by_house["baseline_cost"].sum() if not grouped_by_house.empty else 0.0
        total_cost_day     = grouped_by_house["actual_cost"].sum()   if not grouped_by_house.empty else 0.0
        cost_savings_day   = baseline_cost_day - total_cost_day

        cost_array    = grouped_by_house["actual_cost"].values if not grouped_by_house.empty else []
        reward_array  = grouped_by_house["reward"].values      if not grouped_by_house.empty else []
        p2p_buy_array = grouped_by_house["p2p_buy"].values     if not grouped_by_house.empty else []
        p2p_sell_array= grouped_by_house["p2p_sell"].values    if not grouped_by_house.empty else []

        def fair(x):
            return compute_jains_fairness(x) if len(x) > 0 else 0.0

        fairness_cost     = fair(cost_array)
        fairness_reward   = fair(reward_array)
        fairness_p2p_buy  = fair(p2p_buy_array)
        fairness_p2p_sell = fair(p2p_sell_array)
        p2p_total_array   = p2p_buy_array + p2p_sell_array
        fairness_p2p_total= fair(p2p_total_array)

        day_p2p_buy  = p2p_buy_array.sum()
        day_p2p_sell = p2p_sell_array.sum()

        daily_summaries.append({
            "day":                day_idx + 1,
            "day_savings_ratio":  day_savings_ratio,
            "day_cost_savings":   cost_savings_day,
            "day_total_demand":   total_demand_day,
            "day_total_solar":    total_solar_day,
            "fairness_cost":      fairness_cost,
            "fairness_reward":    fairness_reward,
            "fairness_p2p_buy":   fairness_p2p_buy,
            "fairness_p2p_sell":  fairness_p2p_sell,
            "fairness_p2p_total": fairness_p2p_total,
            "day_p2p_traded":     day_with_p2p,
            "day_no_p2p":         day_no_p2p,
            "day_p2p_buy":        day_p2p_buy,
            "day_p2p_sell":       day_p2p_sell
        })

        all_logs.extend(day_logs)

    all_days_df = pd.DataFrame(all_logs)
    combined_csv_path = os.path.join(output_folder, "dqn_evaluation_all_days_combined.csv")
    all_days_df.to_csv(combined_csv_path, index=False)
    print(f"\nSaved combined step-level logs for ALL days to: {combined_csv_path}")


    step_timing_df = pd.DataFrame(step_timing_records)
    step_timing_csv = os.path.join(output_folder, "dqn_step_timing_log.csv")
    step_timing_df.to_csv(step_timing_csv, index=False)
    print(f"Saved step timing logs to: {step_timing_csv}")

    eval_end = time.time()
    total_eval_time = eval_end - evaluation_start

    # Multi-day overall P2P savings
    if total_no_p2p_import_sum > 1e-9:
        multi_day_savings_ratio = (
            total_no_p2p_import_sum - total_with_p2p_import_sum
        ) / total_no_p2p_import_sum
    else:
        multi_day_savings_ratio = 0.0

    daily_summary_df = pd.DataFrame(daily_summaries)

    # Final row for summary
    final_row = {
        "day":               "ALL",
        "day_savings_ratio": multi_day_savings_ratio,
        "day_cost_savings":  daily_summary_df["day_cost_savings"].sum(),
        "day_total_demand":  np.nan,
        "day_total_solar":   np.nan,
        "fairness_cost":     np.nan,
        "fairness_reward":   np.nan,
        "fairness_p2p_buy":  np.nan,
        "fairness_p2p_sell": np.nan,
        "fairness_p2p_total": np.nan,
        "day_p2p_traded":    daily_summary_df["day_p2p_traded"].sum(),
        "day_no_p2p":        daily_summary_df["day_no_p2p"].sum(),
        "day_p2p_buy":       daily_summary_df["day_p2p_buy"].sum(),
        "day_p2p_sell":      daily_summary_df["day_p2p_sell"].sum()
    }
    final_row_df = pd.DataFrame([final_row])
    daily_summary_df = pd.concat([daily_summary_df, final_row_df], ignore_index=True)

    summary_csv = os.path.join(output_folder, "dqn_evaluation_all_days_summary.csv")
    daily_summary_df.to_csv(summary_csv, index=False)
    print(f"Saved DQN day-level summary + final multi-day row to: {summary_csv}")
    print("\n=========================================================")
    print(f"DQN Evaluation finished for {days_to_run} days.")
    print(f"Multi-day overall savings ratio: {multi_day_savings_ratio:.3f}")
    print(f"Total evaluation time: {total_eval_time:.2f} seconds")

    # House-level summary
    house_level_df = all_days_df.groupby("house").agg({
        "baseline_cost": "sum",
        "actual_cost": "sum"
    })
    house_level_df["cost_savings"] = house_level_df["baseline_cost"] - house_level_df["actual_cost"]
    house_summary_csv = os.path.join(output_folder, "dqn_evaluation_house_level_summary.csv")
    house_level_df.to_csv(house_summary_csv)
    print(f"Saved DQN final cost savings per house to: {house_summary_csv}")

    # Generate plots
    # 1) Daily savings ratio
    plt.figure()
    plt.bar(daily_summary_df["day"].astype(str), daily_summary_df["day_savings_ratio"])
    plt.xlabel("Day")
    plt.ylabel("Daily Energy Savings Ratio")
    plt.title("Daily Energy Savings Ratio Across All Days (DQN)")
    plt.savefig(os.path.join(output_folder, "dqn_plot_daily_savings_ratio.png"))
    plt.close()

    # Exclude "ALL" row for some plots
    daily_data_only = daily_summary_df[daily_summary_df["day"] != "ALL"]

    # 2) Demand vs Solar
    x_vals = daily_data_only["day"].astype(int)
    plt.figure()
    plt.bar(x_vals - 0.2, daily_data_only["day_total_demand"], width=0.4, label="Total Demand")
    plt.bar(x_vals + 0.2, daily_data_only["day_total_solar"],  width=0.4, label="Total Solar")
    plt.xlabel("Day")
    plt.ylabel("kWh (Sum Over the Day)")
    plt.title("Total Demand vs. Total Solar Per Day (DQN)")
    plt.legend()
    plt.savefig(os.path.join(output_folder, "dqn_plot_demand_vs_solar_per_day.png"))
    plt.close()

    # Summation per step across all days
    step_group = all_days_df.groupby(["day", "step"]).sum(numeric_only=True).reset_index()
    step_group["global_step"] = (step_group["day"] - 1) * 96 + step_group["step"]

    # 3) Grid Import vs P2P Buy
    plt.figure()
    plt.plot(step_group["global_step"], step_group["grid_import_with_p2p"], label="Grid Import (with P2P)")
    plt.plot(step_group["global_step"], step_group["p2p_buy"], label="P2P Buy")
    plt.xlabel("Global Step (day concatenated)")
    plt.ylabel("kWh (sum over all houses)")
    plt.title("Grid Import vs. P2P Buy (All Days Combined, DQN)")
    plt.legend()
    plt.savefig(os.path.join(output_folder, "dqn_plot_combined_import_vs_p2p.png"))
    plt.close()

    # 4) Grid Export vs P2P Sell
    plt.figure()
    plt.plot(step_group["global_step"], step_group["grid_export"], label="Grid Export")
    plt.plot(step_group["global_step"], step_group["p2p_sell"], label="P2P Sell")
    plt.xlabel("Global Step (day concatenated)")
    plt.ylabel("kWh (sum over all houses)")
    plt.title("Grid Export vs. P2P Sell (All Days Combined, DQN)")
    plt.legend()
    plt.savefig(os.path.join(output_folder, "dqn_plot_combined_export_vs_p2p.png"))
    plt.close()

    # 5) day_no_p2p vs day_p2p_traded
    plt.figure()
    x_vals = daily_summary_df["day"].astype(str)
    plt.bar(x_vals, daily_summary_df["day_no_p2p"], label="Grid Import No P2P")
    plt.bar(
        x_vals,
        daily_summary_df["day_p2p_traded"],
        bottom=daily_summary_df["day_no_p2p"],
        label="P2P Traded"
    )
    plt.xlabel("Day")
    plt.ylabel("kWh")
    plt.title("Daily Grid Import No P2P vs. P2P Traded (DQN, Steps w/ P2P Activity Only)")
    plt.legend()
    plt.savefig(os.path.join(output_folder, "dqn_plot_filtered_no_p2p_and_p2p_traded.png"))
    plt.close()

    print("All combined DQN evaluation plots saved. Evaluation complete.")


if __name__ == "__main__":
    main()

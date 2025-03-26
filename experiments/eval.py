import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

############## NOTE ################

# please try to run this and confirm you are getting all the files in the specific folder 
# I have used similar code for evaluation of other alogirthms as well so make sure you run each one individually
# also I check the folder and store and copy because this will overwrite the same folders if you just keep running 
# the code again and again and again with different data.


# Make sure you can import the local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solar_sharer_env import SolarSharer
from maddpg.trainer.maddpg import MADDPG #you might not need this

#please change this depending on the directory structure and path of the model

def compute_jains_fairness(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    if np.all(values == 0):
        return 1.0
    numerator = (values.sum()) ** 2
    denominator = len(values) * (values**2).sum()
    return numerator / denominator

def main():
    model_path       = "maddpg_best.pth"
    data_path        = "/Users/ananygupta/Desktop/solar_trader_maddpg/data/filtered_pivoted_austin.csv"
    alpha, beta      = 0.3, 0.5
    gamma, delta     = 0.5, 0.4

    max_grid_price   = 0.2112
    days_to_evaluate = 30

    output_folder    = "evaluation_current"
    os.makedirs(output_folder, exist_ok=True)


    # different granularity, e.g. time_freq="30T". 
    # If you leave them at defaults, it remains 15T and 96 steps/day.
    env = SolarSharer(
        data_path=data_path,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        max_grid_price=max_grid_price,
        # time_freq="30T",
        # agg_method="mean"
    )


    eval_num_steps = env.num_steps

    house_ids = env.house_ids
    num_agents = env.num_agents

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Could not find trained model at: {model_path}")
    maddpg = torch.load(model_path, weights_only=False)

    for agent in maddpg.agents:
        agent.actor.eval()
    if hasattr(maddpg, "current_noise"):
        maddpg.current_noise = 0.0

    evaluation_start = time.time()
    all_logs = []
    daily_summaries = []

    total_no_p2p_import_sum   = 0.0
    total_with_p2p_import_sum = 0.0

    # ============ We'll store step timing in a separate list ==============
    step_timing_records = []

    for day_idx in range(days_to_evaluate):
        obs = env.reset()
        done = False
        step_count = 0
        day_logs = []

        while not done:
            step_start_time = time.time()

            actions = maddpg.select_actions(obs)
            next_obs, rewards, done, info = env.step(actions)

            step_end_time   = time.time()
            step_duration   = step_end_time - step_start_time

            print(f"[Day {day_idx+1}, Step {step_count}] Step time: {step_duration:.6f} seconds")

            step_timing_records.append({
                "day":        day_idx + 1,
                "step":       step_count,
                "step_time_s": step_duration
            })

            for i, hid in enumerate(house_ids):
                grid_import_no   = float(info["grid_import_no_p2p"][i])
                grid_import_with = float(info["grid_import_with_p2p"][i])
                grid_export      = float(info["grid_export"][i])
                p2p_buy          = float(info["p2p_buy"][i])
                p2p_sell         = float(info["p2p_sell"][i])
                cost_i           = float(info["costs"][i])
                reward_i         = float(rewards[i])

                house_demand    = float(env.demands[hid][step_count])
                house_solar     = float(env.solars[hid][step_count])
                grid_price_now  = env.get_grid_price(step_count)
                baseline_cost   = grid_import_no * grid_price_now

                day_logs.append({
                    "day":               day_idx + 1,
                    "step":              step_count,
                    "step_time_s":       step_duration,
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

            obs = next_obs
            step_count += 1

            
            if step_count >= eval_num_steps:
                break

        day_df = pd.DataFrame(day_logs)


        steps_summed = day_df.groupby("step")[["p2p_buy", "p2p_sell"]].sum(numeric_only=True)
        valid_steps  = steps_summed[(steps_summed["p2p_buy"] + steps_summed["p2p_sell"]) > 8].index
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

        # For other metrics, consider the full day_df
        grouped_by_step  = day_df.groupby("step").sum(numeric_only=True)
        grouped_by_house = day_df.groupby("house").sum(numeric_only=True)

        total_demand_day   = grouped_by_step["total_demand"].sum()
        total_solar_day    = grouped_by_step["total_solar"].sum()
        baseline_cost_day  = grouped_by_house["baseline_cost"].sum()
        total_cost_day     = grouped_by_house["actual_cost"].sum()
        cost_savings_day   = baseline_cost_day - total_cost_day

        cost_array    = grouped_by_house["actual_cost"].values
        reward_array  = grouped_by_house["reward"].values
        p2p_buy_array = grouped_by_house["p2p_buy"].values
        p2p_sell_array= grouped_by_house["p2p_sell"].values

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

    # Combine logs for all days
    all_days_df = pd.DataFrame(all_logs)
    combined_csv_path = os.path.join(output_folder, "evaluation_all_days_combined.csv")
    all_days_df.to_csv(combined_csv_path, index=False)
    print(f"\nSaved combined step-level logs for ALL days to: {combined_csv_path}")

    # Save step timings to a separate CSV
    step_timing_df = pd.DataFrame(step_timing_records)
    step_timing_csv = os.path.join(output_folder, "step_timing_log.csv")
    step_timing_df.to_csv(step_timing_csv, index=False)
    print(f"Saved step timing logs to: {step_timing_csv}")

    eval_end = time.time()
    total_eval_time = eval_end - evaluation_start

    if total_no_p2p_import_sum > 1e-9:
        multi_day_savings_ratio = (
            (total_no_p2p_import_sum - total_with_p2p_import_sum)
            / total_no_p2p_import_sum
        )
    else:
        multi_day_savings_ratio = 0.0

    daily_summary_df = pd.DataFrame(daily_summaries)

    # please modify this to include relevant data you might need, I just needed these
    # you might have to make similar changes in the train.py and environment file as well because some of these come 
    # directly from there.
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

    summary_csv = os.path.join(output_folder, "evaluation_all_days_summary.csv")
    daily_summary_df.to_csv(summary_csv, index=False)
    print(f"Saved day-level summary + final multi-day row to: {summary_csv}")
    print("\n=========================================================")
    print(f"Evaluation finished for {days_to_evaluate} days.")
    print(f"Multi-day overall savings ratio: {multi_day_savings_ratio:.3f}")
    print(f"Total evaluation time: {total_eval_time:.2f} seconds")

    # House-level summary
    house_level_df = all_days_df.groupby("house").agg({
        "baseline_cost": "sum",
        "actual_cost": "sum"
    })
    house_level_df["cost_savings"] = house_level_df["baseline_cost"] - house_level_df["actual_cost"]
    house_summary_csv = os.path.join(output_folder, "evaluation_house_level_summary.csv")
    house_level_df.to_csv(house_summary_csv)
    print(f"Saved final cost savings per house to: {house_summary_csv}")

    # Existing Plots
    plt.figure()
    plt.bar(daily_summary_df["day"].astype(str), daily_summary_df["day_savings_ratio"])
    plt.xlabel("Day")
    plt.ylabel("Daily Energy Savings Ratio")
    plt.title("Daily Energy Savings Ratio Across All Days")
    plt.savefig(os.path.join(output_folder, "plot_daily_savings_ratio.png"))
    plt.close()

    # Exclude the "ALL" row for some plots
    daily_data_only = daily_summary_df[daily_summary_df["day"] != "ALL"]

    x_vals = daily_data_only["day"].astype(int)
    plt.figure()
    plt.bar(x_vals - 0.2, daily_data_only["day_total_demand"], width=0.4, label="Total Demand")
    plt.bar(x_vals + 0.2, daily_data_only["day_total_solar"], width=0.4, label="Total Solar")
    plt.xlabel("Day")
    plt.ylabel("kWh (Sum Over the Day)")
    plt.title("Total Demand vs. Total Solar Per Day")
    plt.legend()
    plt.savefig(os.path.join(output_folder, "plot_demand_vs_solar_per_day.png"))
    plt.close()

    step_group = all_days_df.groupby(["day", "step"]).sum(numeric_only=True).reset_index()


    step_group["global_step"] = (step_group["day"] - 1) * eval_num_steps + step_group["step"]

    plt.figure()
    plt.plot(step_group["global_step"], step_group["grid_import_with_p2p"], label="Grid Import (with P2P)")
    plt.plot(step_group["global_step"], step_group["p2p_buy"], label="P2P Buy")
    plt.xlabel("Global Step (day concatenated)")
    plt.ylabel("kWh (sum over all houses)")
    plt.title("Grid Import vs. P2P Buy (All Days Combined)")
    plt.legend()
    plt.savefig(os.path.join(output_folder, "plot_combined_import_vs_p2p.png"))
    plt.close()

    plt.figure()
    plt.plot(step_group["global_step"], step_group["grid_export"], label="Grid Export")
    plt.plot(step_group["global_step"], step_group["p2p_sell"], label="P2P Sell")
    plt.xlabel("Global Step (day concatenated)")
    plt.ylabel("kWh (sum over all houses)")
    plt.title("Grid Export vs. P2P Sell (All Days Combined)")
    plt.legend()
    plt.savefig(os.path.join(output_folder, "plot_combined_export_vs_p2p.png"))
    plt.close()

    plt.figure()
    x_vals = daily_summary_df["day"].astype(str)
    plt.bar(x_vals, daily_summary_df["day_no_p2p"], label="Grid Import No P2P")
    plt.bar(x_vals,
            daily_summary_df["day_p2p_traded"],
            bottom=daily_summary_df["day_no_p2p"],
            label="P2P Traded")
    plt.xlabel("Day")
    plt.ylabel("kWh")
    plt.title("Daily Grid Import No P2P and P2P Traded (Steps with P2P activity only)")
    plt.legend()
    plt.savefig(os.path.join(output_folder, "plot_filtered_no_p2p_and_p2p_traded.png"))
    plt.close()

    print("All combined plots saved. Evaluation complete.")

if __name__ == "__main__":
    main()

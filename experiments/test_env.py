

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solar_sharer_env import SolarSharer

def main():
    env = SolarSharer(
        data_path="data/filtered_pivoted_austin.csv",
        alpha=1.0,
        beta=1.0,
        gamma=0.5,
        delta=0.3,
        max_grid_price=0.2112
    )
    num_agents = env.num_agents
    print(f"\nInitialized SolarSharer with {num_agents} agents.")

    obs = env.reset()
    print("\nAfter reset:")
    for i, obs_i in enumerate(obs):
        print(f"  Agent {i} initial observation: {obs_i}")

    n_steps = 5
    print(f"\nTaking {n_steps} random steps...")
    for step in range(n_steps):
      
        random_actions = np.random.rand(num_agents, 4).astype(np.float32)
        next_obs, rewards, done, info = env.step(random_actions)

        print(f"\nStep {step + 1}:")
        print(f"  Actions taken (shape={random_actions.shape}):\n    {random_actions}")
        for i, (obs_i, r_i) in enumerate(zip(next_obs, rewards)):
            print(f"  Agent {i} - Next Obs: {obs_i}, Reward: {r_i:.3f}")
        print(f"  Done: {done}")
        print(f"  Info: {info}")
       
        if done:
            print("\nEpisode finished early (done=True). Resetting environment...\n")
            obs = env.reset()
            continue
        else:
            obs = next_obs
 
    env.save_log(filename="test_env_log.csv")
    print("\nEnvironment log saved to 'test_env_log.csv'.")

    print("\nTest run complete!")

if __name__ == "__main__":
    main()
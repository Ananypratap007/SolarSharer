import gym
import pandas as pd
import numpy as np

class SolarSharer(gym.Env):

    def __init__(
        self,
        data_path="/Users/ananygupta/Desktop/solar_trader_maddpg/data/filtered_pivoted_austin.csv",
        alpha=0.3,       # penalty on grid usage
        beta=0.5,        # reward for selling to peers
        gamma=0.5,       # reward for buying from peers
        delta=0.4,       # fairness weighting
        max_grid_price=0.2112,  # for Oklahoma (example)
        
        # these weights act as a scalling factor, the reward fucntion is already normalized
        # so inorder to decide correct weights please run grid_search.py or run experiments to find the best weights
        # for the reward function

        time_freq="15T",  # "15T", "30T", "1H", "3H", "6H"



        
        agg_method="mean" 
    ):
        super().__init__()

        # ==========  Load data ======================================
        try:
            all_data = pd.read_csv(data_path)
            all_data["local_15min"] = pd.to_datetime(all_data["local_15min"], utc=True)
            all_data.set_index("local_15min", inplace=True)

            # Clip negative solar values to 0
            solar_cols = [c for c in all_data.columns if "total_solar_" in c]
            all_data[solar_cols] = all_data[solar_cols].clip(lower=0.0)

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file {data_path} not found.")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Data file {data_path} is empty.")
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")

        freq_offset = pd.tseries.frequencies.to_offset(time_freq)
        if agg_method == "mean":
            all_data = all_data.resample(time_freq).mean()
        else:
     
            all_data = all_data.resample(time_freq).sum()

        # Store the resampled dataset
        self.all_data = all_data

    
        minutes_per_step = freq_offset.nanos / 1e9 / 60.0
        self.steps_per_day = int(24 * 60 // minutes_per_step)

        total_rows = len(self.all_data)
        self.total_days = total_rows // self.steps_per_day
        if self.total_days < 1:
            raise ValueError(
                f"After resampling, dataset has {total_rows} rows, which is "
                f"less than a single day of {self.steps_per_day} steps."
            )

        self.house_ids = [
            col.split("_")[1] for col in self.all_data.columns
            if col.startswith("grid_")
        ]
        self.num_agents = len(self.house_ids)

       
        self.original_no_p2p_import = {}
        for hid in self.house_ids:
            col_grid = f"grid_{hid}"
            self.original_no_p2p_import[hid] = self.all_data[col_grid].clip(lower=0.0).values
     



        # ========== SPACES (Observation & Action) ===================================

        # Observations: (num_agents, 6)
        #  [own_demand, own_solar, grid_price, peer_price,
        #   total_demand_others, total_solar_others]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_agents, 6),
            dtype=np.float32
        )

        # Actions: (num_agents, 4)
        #  [buyGrid, sellGrid, buyPeers, sellPeers]
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_agents, 4),
            dtype=np.float32
        )

        # ========== REWARD FUNCTION PARAMETERS ======================================
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.max_grid_price = max_grid_price

        self.data = None
        self.env_log = []

        self.day_index = -1   


        self.current_step = 0
        self.num_steps = self.steps_per_day  

        self.demands = {}
        self.solars = {}

        self.previous_actions = {
            hid: np.zeros(4) for hid in self.house_ids
        }


    # Price Functions

    def get_grid_price(self, step_idx):
        # Example price function for Oklahoma
        minutes_per_step = 24 * 60 / self.steps_per_day
        hour = int((step_idx * minutes_per_step) // 60) % 24
        if hour < 14 or hour >= 19:
            return 0.0434
        else:
            return 0.2112

    def get_peer_price(self, step_idx):
        grid_price = self.get_grid_price(step_idx)
        return 0.90 * grid_price

   


   ##########################################################################
    # Gym Required Methods
 
    def reset(self):
        
      
        #self.day_index = (self.day_index + 1) % self.total_days
        self.day_index = np.random.randint(0, self.total_days)

        start_row = self.day_index * self.steps_per_day
        end_row = start_row + self.steps_per_day
        day_data = self.all_data.iloc[start_row:end_row].copy()
        self.data = day_data  

        self.no_p2p_import_day = {}
        for hid in self.house_ids:
            self.no_p2p_import_day[hid] = self.original_no_p2p_import[hid][start_row:end_row]


        self.demands = {}
        self.solars = {}

        for hid in self.house_ids:
            col_grid = f"grid_{hid}"
            col_solar = f"total_solar_{hid}"

            grid_series = day_data[col_grid].fillna(0.0)
            solar_series = day_data[col_solar].fillna(0.0).clip(lower=0.0)

            demand_array = grid_series.values + solar_series.values
            demand_array = np.clip(demand_array, 0.0, None)

            self.demands[hid] = demand_array
            self.solars[hid]  = solar_series.values

        self.current_step = 0
        self.env_log = []
        for hid in self.house_ids:
            self.previous_actions[hid] = np.zeros(4)

        obs = self._get_obs()
        obs_list = [obs[i] for i in range(self.num_agents)]
        return obs_list

    def step(self, actions):
        
        # Validate & clamp
        actions = np.array(actions, dtype=np.float32)
        if actions.shape != (self.num_agents, 4):
            raise ValueError(
                f"Actions shape mismatch: got {actions.shape}, "
                f"expected {(self.num_agents, 4)}"
            )
        actions = np.clip(actions, 0.0, 1.0)

        a_buyGrid  = actions[:, 0]
        a_sellGrid = actions[:, 1]
        a_buyPeers = actions[:, 2]
        a_sellPeers= actions[:, 3]

        # Current demands & solars
        demands = []
        solars  = []
        for hid in self.house_ids:
            demands.append(self.demands[hid][self.current_step])
            solars.append(self.solars[hid][self.current_step])
        demands = np.array(demands, dtype=np.float32)
        solars  = np.array(solars,  dtype=np.float32)

        grid_price = self.get_grid_price(self.current_step)
        peer_price = self.get_peer_price(self.current_step)

        # Local coverage from solar
        local_coverage = np.copy(solars)

        # Determine shortfall / surplus
        shortfall = np.maximum(demands - local_coverage, 0.0)
        surplus   = np.maximum(local_coverage - demands, 0.0)

        grid_import_with_p2p = np.zeros(self.num_agents, dtype=np.float32)
        grid_export = np.zeros(self.num_agents, dtype=np.float32)

        # Grid import / export
        netGrid = a_buyGrid - a_sellGrid
        for i in range(self.num_agents):
            if netGrid[i] > 0:
                buy_amount = netGrid[i] * shortfall[i]
                grid_import_with_p2p[i] = min(buy_amount, shortfall[i])
            elif netGrid[i] < 0:
                sell_amount = -netGrid[i] * surplus[i]
                grid_export[i] = min(sell_amount, surplus[i])

        final_shortfall = shortfall - grid_import_with_p2p
        final_surplus   = surplus - grid_export

        # Peer-to-peer buy / sell
        netPeer = a_buyPeers - a_sellPeers
        p2p_buy_request = np.zeros(self.num_agents, dtype=np.float32)
        p2p_sell_offer  = np.zeros(self.num_agents, dtype=np.float32)

        for i in range(self.num_agents):
            if netPeer[i] > 0:
                p2p_buy_request[i] = min(netPeer[i] * final_shortfall[i],
                                         final_shortfall[i])
            elif netPeer[i] < 0:
                p2p_sell_offer[i]  = min(-netPeer[i] * final_surplus[i],
                                         final_surplus[i])

        total_sell = np.sum(p2p_sell_offer)
        total_buy  = np.sum(p2p_buy_request)
        matched = min(total_sell, total_buy)

        if matched > 1e-9:
            sell_fraction = p2p_sell_offer / (total_sell + 1e-12)
            buy_fraction  = p2p_buy_request / (total_buy  + 1e-12)
            actual_sold   = matched * sell_fraction
            actual_bought = matched * buy_fraction
        else:
            actual_sold   = np.zeros(self.num_agents, dtype=np.float32)
            actual_bought = np.zeros(self.num_agents, dtype=np.float32)

        final_surplus   -= actual_sold
        final_shortfall -= actual_bought

        # Any leftover shortfall is forced to grid
        forced_grid_buy = np.maximum(final_shortfall, 0.0)
        grid_import_with_p2p += forced_grid_buy  # add forced buy
        final_shortfall -= forced_grid_buy       # leftover shortfall becomes zero

        # Compute costs (same as original)
        feed_in_tariff = 0.04  # for selling back to the grid
        costs = (
            (grid_import_with_p2p * grid_price)
            - (grid_export * feed_in_tariff)
            + (actual_bought * peer_price)
            - (actual_sold * peer_price)
        )

        # Compute reward
        final_rewards = self._compute_rewards(
            grid_import=grid_import_with_p2p,
            actual_sold=actual_sold,
            actual_bought=actual_bought,
            grid_price=grid_price,
            peer_price=peer_price
        )

        no_p2p_import_this_step = np.array([
            self.no_p2p_import_day[hid][self.current_step] for hid in self.house_ids
        ], dtype=np.float32)

        # Info & logging
        info = {
            "p2p_buy":               actual_bought,
            "p2p_sell":              actual_sold,
            "grid_import_with_p2p":  grid_import_with_p2p,
            "grid_import_no_p2p":    no_p2p_import_this_step,
            "grid_export":           grid_export,
            "costs":                 costs,
            "step":                  self.current_step
        }

        self.env_log.append([
            self.current_step,
            np.sum(grid_import_with_p2p),
            np.sum(grid_export),
            np.sum(actual_bought),
            np.sum(actual_sold),
            np.sum(costs)
        ])

        # Increment step, check done
        self.current_step += 1
        done = (self.current_step >= self.num_steps)

        # Next observation
        obs_next = self._get_obs()
        obs_next_list = [obs_next[i] for i in range(self.num_agents)]
        rewards_list  = [final_rewards[i] for i in range(self.num_agents)]

        return obs_next_list, rewards_list, done, info

    def _get_obs(self):
        # Build observation array for each agent
        obs = []
        step = min(self.current_step, self.num_steps - 1)  # clamp safety

        # Compute total demand/solar across all houses
        total_demand_all = 0.0
        total_solar_all  = 0.0
        for hid in self.house_ids:
            total_demand_all += self.demands[hid][step]
            total_solar_all  += self.solars[hid][step]

        grid_price = self.get_grid_price(step)
        peer_price = self.get_peer_price(step)

        for hid in self.house_ids:
            own_demand = self.demands[hid][step]
            own_solar  = self.solars[hid][step]

            total_demand_others = total_demand_all - own_demand
            total_solar_others  = total_solar_all  - own_solar
            if total_solar_others < 0.0:
                total_solar_others = 0.0

            obs.append([
                own_demand,
                own_solar,
                grid_price,
                peer_price,
                total_demand_others,
                total_solar_others
            ])

        return np.array(obs, dtype=np.float32)

    def _compute_jains_index(self, usage_array):
        """ Simple Jain's Fairness Index. """
        x = np.array(usage_array, dtype=np.float32)
        numerator = (np.sum(x))**2
        denominator = len(x) * np.sum(x**2) + 1e-8
        return numerator / denominator

    # def _is_daytime(self, step_idx):
    #     """
    #     Returns True if the timestamp at 'step_idx' is between 7:30 and 19:30,
    #     else False. Adjust as needed for your dataset/time boundaries.
    #     """
    #     if step_idx >= self.num_steps:
    #         step_idx = self.num_steps - 1
    #     dt = self.data.index[step_idx]  # a pandas Timestamp
    #     hour = dt.hour
    #     minute = dt.minute

    #     # Check if time is >= 7:30 and < 19:30
    #     if ((hour > 7 or (hour == 7 and minute >= 30))
    #         and (hour < 19 or (hour == 19 and minute < 30))):
    #         return True
    #     else:
    #         return False

    def _compute_rewards(
        self,
        grid_import,
        actual_sold,
        actual_bought,
        grid_price,
        peer_price
    ):
        
        w1 = self.alpha
        w2 = self.beta
        w3 = self.gamma
        w4 = self.delta

        # # Daytime logic for penalizing grid usage
        # if not self._is_daytime(self.current_step):
        #     w1_effective = 0.0
        # else:
        #     w1_effective = w1

        total_solar_now = 0.0
        for hid in self.house_ids:
            total_solar_now += self.solars[hid][self.current_step]
        

        # please adjust this threshold as needed or use simpler daytime logic above
        # this threshold depends on your dataset and how you define "daytime" for 100 houses 
        # inaccuracy in solar generation itself can result in 0.1 total solar production
        THRESHOLD = 0.1
        if total_solar_now < THRESHOLD:
            w1_effective = 0.0
        else:
            w1_effective = w1

        # Normalize by total usage to distribute reward/penalty
        G = np.sum(grid_import)
        S = np.sum(actual_sold)
        B = np.sum(actual_bought)
        G = max(G, 1e-8)
        S = max(S, 1e-8)
        B = max(B, 1e-8)

        jfi = self._compute_jains_index(actual_bought + actual_sold)

        p_grid_norm = grid_price / self.max_grid_price
        p_peer_norm = peer_price / self.max_grid_price

        final_rewards = np.zeros(len(grid_import), dtype=np.float32)
        for i in range(len(grid_import)):
            g_i = grid_import[i]
            s_i = actual_sold[i]
            b_i = actual_bought[i]

            # Grid penalty (only if daytime)
            term_grid = - w1_effective * (g_i / G) * p_grid_norm

            # P2P sell / buy
            term_sell = w2 * (s_i / S) * p_peer_norm
            term_buy  = w3 * (b_i / B) * p_peer_norm

            # Fairness
            term_fair = w4 * jfi

            final_rewards[i] = term_grid + term_sell + term_buy + term_fair

        return final_rewards

    def save_log(self, filename="env_log.csv"):
        """Save environment step log to CSV."""
        columns = [
            "Step", "Total_Grid_Import", "Total_Grid_Export",
            "Total_P2P_Buy", "Total_P2P_Sell", "Total_Cost",
        ]
        df = pd.DataFrame(self.env_log, columns=columns)
        df.to_csv(filename, index=False)
        print(f"Environment log saved to {filename}")

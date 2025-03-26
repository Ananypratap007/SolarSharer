import numpy as np

class StrictRoundRobinPolicy:
    

    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.global_step_counter = 0

    def select_actions(self, obs_list, day_idx):
        
        
        step_idx = self.global_step_counter
        active_seller = step_idx % self.num_agents
        active_buyer  = (step_idx + 1) % self.num_agents

        actions = np.zeros((self.num_agents, 4), dtype=np.float32)

        for i in range(self.num_agents):
            own_demand = obs_list[i][0]
            own_solar  = obs_list[i][1]

            if own_solar > own_demand:
                if i == active_seller:
                    
                    actions[i, 3] = 1.0 
                else:
                   
                    actions[i, 1] = 1.0            
            else:
                if i == active_buyer:
                    
                    actions[i, 2] = 1.0 
                else:
                    
                    actions[i, 0] = 1.0  
        self.global_step_counter += 1
        return actions

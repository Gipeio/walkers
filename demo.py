import numpy as np
import pandas as pd
import seaborn as sns
import mesa

# 1) create agent
class money_agent(mesa.Agent):
    
    # 1)
    def __init__(self, model):
        super().__init__(model)
        # initial value
        self.wealth = 1
    
    # 3) make the agent DO
    def say_wealth(self):
        print(f"Hello, i'm agent n°{self.unique_id}  and i'm broke!")
        
# 2) create model: the list of agents
class money_model(mesa.Model):
    
    # 2)
    def __init__(self, n=10, seed=None):
        super().__init__(seed=seed)
        self.num_agents = n
        # Create Agents
        for _ in range(n):
            money_agent(self)
        
    # 3)
    def step(self):
        self.agents.shuffle_do("say_wealth")
        

starter_model = money_model(10)
starter_model.step()
import numpy as np
import pandas as pd
import seaborn as sns
import mesa
import matplotlib.pyplot as plt

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
        
    def exchange(self):
        if self.wealth > 0:
            other_agent = self.random.choice(self.model.agents)
            if other_agent is not None:
                other_agent.wealth += 1
                self.wealth -= 1
        
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
        self.agents.shuffle_do("exchange")
        
# 4) execute and visualize data
all_wealth = []
# This runs the model 100 times, each model executing 30 steps.
for _ in range(100):
    # Run the model
    model = money_model(10)
    for _ in range(30):
        model.step()

    # Store the results
    for agent in model.agents:
        all_wealth.append(agent.wealth)

# Use seaborn to create plot
g = sns.histplot(all_wealth, discrete=True)
g.set(title="Wealth distribution", xlabel="Wealth", ylabel="number of agents");

# DL plot
plt.savefig("wealth_distribution.png")
plt.show()
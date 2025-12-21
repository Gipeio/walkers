import numpy as np
import pandas as pd
import seaborn as sns
import mesa
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid
import matplotlib.pyplot as plt

def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.agents]
    x = sorted(agent_wealths)
    n = model.num_agents
    B = sum(xi * (n - i) for i, xi in enumerate(x)) / (n * sum(x))
    return 1 + (1 / n) - 2 * B


class money_agent(CellAgent):
    
    def __init__(self, model, cell):
        super().__init__(model)
        self.cell = cell
        self.wealth = 1
    
    def move(self):
        self.cell = self.cell.neighborhood.select_random_cell()
        
    def exchange(self):
        cellmates = [
            a for a in self.cell.agents if a is not self
        ]
        if self.wealth > 0 and cellmates:
            other_agent = self.random.choice(cellmates)
            other_agent.wealth += 1
            self.wealth -= 1
    
class money_model(mesa.Model):
    
    def __init__(self, n, width, height, seed=None):
        super().__init__(seed=seed)
        self.num_agents = n
        self.grid = OrthogonalMooreGrid((width,height),torus=True,random=self.random)
        self.datacollector = mesa.DataCollector(
            model_reporters={"Gini": compute_gini},
            agent_reporters={"Wealth": "wealth"}
        )
        
        agents = money_agent.create_agents(
            self,
            self.num_agents,
            self.random.choices(self.grid.all_cells.cells, k=self.num_agents)
        )
        
    def step(self):
        self.datacollector.collect(self)
        self.agents.shuffle_do("move")
        self.agents.do("exchange")
        
model = money_model(100,10,10)
for _ in range(20):
    model.step()
    
agent_wealth = model.datacollector.get_agent_vars_dataframe()
print(agent_wealth.head)

### PLOTS

# # GINI

# # gini = model.datacollector.get_model_vars_dataframe()
# # g = sns.lineplot(data=gini)
# # g.set(title="Gini Coefficient over Time", ylabel="Gini Coefficient")

# # WEALTH REPARTITION AT THE END OF THE SIMULATION

# last_step = agent_wealth.index.get_level_values("Step").max()
# end_wealth = agent_wealth.xs(last_step, level="Step")[
#     "Wealth"
# ]
# g = sns.histplot(end_wealth, discrete=True)
# g.set(
#     title="Distribution of wealth at the end of simulation",
#     xlabel="Wealth",
#     ylabel="number of agents",
# )

# # WEALTH OF A GIVEN AGENT

# one_agent_wealth = agent_wealth.xs(54, level="AgentID")

# g = sns.lineplot(data=one_agent_wealth, x="Step", y="Wealth")
# g.set(title="Wealth of specified agent over time")

# WEALTH OF MULTIPLE GIVEN AGENTS

agent_list = [3, 47, 89]

multiple_agent_wealth = agent_wealth[
    agent_wealth.index.get_level_values("AgentID").isin(agent_list)
]

g = sns.lineplot(data=multiple_agent_wealth, x="Step", y="Wealth", hue="AgentID")
g.set(title="Wealth of specified agents over time")

plt.show()
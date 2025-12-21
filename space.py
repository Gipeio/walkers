import numpy as np
import pandas as pd
import seaborn as sns
import mesa
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid
import matplotlib.pyplot as plt

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
        self.grid = OrthogonalMooreGrid((width,height),True,random=self.random)
        
        money_agent.create_agents(
            self,
            self.num_agents,
            self.random.choices(self.grid.all_cells.cells, k=self.num_agents)
        )
    
    def step(self):
        self.agents.shuffle_do("move")
        self.agents.do("exchange")
        
model = money_model(100, 10, 10)
for _ in range(20):
    model.step()
    
agent_counts = np.zeros((model.grid.width, model.grid.height))

for cell in model.grid.all_cells:
    agent_counts[cell.coordinate] = len(cell.agents)
g = sns.heatmap(agent_counts,cmap="viridis", annot=True, cbar=False, square=True)
g.figure.set_size_inches(5,5)
g.set(title="Number of agents on each cell of the grid")

plt.show()
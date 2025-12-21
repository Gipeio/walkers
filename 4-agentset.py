import numpy as np
import pandas as pd
import seaborn as sns
import mesa
import matplotlib.pyplot as plt

def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.agents]
    x = sorted(agent_wealths)
    n = model.num_agents
    B = sum(xi * (n - i) for i, xi in enumerate(x)) / (n * sum(x))
    return 1 + (1 / n) - 2 * B


class money_agent(mesa.Agent):

    def __init__(self, model, ethnicity):
        super().__init__(model)
        self.wealth = 1
        self.ethnicity = ethnicity

    def give_money(self, similars):
        if self.wealth > 0:
            other_agent = self.random.choice(similars)
            other_agent.wealth += 1
            self.wealth -= 1


class money_model(mesa.Model):

    def __init__(self, n):
        super().__init__()
        self.num_agents = n
        
        ethnicities = ["Green", "Blue", "Mixed"]

        # Create agents
        money_agent.create_agents(
            model=self,
            n=n,
            ethnicity = self.random.choices(ethnicities, k=self.num_agents)
        )
        
        self.datacollector = mesa.DataCollector(
            agent_reporters={"Wealth": "wealth", "Ethnicity": "ethnicity"}
        )



    def step(self):
        self.datacollector.collect(self)
        grouped_agents = model.agents.groupby("ethnicity")
        for ethnic, similars in grouped_agents:
            if ethnic != "Mixed":
                similars.shuffle_do("give_money", similars)
            else:
                similars.shuffle_do("give_money", self.agents)
        
model = money_model(100)
for _ in range(20):
    model.step()

# get the data
data = model.datacollector.get_agent_vars_dataframe()
# assign histogram colors
palette = {"Green": "green", "Blue": "blue", "Mixed": "purple"}
g = sns.histplot(data=data, x="Wealth", hue="Ethnicity", discrete=True, palette=palette)
g.set(title="Wealth distribution", xlabel="Wealth", ylabel="number of agents");

plt.show()
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Set hyperparameters
num_episodes = 1000
max_steps_per_episode = 200
batch_size = 64
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
gamma = 0.99  # Discount factor

class Brain(nn.Module):

    def __init__(self, n_observations: int, n_actions=4, layer_size=32) -> None:
        super(Brain, self).__init__()
        self.layer1 = nn.Linear(n_observations, layer_size)
        self.layer2 = nn.Linear(layer_size, layer_size)
        self.layer3 = nn.Linear(layer_size, n_actions)

    def forward(self, x) -> None:
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class Agent():

    def __init__(self, brain: Brain) -> None:
        self.brain = brain
    
    def decision(self, game_state):
        probabilities = self.brain(game_state)
        return torch.argmax(probabilities)

# game_state = torch.tensor([1, 1, 1, 1, 1]).float()

# brain = Brain(5)
# agent = Agent(brain)
# move = agent.decision(game_state)
# print(move)

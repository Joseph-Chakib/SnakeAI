import math
from mimetypes import init
import random
from typing import NamedTuple
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
steps_per_episode = 256
batch_size = 5
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
gamma = 0.99  # Discount factor

class Brain(nn.Module):

    def __init__(self, n_observations: int, n_actions=4, layer_size=128) -> None:
        super(Brain, self).__init__()
        self.layer1 = nn.Linear(n_observations, layer_size)
        self.layer2 = nn.Linear(layer_size, layer_size)
        self.layer3 = nn.Linear(layer_size, n_actions)
        print(self.layer1, self.layer2, self.layer3)

    def forward(self, x) -> None:
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class Memory(NamedTuple):
    current_state: torch.tensor
    action: torch.tensor
    reward: int
    next_state: torch.tensor
    done: int


# class ReplayBuffer():

#     def __init__(self, batch_size=200) -> None:
#         self.buffer = []
#         self.batch_size = batch_size

#     # def store(self, experience: torch.tensor) -> None:
#     #     self.buffer = torch.cat((self.buffer, experience), dim=1)

#     def store(self, memory: Memory):
#         self.buffer.append(memory)

#     def get_buffer(self):
#         return self.buffer

class Agent():

    def __init__(self, brain: Brain, epsilon=1, decay=0.999, gamma=0.1, learning_rate=0.001) -> None:
        self.brain = brain
        self.collision_buffer = []
        self.step_buffer = []
        self.apple_buffer = []
        self.buffer_limit = 100000
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.decay = decay
        self.loss = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.brain.parameters(), lr=learning_rate)
    
    def decision(self, game_state) -> torch.tensor:
        random_float = random.uniform(0, 1)
        self.epsilon *= self.decay
        if random_float < self.epsilon * 1:
            move = random.randint(0, 3)
            return_value = torch.tensor(move)
            # print(f'Function call data: roll: {random_float} | epsilon {self.epsilon} | move {return_value}')
            return return_value
        else:
            # something may be going wrong here
            probabilities = self.brain(game_state)
            return torch.argmax(probabilities)

    def store(self, current_state, action, reward, next_state, done) -> None:
        memory = Memory(current_state, action, reward, next_state, done)
        if reward == -50:
            self.collision_buffer.append(memory)
        elif reward > -50 and reward < 20:
            self.step_buffer.append(memory)
        else:
            self.apple_buffer.append(memory)

    def buffer_size(self):
        return len(self.collision_buffer), len(self.step_buffer), len(self.apple_buffer)

    def predict(self, game_state):
        q_values = self.brain(game_state)
        return q_values

    def target(self, experience):

        current_state = experience[0]
        action = experience[1]
        reward = experience[2]
        next_state = experience[3]
        terminal = experience[-1]

        print(f'Experience: ({current_state}, {action}, {reward}, {next_state}, {terminal})')

        if terminal:
            target = torch.tensor(reward).float()
        else:
            value = reward + self.gamma * torch.max(self.predict(next_state))
            target = torch.tensor(value).float()

        return target

    def backward(self, prediction, target):
        loss = self.loss(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print(f'function call: prediction: {prediction} | target: {target} | loss: {loss}')

    def train(self, batches):
        length_one, length_two, length_three = self.buffer_size()
        if length_three < self.batch_size:
            print('Not enough apple experiences')
            return None
        for batch in batches:
            # fix this code so that only the q_value of the action taken is returned not the maximum 
            for experience in batch:
                target = self.target(experience)
                prediction = self.predict(experience[0])[experience[1]]
                self.backward(prediction, target)

    def sample(self):
        length_one, length_two, length_three = self.buffer_size()
        if length_three < self.batch_size:
            return None
        batch3 = random.sample(self.collision_buffer, self.batch_size)
        batch2 = random.sample(self.step_buffer, self.batch_size)
        batch1 = random.sample(self.apple_buffer, self.batch_size)
        # current_states, actions, rewards, next_states, dones = zip(*batch)
        # current_states = torch.tensor(current_states)
        # actions = torch.tensor(actions)
        # rewards = torch.tensor(rewards)
        # next_states = torch.tensor(next_states)
        # dones = torch.tensor(dones)
        # return current_states, actions, rewards, next_states, dones
        return batch1, batch2, batch3



# TODO List
# After a certain amount of steps we are going to sample from the buffer data and train the model
# To train the model: go through each experience, get a prediction, get a target value, calculate loss, using adam optimize the model
# To get target value return action value or return bellman equation
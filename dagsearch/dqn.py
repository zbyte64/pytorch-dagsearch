#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import math
import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, world_size, action_size):
        super(DQN, self).__init__()
        self.f1 = nn.Linear(world_size, world_size // 2)
        self.f2 = nn.Linear(world_size // 2, world_size // 4)
        self.head = nn.Linear(world_size // 4, action_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        return self.head(x)


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

class Trainer(object):
    def __init__(self, world):
        self.world = world
        self.world_size = world.observe().shape[0]
        self.action_size = world.actions_shape()[0]
        self.policy_net = DQN(self.world_size, self.action_size)
        self.target_net = DQN(self.world_size, self.action_size)
        self.memory = ReplayMemory(10000)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state)
        else:
            return torch.normal(torch.zeros(self.action_size)).to(device)

    def optimize_trainer_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        state_action_values = self.policy_net(state_batch)
        next_state_values = self.target_net(state_batch)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)#.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def train(self, dataset, epochs=5):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        optimizer_ft = optim.SGD(self.world.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.MSELoss()

        def make_one_hot(labels, num_classes):
            y = torch.eye(num_classes)
            return y[labels]

        state = self.world.observe()
        last_loss = None
        for e in range(epochs):
            for i, batch in enumerate(dataset):
                x, y = batch
                _y = make_one_hot(y, 10) #TODO generalize training
                prior_py = torch.softmax(self.world(x), dim=1)
                prior_world_loss = criterion(prior_py, _y)

                # Select and perform an action
                action = self.select_action(state)
                a, d = torch.argmax(action[:-1]).item(), torch.sigmoid(action[-1]).item()
                self.world.perform_action(a, d)


                py = torch.softmax(self.world(x), dim=1)
                world_loss = criterion(py, _y)
                world_loss.backward()
                optimizer_ft.step()

                reward = (prior_world_loss - world_loss)
                if last_loss is not None:
                    reward = (last_loss - world_loss)
                last_loss = world_loss
                print('World Loss: %s , Reward: %s' % (world_loss.item(), reward.item()))
                reward = torch.sigmoid(torch.tensor([reward], device=device))
                next_state = self.world.observe()

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.optimize_trainer_model()
                # Update the target network, copying all weights and biases in DQN
                if i % TARGET_UPDATE == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

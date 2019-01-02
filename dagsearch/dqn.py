#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
import heapq

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
    def __init__(self, world, env):
        self.world = world
        self.env = env
        self.world_size = world.observe().shape[0]
        self.action_size = len(world.actions())
        self.policy_net = DQN(self.world_size, self.action_size)
        self.target_net = DQN(self.world_size, self.action_size)
        self.memory = ReplayMemory(10000)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.optimizer.zero_grad()
        self.steps_done = 0
        self.score_board = []

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1, keepdim=True)[1]
        else:
            return torch.from_numpy(np.array([[self.env.action_space.sample()]]))


    def optimize_trainer_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.from_numpy(np.array(batch.reward, dtype=np.float32))
        #print(state_batch.shape, action_batch.shape)
        #print(action_batch)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        #print(state_action_values.shape, expected_state_action_values.shape)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def train(self, iterations=1000):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer.zero_grad()

        state = self.env.observe()
        for i in range(iterations):
            # Select and perform an action
            action = self.select_action(state)
            ob, reward, episode_over, info = self.env.step(action.item())
            if episode_over:
                print('episode over')
            else:
                print('Reward %.2f , Loss %.2f ' % (reward, info['loss']))
            #print(action.item(), reward, episode_over, info)
            next_state = ob
            # Store the transition in memory
            self.memory.push(state, action, next_state, reward)
            state = next_state

            # Perform one step of the optimization (on the target network)
            self.optimize_trainer_model()
            # Update the target network, copying all weights and biases in DQN
            if i % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            if episode_over:
                #TODO record score
                self.env.reset()

    def score_model(self):
        self.world.mov_fork(self.world)
        valid_data = next(self.world.valid_data)
        x, y = valid_data
        py = self.world.graph(x)
        loss = self.world.criterion(y, py)
        heapq.heappush(self.score_board, (loss, self.world.graph))

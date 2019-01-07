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
from .env import *

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'hidden_state'))


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


class DRQN(nn.Module):
    def __init__(self, world_size, action_size, hidden_size=8, num_layers=1):
        super(DRQN, self).__init__()
        self.embedding_size = world_size * 2
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.f1 = nn.Linear(world_size, world_size * 2)
        self.f2 = nn.Linear(world_size * 2, self.embedding_size*num_layers)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size,
            num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(self.hidden_size*num_layers, action_size)
        self.add_module('f1', self.f1)
        self.add_module('f2', self.f2)
        self.add_module('lstm', self.lstm)
        self.add_module('head', self.head)

    def forward(self, x, hidden):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = x.view((x.shape[0], self.num_layers, self.embedding_size))
        x, hidden = self.lstm(x, hidden)
        x = x.contiguous().view((x.shape[0], -1))
        return self.head(x), hidden


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
        self.hidden_size = 100
        self.hidden_layers = 1
        self.policy_net = DRQN(self.world_size, self.action_size, self.hidden_size, self.hidden_layers).to(device)
        self.target_net = DRQN(self.world_size, self.action_size, self.hidden_size, self.hidden_layers).to(device)
        self.memory = ReplayMemory(10000)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.steps_done = 0
        self.score_board = []
        self._hidden_state = self._make_hidden_state()
    
    def _make_hidden_state(self, bs=1):
        return (
            torch.zeros(self.hidden_layers, bs, self.hidden_size).to(device),
            torch.zeros(self.hidden_layers, bs, self.hidden_size).to(device)
        )

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        with torch.no_grad():
            _hs_has_nan = torch.sum(torch.stack((torch.isnan(self._hidden_state[0]), torch.isnan(self._hidden_state[1])))).item()
            assert not _hs_has_nan, str(self._hidden_state)
            if sample > eps_threshold:   
                y, hidden = self.policy_net(state.to(device), self._hidden_state)
                a = y.max(1, keepdim=True)[1]
                hidden[0][torch.isnan(hidden[0])] = 0.
                hidden[1][torch.isnan(hidden[1])] = 0.
            else:
                a, hidden = (
                    torch.tensor([[self.env.action_space.sample()]], dtype=torch.int64),
                    self._make_hidden_state()
                )
            self._hidden_state = hidden
            _hs_has_nan = torch.sum(torch.stack((torch.isnan(self._hidden_state[0]), torch.isnan(self._hidden_state[1])))).item()
            assert not _hs_has_nan, str(self._hidden_state)
            return a, self._hidden_state


    def optimize_trainer_model(self):
        if len(self.memory) < BATCH_SIZE:
            return -1.
        self.optimizer.zero_grad()
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
                                                    if s is not None]).to(device)
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)
        #reconstruct prior memory for training policy net
        hidden_state = (
            torch.cat([b[0] for b in batch.hidden_state], dim=1).to(device),
            torch.cat([b[1] for b in batch.hidden_state], dim=1).to(device)
        )
        _hs_has_nan = torch.sum(torch.stack((torch.isnan(hidden_state[0]), torch.isnan(hidden_state[1])))).item()
        assert not _hs_has_nan, str(hidden_state)
        #this emits a state which can be fed into target net, but should we?
        state_action_values, next_hidden_state = self.policy_net(state_batch, hidden_state)
        next_hidden_state[0][torch.isnan(next_hidden_state[0])] = 0.
        next_hidden_state[1][torch.isnan(next_hidden_state[1])] = 0.
        _hs_has_nan = torch.sum(torch.stack((torch.isnan(next_hidden_state[0]), torch.isnan(next_hidden_state[1])))).item()
        assert not _hs_has_nan, str(next_hidden_state)
        state_action_values = state_action_values.gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE).to(device)
        #next_hidden_state = torch.cat([h for s, h in zip(batch.next_state, batch.hidden_state)
        #                                            if s is not None])
        #next_hidden_state[0][torch.isnan(next_hidden_state[0])] = 0.
        #next_hidden_state[1][torch.isnan(next_hidden_state[1])] = 0.
        next_state_values[non_final_mask] = self.target_net(non_final_next_states, next_hidden_state)[0].max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        expected_state_action_values[torch.isnan(expected_state_action_values)] = self.env.action_space.sample()
        state_action_values[torch.isnan(state_action_values)] = self.env.action_space.sample()

        # Compute Huber loss
        #print(state_action_values.shape, expected_state_action_values.shape)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        _loss = float(loss.item())
        assert _loss >= 0., str((expected_state_action_values, state_action_values, loss))
        
        # Optimize the model
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return _loss

    def train(self, iterations=1000):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer.zero_grad()

        state = self.env.observe()
        for i in range(iterations):
            # Select and perform an action
            prior_hidden_state = self._hidden_state
            action, next_hidden_state = self.select_action(state)
            assert action.dtype == torch.int64, str(action.dtype)
            ob, reward, episode_over, info = self.env.step(int(action.item()))
            if episode_over:
                print('episode over')
                self._hidden_state = self._make_hidden_state()
            #print(action.item(), reward, episode_over, info)
            next_state = ob
            # Store the transition in memory
            self.memory.push(state, action.to(device), next_state, reward, prior_hidden_state)
            state = next_state

            # Perform one step of the optimization (on the target network)
            trainer_loss = self.optimize_trainer_model()
            # Update the target network, copying all weights and biases in DQN
            if i % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                print('Trainer Loss %.4f , Loss %.4f , Gas %.2f' % (trainer_loss, info['loss'], info['gas']))
            if episode_over:
                self.score_model()
                self.env.reset()

    def score_model(self):
        #self.world.mov_fork(self.world)
        valid_data = next(self.world.valid_data)
        x, y = valid_data
        x = x.to(device)
        y = y.to(device)
        py = self.world.graph(x)
        loss = self.world.criterion(py, y)
        print('Validation loss:', loss)
        #.heappush(self.score_board, (loss, self.world.graph))

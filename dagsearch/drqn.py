#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
import heapq
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .env import *


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        t = args #Transition(*args)
        if len(self.memory) < self.capacity:
            self.memory.append(t)
        else:
            self.memory[self.position] = t
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class NavEmbed(nn.Module):
    def __init__(self, world_size, action_size, num_layers=2):
        super(NavEmbed, self).__init__()
        self.world_size = world_size
        self.hidden_size = world_size * 2
        self.num_layers = num_layers
        self.input_size = world_size + action_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size,
            num_layers=num_layers)
        self.hidden_state = None
        self.discrete_mask = torch.ones(world_size, device=device, dtype=torch.int64)
        self.add_module('lstm', self.lstm)
        
    def generate_hidden_state(self, bs=1):
        return (
            torch.zeros(self.num_layers, bs, self.hidden_size).to(device),
            torch.zeros(self.num_layers, bs, self.hidden_size).to(device)
        )
    
    def forward(self, x, action, hidden_state=None):
        if hidden_state is None:
            if self.hidden_state is None:
                self.hidden_state = self.generate_hidden_state(x.shape[0])
            hidden_state = self.hidden_state
            manage_hidden_state = True
        else:
            manage_hidden_state = False
        if len(x.shape) == 2:
            # M x B x V
            x = x.view((1, x.shape[0], -1))
            action = action.view((1, x.shape[0], -1))
            reshape = True
        else:
            reshape = False
        n_x = torch.cat([x, action], dim=2)
        #print(n_x.shape, hidden_state[0].shape, x.shape, action.shape, self.input_size)
        n_x, hidden = self.lstm(n_x, hidden_state)
        #hidden[0][torch.isnan(hidden[0])] = 0.
        #hidden[1][torch.isnan(hidden[1])] = 0.
        z1, z2 = n_x[:,:,:self.world_size], n_x[:,:,self.world_size:]
        TINY = 1e-8
        probs = torch.clamp(torch.sigmoid(z1 + z2), TINY, 1 - TINY)
        scale = torch.sigmoid(z2) + TINY
        mu = z1 + x
        cz = torch.distributions.normal.Normal(mu, scale)
        dz = torch.distributions.bernoulli.Bernoulli(probs=probs)
        def log_prob(sample):
            c_prob = cz.log_prob(sample)
            d_prob = dz.log_prob(sample)
            #assume sample is discrete/bernoulli otherwise normal dist
            c_mask = torch.max(torch.isnan(d_prob), dim=0, keepdim=True)[0].squeeze()
            self.discrete_mask[c_mask] = 0
            d_mask = self.discrete_mask
            c_mask = 1 - d_mask
            prob = torch.cat([c_prob[c_mask], d_prob[d_mask]], dim=-1)
            return prob
        if manage_hidden_state:
            self.hidden_state = hidden
        if reshape:
            hidden = hidden[0].view(x.shape[1], -1)
        return log_prob, hidden


class DQN(nn.Module):
    def __init__(self, world_size, action_size):
        super(DQN, self).__init__()
        self.f1 = nn.Linear(world_size, world_size // 2)
        self.f2 = nn.Linear(world_size // 2, world_size // 4)
        self.head = nn.Linear(world_size // 4, action_size)
        self.add_module('f1', self.f1)
        self.add_module('f2', self.f2)
        self.add_module('head', self.head)

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


class Trainer(object): #Actor?
    def __init__(self, world, env):
        self.world = world
        self.env = env
        self.world_size = world.observe().shape[0]
        self.action_size = len(world.actions())
        self.embeding_layers = 2
        self.dqn_size = self.world_size + self.world_size * self.embeding_layers * 2
        self.embeding = NavEmbed(self.world_size, self.action_size, num_layers=self.embeding_layers).to(device)
        self.policy_net = DQN(self.dqn_size, self.action_size).to(device)
        self.target_net = DQN(self.dqn_size, self.action_size).to(device)
        self.memory = ReplayMemory(10000)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.embeding_optimizer = optim.RMSprop(self.embeding.parameters())
        self.embeding_criterion = nn.MSELoss()
        self.embeding_memory = ReplayMemory(100)
        self.steps_done = 0
        
    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        with torch.no_grad():
            if sample > eps_threshold:   
                y = self.policy_net(state)
                a = y.max(1, keepdim=True)[1]
            else:
                a = torch.tensor([[self.env.action_space.sample()]], dtype=torch.int64)
            return a
    
    def encode_world(self, state, prior_action):
        state = state.to(device)
        with torch.no_grad():
            prior_action_encode = torch.eye(self.action_size)[int(prior_action.cpu().item())].view(1, -1).type(torch.float32).to(device)
            _, state_embed = self.embeding(state, prior_action_encode)
        state_embed = state_embed.view(state.shape[0], -1)
        obs = torch.cat([state, state_embed.detach()], dim=1).to(device)
        return obs

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
                #this emits a state which can be fed into target net, but should we?
        state_action_values = self.policy_net(state_batch)
        state_action_values = state_action_values.gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE).to(device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
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
    
    def optimize_embeding_model(self, max_len=BATCH_SIZE):
        bs = min(len(self.embeding_memory), 16)
        if not bs:
            return -1.
        #[[(state, action)]]
        self.embeding.train()
        self.embeding_optimizer.zero_grad()
        sessions = self.embeding_memory.sample(bs)
        #=> [(states, actions)] => states, actions
        loss = torch.tensor(0., device=device)
        updates = False
        for session in sessions:
            #[(state, action)]
            session = session[0] # stored as tuple from ReplayMemory
            #print(len(session[0]))
            if len(session) < 2:
                continue
            states = torch.stack([x[0].to(device) for x in session], dim=1).view(len(session), 1, -1)
            actions = torch.stack([x[1].to(device) for x in session], dim=1).view(len(session), 1, -1)
            action_encode = torch.eye(self.action_size)[actions].squeeze(2).type(torch.float32).to(device)
            hidden_state = self.embeding.generate_hidden_state(1)
            log_prob, _ = self.embeding(states[:-1], action_encode[:-1], hidden_state)
            next_state = states[1:].detach()
            _loss = -log_prob(next_state)
            _loss = torch.mean(_loss)
            loss = loss + _loss
        if updates:
            loss.backward()
            self.embeding_optimizer.step()
        return loss.item()

    def train(self, iterations=1000):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer.zero_grad()

        #start with a short episode
        state = self.env.observe().to(device)
        action = torch.zeros(1,1).to(device)
        #create embed repr of world by predicting action result
        obs = self.encode_world(state, action)
        session_memory = list()
        self.embeding_memory.push(session_memory)
        for i in range(iterations):
            # Select and perform an action
            action = self.select_action(obs)
            assert action.dtype == torch.int64, str(action.dtype)
            next_state, reward, episode_over, info = self.env.step(int(action.item()))
            session_memory.append((state, action))
            next_state.to(device)
            if episode_over:
                print('episode over')
                self.prior_state = None
                self.embeding.hidden_state = None
                session_memory.append((next_state, action))
                session_memory = list()
                self.embeding_memory.push(session_memory)
                state = self.env.reset().to(device)
                next_obs = self.encode_world(state, torch.zeros(1,1).to(device))
            else:
                next_obs = self.encode_world(next_state, action)
            # Store the transition in memory
            self.memory.push(obs, action.to(device), next_obs, reward)
            
            obs = next_obs

            # Perform one step of the optimization (on the target network)
            trainer_loss = self.optimize_trainer_model()
            # Update the target network, copying all weights and biases in DQN
            if i % TARGET_UPDATE == 0:
                embedding_loss = self.optimize_embeding_model()
                self.target_net.load_state_dict(self.policy_net.state_dict())
                print('Trainer Loss %.4f , Embedding Loss %.4f , Loss %.4f , Gas %.2f' % (trainer_loss, embedding_loss, info['loss'], info['gas']))

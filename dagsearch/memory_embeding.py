import random
import torch
import torch.nn as nn

from .env import *


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *t):
        if len(self.memory) < self.capacity:
            self.memory.append(t)
        else:
            self.memory[self.position] = t
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class SessionMemory(object):
    def __init__(self, capacity):
        self.sessions = ReplayMemory(capacity)
        self.transitions = ReplayMemory(10000)

    def record(self, current_session, state, action, next_state, reward, episode_over, obs):
        if current_session:
            prior_obs = current_session[-1].pop()
            prior_action = current_session[-1][1]
            self.transitions.push(prior_obs, prior_action, obs, reward)
        current_session.append([state, action, obs])
        if episode_over:
            current_session[-1].pop() #no transition
            current_session.append((next_state, torch.zeros((1,1), dtype=torch.int64)))
            self.sessions.push(current_session)
            #self.save_session(str(self.memory.position), self.current_session)
            return []
        return current_session
    
    def sample_transitions(self, batch_size):
        return self.transitions.sample(batch_size)
    
    def save_session(self, key, session):
        outfile = open('./session/%s.pickle' % key, 'wb')
        pickle.dump(session, outfile)

    def sample(self, batch_size):
        return self.sessions.sample(batch_size)

    def __len__(self):
        return len(self.sessions)


class MemoryEmbed(nn.Module):
    def __init__(self, world_size, action_size, num_layers=1):
        super(MemoryEmbed, self).__init__()
        self.world_size = world_size
        self.action_size = action_size
        self.hidden_size = world_size * 2
        self.num_layers = num_layers
        self.input_size = world_size + action_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size,
            num_layers=num_layers)
        self.discrete_mask = torch.ones(world_size, device=device, dtype=torch.int64)
        self.add_module('lstm', self.lstm)
        
    def generate_hidden_state(self, bs=1):
        return (
            torch.zeros(self.num_layers, bs, self.hidden_size).to(device),
            torch.zeros(self.num_layers, bs, self.hidden_size).to(device)
        )
    
    def forward(self, x, action, hidden_state):
        n_x = torch.cat([x, action], dim=2)
        n_x, hidden = self.lstm(n_x, hidden_state)
        z1, z2 = n_x[:,:,:self.world_size], n_x[:,:,self.world_size:]
        scale = torch.sigmoid(z2) + 1e-5
        mu = z1
        cz = torch.distributions.normal.Normal(mu, scale)
        return cz, hidden
    
    def sample_loss(self, memory, max_len=128, n_batches=16):
        bs = min(len(memory), n_batches)
        if not bs:
            return []
        sessions = memory.sample(bs)
        #=> [(states, actions)] => states, actions
        loss = []
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
            hidden_state = self.generate_hidden_state(1)
            z, _ = self(states[:-1], action_encode[:-1], hidden_state)
            next_state = states[1:]
            _loss = 1-z.log_prob(next_state)
            _loss = torch.mean(_loss)
            loss.append(_loss)
        return loss

import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
import math
import random
from collections import namedtuple
from .env import *


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


class Agent(object):
    def __init__(self, env, memory, embeding, policy_net):
        self.env = env
        self.memory = memory
        self.embeding = embeding
        self.hidden_state = embeding.generate_hidden_state()
        self.policy_net = policy_net.to(device)
        self.steps_done = 0
        self.prior_action = self.generate_initial_action()
        self.current_session = []

    def generate_initial_action(self):
        return torch.zeros(1).unsqueeze(0)

    @property
    def action_size(self):
        return self.embeding.action_size

    @property
    def world(self):
        return self.env.world

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        with torch.no_grad():
            if sample > eps_threshold:
                y = torch.softmax(self.policy_net(state), dim=1)
                m = torch.distributions.categorical.Categorical(y)
                a = m.sample().unsqueeze(1)
                _a = a.item()
                if _a > self.action_size or _a < 0:
                    a = torch.tensor([[self.env.action_space.sample()]], dtype=torch.int64)
            else:
                a = torch.tensor([[self.env.action_space.sample()]], dtype=torch.int64)
            return a

    def encode_world(self, state, prior_action):
        with torch.no_grad():
            prior_action = prior_action.type(torch.int64).squeeze()
            prior_action_encode = torch.eye(self.action_size)[prior_action]
            prior_action_encode = prior_action_encode.type(torch.float32).to(device)
            mb_state = state.unsqueeze(1)
            prior_action_encode = prior_action_encode.unsqueeze(0).unsqueeze(1)
            _, state_embed = self.embeding(mb_state, prior_action_encode, self.hidden_state)
        self.hidden_state = state_embed
        state_embed = state_embed[1].view(state.shape[0], -1)
        obs = torch.cat([state, state_embed.detach()], dim=1).to(device)
        return obs

    def tick(self):
        state = self.env.observe().to(device)
        #create embed repr of world by predicting action result
        obs = self.encode_world(state, self.prior_action).to(device)
        # Select and perform an action
        action = self.select_action(obs)
        assert action.dtype == torch.int64, str(action.dtype)
        next_state, reward, episode_over, info = self.env.step(int(action.item()))
        self.current_session = self.memory.record(
            self.current_session,
            state, action, next_state, reward, episode_over, obs)
        if episode_over:
            self.end_episode()
        # Update the target network, copying all weights and biases in DQN
        if self.steps_done % TARGET_UPDATE == 0:
            print('Loss %.4f , Gas %.2f' % (info['loss'], info['gas']))
        self.steps_done += 1

    def tick_for(self, n):
        for i in range(n):
            self.tick()

    def end_episode(self):
        self.env.reset()
        self.hidden_state = self.embeding.generate_hidden_state()
        self.prior_action = self.generate_initial_action()


class Trainer(object):
    def __init__(self, policy_net, agents, target_net=None):
        self.policy_net = policy_net
        self.target_net = target_net or copy.deepcopy(policy_net)
        self.agents = agents
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

    @property
    def envs(self):
        return [agent.env for agent in self.agents]

    @property
    def memory(self):
        return self.agents[0].memory

    @property
    def embeding(self):
        return self.agents[0].embeding

    def tick(self):
        for agent in self.agents:
            #perform action in environment
            agent.tick()
        self.optimize()

    def set_policy(self, policy_net):
        self.policy_net = policy_net
        self.target_net = copy.deepcopy(policy_net)
        for agent in self.agents:
            agent.policy_net = policy_net

    def tick_for(self, n):
        for i in range(n):
            self.tick()
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize(self):
        self.optimizer.zero_grad()
        loss = self.sample_loss()
        if loss is not None:
            loss.backward()
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

    def sample_loss(self):
        if len(self.memory.transitions) < BATCH_SIZE:
            return None
        transitions = self.memory.sample_transitions(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s.to(device) for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat([s.to(device) for s in batch.state])
        action_batch = torch.cat([a.to(device) for a in batch.action])
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device)
        #reconstruct prior memory for training policy net
                #this emits a state which can be fed into target net, but should we?
        state_action_values = self.policy_net(state_batch)
        state_action_values = state_action_values.gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE).to(device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        expected_state_action_values[torch.isnan(expected_state_action_values)] = self.envs[0].action_space.sample()
        state_action_values[torch.isnan(state_action_values)] = self.envs[0].action_space.sample()

        # Compute Huber loss
        #print(state_action_values.shape, expected_state_action_values.shape)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        _loss = float(loss.item())
        assert _loss >= 0., str((expected_state_action_values, state_action_values, loss))

        # Optimize the model
        #loss.backward()
        #for param in self.policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        return loss

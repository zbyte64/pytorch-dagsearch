import torch
from torch import nn
from torch import functional as F
import torch.optim as optim
import numpy as np
import random
import networkx as nx
import copy
from functools import lru_cache, partial
import time
from tensorboardX import SummaryWriter

from .dag import StackedGraph
from .env import *

def inf_data(dataloader):
    i = None
    while True:
        if i is None:
            i = iter(dataloader)
        try:
            yield next(i)
        except StopIteration:
            i = None


def one_hot(num_classes):
    y = torch.eye(num_classes)
    def f(labels):
        return y[torch.tensor(labels, dtype=torch.int64)]
    return f


class World(object):
    def __init__(self, graph, test_dataloader, valid_dataloader, loss_fn, initial_gas=600):
        super(World, self).__init__()
        self.initial_graph = graph
        self.initial_gas = initial_gas
        self.criterion = loss_fn
        self.test_data = inf_data(test_dataloader)
        self.valid_data = inf_data(valid_dataloader)
        self.summary = SummaryWriter()
        self.rebuild()
        self.summary.add_graph(self.graph, next(self.test_data)[0])
        self.one_hot_node = one_hot(20)
        self.one_hot_param = one_hot(10)

    def rebuild(self):
        self.graph = copy.deepcopy(self.initial_graph)
        self.forked_graph = copy.deepcopy(self.initial_graph)
        self.graph_optimizer = optim.SGD(self.graph.parameters(), lr=0.1, momentum=0.9)
        self.forked_graph_optimizer = optim.SGD(self.forked_graph.parameters(), lr=0.0001, momentum=0.9)
        self._graph_loss = 0.0
        self._forked_graph_loss = 0.0
        self.cooldown = 80
        self.node_index = 0
        self.cell_index = 0
        self.param_index = 0
        self.input_index = -1
        self.param_state = torch.zeros(4)
        self.gas = self.initial_gas
        self.current_loss = 0.
        self.lowest_loss = 0.
        self.ticks = 0

    @property
    def active_nodes(self):
        if not hasattr(self.graph, 'stack'):
            return list(self.graph.nodes.values())
        return self.graph.stack[-1]

    @property
    def input_nodes(self):
        return self.graph.nodes

    @property
    def current_node(self):
        return self.active_nodes[self.node_index]

    @property
    def node_key(self):
        if not hasattr(self.graph, 'stack'):
            return str(self.node_index)
        return str((len(self.graph.stack) - 1) * len(self.initial_graph.nodes) + self.node_index)

    @property
    def current_cell(self):
        if len(self.current_node.cells) <= self.cell_index:
            #assert False, 'All nodes should have the same number of cells'
            self.cell_index = 0
        return self.current_node.cells[self.cell_index]

    @property
    def current_input(self):
        return self.input_nodes[str(self.input_index)]

    def get_param_options(self):
        options = [
            ('new_cell_scale1', -4, 4),
            ('new_cell_scale2', -4, 4),
            ('learning_rate', 0, 8),
            ('initializer', 0, 4),
        ]
        options.extend(self.current_cell.get_param_options())
        return options

    def get_param_state(self):
        return torch.cat((self.param_state, self.current_cell.param_state))
    
    def observe(self):
        graph_state = self.graph.observe()
        #reports visible node size
        max_volume = self.graph.in_volume
        node_state = self.current_node.observe() / max_volume
        #reports the type of cell (one hot)
        cell_state = self.current_cell.observe()
        cell_muted = self.current_node.muted_cells[self.cell_index]
        if self.input_index < len(self.current_node.muted_inputs):
            input_muted = self.current_node.is_input_muted(str(self.input_index))
        else:
            input_muted = 0.
        param_state = self.get_param_state()
        if self.param_index >= param_state.shape[0]:
            assert False, 'This should happen when paging cells'
            self.param_index = 0
        _, p_min, p_max = self.get_param_options()[self.param_index]
        #TODO convey overall network shape
        _c = lambda x: torch.sigmoid(torch.tensor(x, dtype=torch.float32))
        nav_state = torch.tensor([
            (param_state[self.param_index] - p_min) / p_max,
            cell_muted,
            input_muted,
            _c(self.current_loss),
            _c(self.lowest_loss),
            _c(self._graph_loss),
            _c(self._forked_graph_loss),
            _c(self.ticks),
            self.gas / self.initial_gas,
            self.cooldown / 100,
            _c(len(self.graph.nodes)),
            _c(len(self.graph.in_dim)),
        ], dtype=torch.float32)
        return torch.cat([
            nav_state, 
            self.one_hot_node(self.node_index),
            self.one_hot_param(self.param_index),
            self.one_hot_param(param_state[self.param_index] - p_min),
            self.one_hot_node(self.input_index),
            torch.sigmoid(graph_state), torch.sigmoid(node_state), 
            cell_state
        ]).detach()

    def perform_action(self, action_idx):
        '''
        action_idx: [0, n_actions]
        direction: [-1, 1]
        '''
        if self.cooldown > 0:
            self.cooldown -= 1
        actions = self.actions()
        #print('Action:', actions[action_idx], direction)
        return actions[action_idx](self)

    def actions_shape(self):
        actions = self.actions()
        return (len(actions), )

    @lru_cache()
    def _actions(self):
        a = [self.mov_param_up, self.mov_param_down, self.mov_add_stack, self.mov_fork, self.mov_randomize_input_adaptor, self.mov_randomize_cell]
        for f in [self.page_node, self.page_cell, self.page_param, self.page_input]:
            a.append(partial(f, direction=-1))
            a.append(partial(f, direction=1))
        return a

    def actions(self):
        actions = self._actions() + self.current_node.actions()
        return actions

    def _move(self, v, direction, _min, _max):
        v += int(direction * 2)
        if v < _min:
            v = _max
        elif v > _max:
            v = _min
        return v

    def page_node(self, world, direction):
        self.node_index = self._move(self.node_index, direction, 0, len(self.active_nodes)-1)
        #ensure input is less or equal to node index
        self.page_input(world, 0)
        self.page_cell(world, 0)

    def page_cell(self, world, direction):
        self.cell_index = self._move(self.cell_index, direction, 0, len(self.current_node.cells)-1)
        #refresh param position
        self.page_param(world, 0)

    def page_param(self, world, direction):
        o = self.get_param_options()
        self.param_index = self._move(self.param_index, direction, 0, len(o)-1)

    def page_input(self, world, direction):
        m = len(self.input_nodes) - len(self.active_nodes) + self.node_index - 1
        if m < 0:
            self.input_index = -1
        else:
            self.input_index = self._move(self.input_index, direction, 0, m)

    def toggle_param(self, world, direction):
        param_index = world.param_index
        options = self.get_param_options()
        (option_name, _min, _max) = options[param_index]
        _c = self.param_state.shape[0]
        if param_index >= _c:
            param_index -= _c
            v = self.current_cell.param_state[param_index] + direction
            v = max(min(v, _max), _min)
            self.current_cell.param_state[param_index] = v
        else:
            v = self.param_state[param_index] + direction
            v = max(min(v, _max), _min)
            self.param_state[param_index] = v
        if option_name == 'learning_rate':
            lr = float(.5 ** (v + 6))
            self.adjust_learning_rate(lr)

    def mov_param_down(self, world):
        self.toggle_param(world, -1)

    def mov_param_up(self, world):
        self.toggle_param(world, 1)

    def mov_fork(self, world):
        if self.cooldown:
            return
        #TODO compare accuracy with validation set?
        #keep_current = self._graph_loss * ( (direction + 2) / 2) < (self._forked_graph_loss * 0.97)
        keep_current = self._graph_loss < (self._forked_graph_loss * 0.9999)
        if keep_current:
            reward = self._forked_graph_loss - self._graph_loss
        else:
            reward = -self.current_loss
        self.fork_graph(keep_current)
        self._forked_graph_loss = 0.0
        self._graph_loss = 0.0
        self.page_node(world, 0)
        self.cooldown = 50
        return reward

    def fork_graph(self, keep_current=False):
        print('#'*20)
        print('fork graph', keep_current)
        if keep_current:
            print('!'*20, 'Success!')
            winner = self.graph
        else:
            winner = self.forked_graph
            self.graph_optimizer = self.forked_graph_optimizer
        self.graph = winner
        self.forked_graph = copy.deepcopy(winner)
        self.forked_graph_optimizer = optim.SGD(self.forked_graph.parameters(), lr=0.0001, momentum=0.9)
        #copy gradients?
        self.forked_graph_optimizer.load_state_dict(self.graph_optimizer.state_dict())

    def adjust_learning_rate(self, lr):
        #lr = args.lr * (0.1 ** (epoch // 30))
        print('#'*30)
        print('lr', lr)
        for param_group in self.graph_optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, iterations=1):
        f_loss = 0.
        g_loss = 0.
        self.graph.to(device)
        self.forked_graph.to(device)
        for i in range(iterations):
            self.graph_optimizer.zero_grad()
            self.forked_graph_optimizer.zero_grad()
            x, y = next(self.test_data)
            vx = x.to(device)
            vy = y.to(device)

            g = -time.time()
            py = self.graph(vx)
            graph_loss = self.criterion(py, vy)
            graph_loss.backward()
            self.graph_optimizer.step()
            g += time.time()

            forked_loss = self.criterion(self.forked_graph(vx), vy)
            forked_loss.backward()
            self.forked_graph_optimizer.step()

            self._graph_loss += graph_loss.item()
            self._forked_graph_loss += forked_loss.item()

            f_loss += forked_loss.item()
            g_loss += graph_loss.item()
            self.gas -= g
            abort = False
            reset = False
            if g_loss in (float('inf'), float('nan')):
                reset = True
                abort = f_loss in (float('inf'), float('nan'))
            self.summary.add_scalar('time_taken', g, global_step=self.ticks)
            self.summary.add_scalars('loss', {
                'forked_loss': forked_loss.item(),
                'main_loss': graph_loss.item(),
            }, global_step=self.ticks)
            self.summary.add_histogram('y', y.numpy(), global_step=self.ticks)
            try:
                self.summary.add_histogram('predicted_y', py.cpu().detach().numpy(), global_step=self.ticks)
            except:
                abort = True
            #images = x[0]
            #self.summary.add_images('x', images, global_step=self.ticks)
            self.ticks += 1
            if abort or reset:
                if abort:
                    self.gas = -1.
                else:
                    self.fork_graph()
                return {
                    'delta_loss': 0.,
                    'loss': 0.,
                    'forked_loss': 0.,
                    'gas': self.gas,
                    'reward': -1.
                }
        reward = 0.
        if self.lowest_loss is None:
            self.lowest_loss = g_loss
        elif self.lowest_loss > g_loss:
            l_delta = self.lowest_loss - g_loss
            reward = 1. + l_delta
            self.lowest_loss = g_loss
            self.gas += l_delta * self.initial_gas
        if self.gas < 0:
            #add final score
            reward -= g_loss
        reward = np.tanh(reward)
        info = {
            'loss': g_loss,
            'forked_loss': f_loss,
            'gas': self.gas,
            'reward': reward
        }
        self.current_loss = g_loss
        return info

    def draw(self):
        G = nx.OrderedDiGraph()
        for ni, node in self.graph.nodes.items():
            in_node_n = 'node_%s_input' % ni
            out_node_n = 'node_%s_output' % ni
            G.add_node(in_node_n, in_dim=node.in_dim)
            G.add_node(out_node_n, out_dim=node.out_dim)
            for ii, in_node in node.in_node_adapters.items():
                if not node.is_input_muted(ii):
                    G.add_edge('node_%s_output' % ii, in_node_n)
            for ic, cell in enumerate(node.cells):
                if node.muted_cells[ic] < 0:
                    c_n = 'node_%s_cell_%s' % (ni, cell.__class__.__name__)
                    G.add_node(c_n, **cell.get_param_dict())
                    G.add_edge(in_node_n, c_n)
                    G.add_edge(c_n, out_node_n)
        return G

    def mov_add_stack(self, world):
        if self.cooldown:
            return
        if not self.current_loss or self.current_loss > 1.0:
            return
        print('#'*20)
        print('expanding')
        self.graph.expand()
        self.graph_optimizer = optim.SGD(self.graph.parameters(), lr=0.1, momentum=0.9)
        self.fork_graph(keep_current=True)
        self.cooldown = 50

    def mov_randomize_input_adaptor(self, world):
        key = str(world.input_index)
        if key in self.current_node.in_node_adapters:
            self.current_node.in_node_adapters[key].apply(self.init_weights)
    
    def mov_randomize_cell(self, world):
        '''
        Randomize the weights
        '''
        cell = self.current_cell
        cell.apply(self.init_weights)
    
    def init_weights(self, m):
        f = {
            0:nn.init.kaiming_normal_,
            1:nn.init.kaiming_uniform_,
            2:nn.init.xavier_uniform_,
            3:nn.init.xavier_normal_,
        }[int(self.param_state[3])]
        for k in ['weight', 'weights']:
            if hasattr(m, k):
                w = getattr(m, k)
                if len(w.shape) > 1:
                    f(w)
        if hasattr(m, 'bias'):
            nn.init.constant_(m.bias, 0.)
        
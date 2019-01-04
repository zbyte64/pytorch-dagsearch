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


def inf_data(dataloader):
    i = None
    while True:
        if i is None:
            i = iter(dataloader)
        try:
            yield next(i)
        except StopIteration:
            i = None


class World(object):
    def __init__(self, graph, test_dataloader, valid_dataloader, loss_fn, initial_gas=600):
        super(World, self).__init__()
        self.initial_graph = graph
        self.initial_gas = initial_gas
        self.rebuild()
        self.criterion = loss_fn
        self.test_data = inf_data(test_dataloader)
        self.valid_data = inf_data(valid_dataloader)

    def rebuild(self):
        self.graph = copy.deepcopy(self.initial_graph)
        self.forked_graph = copy.deepcopy(self.initial_graph)
        self.graph_optimizer = optim.SGD(self.graph.parameters(), lr=0.1, momentum=0.9)
        self.graph_optimizer.zero_grad()
        self.forked_graph_optimizer = optim.SGD(self.forked_graph.parameters(), lr=0.1, momentum=0.9)
        self.forked_graph_optimizer.zero_grad()
        self._graph_loss = 0.0
        self._forked_graph_loss = 0.0
        self.cooldown = 0
        self.node_index = 0
        self.cell_index = 0
        self.param_index = 0
        self.input_index = 0
        self.param_state = torch.zeros(3)
        self.gas = self.initial_gas
        self.negative_entropy = 0.
        self.current_loss = 0.

    @property
    def current_node(self):
        return self.graph.nodes[self.node_key]

    @property
    def current_cell(self):
        if len(self.current_node.cells) <= self.cell_index:
            #assert False, 'All nodes should have the same number of cells'
            self.cell_index = 0
        return self.current_node.cells[self.cell_index]

    @property
    def current_input(self):
        return self.graph.nodes[self.input_key]

    @property
    def input_key(self):
        return list(self.graph.nodes.keys())[self.input_index]

    @property
    def node_key(self):
        return list(self.graph.nodes.keys())[self.node_index]

    def get_param_options(self):
        options = [
            ('new_cell_scale1', -4, 4),
            ('new_cell_scale2', -4, 4),
            ('learning_rate', 0, 8),
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
            input_muted = self.current_node.is_input_muted(self.input_key)
        else:
            input_muted = 0.
        param_state = self.get_param_state()
        if self.param_index >= param_state.shape[0]:
            assert False, 'This should happen when paging cells'
            self.param_index = 0
        _, p_min, p_max = self.get_param_options()[self.param_index]
        #TODO convey overall network shape
        nav_state = torch.FloatTensor([
            (param_state[self.param_index] - p_min) / p_max,
            cell_muted,
            input_muted,
            self.current_loss,
            self.negative_entropy,
            self.gas / self.initial_gas,
            self.cooldown / 20,
            len(self.graph.nodes) / 10,
            self.node_index / len(self.graph.nodes),
            self.cell_index / len(self.graph.cell_types),
            self.param_index / param_state.shape[0],
            self.input_index / len(self.graph.nodes),
        ])
        return torch.cat([nav_state, graph_state, node_state, cell_state]).detach()

    def perform_action(self, action_idx):
        '''
        action_idx: [0, n_actions]
        direction: [-1, 1]
        '''
        if self.cooldown:
            self.cooldown -= 1
        actions = self.actions()
        #print('Action:', actions[action_idx], direction)
        return actions[action_idx](self)

    def actions_shape(self):
        actions = self.actions()
        return (len(actions), )

    @lru_cache()
    def nav_actions(self):
        a = [self.mov_fork, self.mov_param_up, self.mov_param_down]
        for f in [self.page_node, self.page_cell, self.page_param, self.page_input]:
            a.append(partial(f, direction=-1))
            a.append(partial(f, direction=1))
        return a

    def actions(self):
        nav_actions = self.nav_actions()
        actions = nav_actions + self.current_node.actions() + self.current_cell.actions()
        return actions

    def _move(self, v, direction, _min, _max):
        v += int(direction * 2)
        if v < _min:
            v = _max
        elif v > _max:
            v = _min
        return v

    def page_node(self, world, direction):
        self.node_index = self._move(self.node_index, direction, 0, len(self.graph.nodes)-1)
        #ensure input is less or equal to node index
        self.page_input(world, 0)
        self.page_cell(world, 0)

    def page_cell(self, world, direction):
        self.cell_index = self._move(self.cell_index, direction, 0, len(self.graph.cell_types)-1)
        #refresh param position
        self.page_param(world, 0)

    def page_param(self, world, direction):
        o = self.get_param_options()
        self.param_index = self._move(self.param_index, direction, 0, len(o)-1)

    def page_input(self, world, direction):
        self.input_index = self._move(self.input_index, direction, 0, self.node_index)

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
        reward = -1.
        if keep_current:
            reward = 1.
        self.fork_graph(keep_current)
        self.page_node(world, 0)
        self.cooldown = 20
        return reward

    def fork_graph(self, keep_current=False):
        print('#'*20)
        print('fork graph', keep_current)
        if keep_current:
            winner = self.graph
        else:
            winner = self.forked_graph
            self.graph_optimizer = self.forked_graph_optimizer
        self.graph = winner
        self.forked_graph = copy.deepcopy(winner)
        self.forked_graph_optimizer = optim.SGD(self.forked_graph.parameters(), lr=0.1, momentum=0.9)
        #copy gradients?
        self.forked_graph_optimizer.load_state_dict(self.graph_optimizer.state_dict())
        self._forked_graph_loss = 0.0
        self._graph_loss = 0.0

    def adjust_learning_rate(self, lr):
        #lr = args.lr * (0.1 ** (epoch // 30))
        print('#'*30)
        print('lr', lr)
        for param_group in self.graph_optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, iterations=1):
        f_loss = 0.
        g_loss = 0.
        for i in range(iterations):
            x, y = next(self.test_data)

            g = -time.time()
            graph_loss = self.criterion(self.graph(x), y)
            graph_loss.backward()
            self.graph_optimizer.step()
            g += time.time()
            self._graph_t_size = g

            forked_loss = self.criterion(self.forked_graph(x), y)
            forked_loss.backward()
            self.forked_graph_optimizer.step()

            self._graph_loss += graph_loss.item()
            self._forked_graph_loss += forked_loss.item()

            #update energy
            volume = sum(map(lambda x: np.prod(x.size()), self.graph.parameters()))
            self.negative_entropy += np.log(volume)
            f_loss += forked_loss.item()
            g_loss += graph_loss.item()
            self.gas -= g
        self.current_loss = g_loss
        return (g_loss, f_loss)

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
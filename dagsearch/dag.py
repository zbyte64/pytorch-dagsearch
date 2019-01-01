import torch
from torch import nn
from torch import functional as F
import torch.optim as optim
import numpy as np
import random
import networkx as nx
import copy


flatten = lambda x: x.view(x.shape[0], -1)

class Identity(nn.Module):
    def forward(self, x):
        return x

identity = Identity()

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(input.shape[0], *self.shape)


class Interpolate(nn.Module):
    def __init__(self, size):
        super(Interpolate, self).__init__()
        self.size = size

    def forward(self, input):
        return nn.functional.interpolate(input, self.size)


class Node(nn.Module):
    def __init__(self, in_nodes, cell_types, in_dim, out_dim, channel_dim=1):
        super(Node, self).__init__()
        _a = (in_dim, out_dim, channel_dim)
        self.cells = nn.ModuleList([s(*_a) for s in filter(lambda s: s.valid(*_a), cell_types)])
        assert self.cells, str(_a)
        self.muted_cells = torch.ones(len(self.cells))
        self.muted_cells[random.randint(0, len(self.cells)-1)] = -1
        self.muted_inputs = torch.ones(len(in_nodes))
        if in_nodes:
            self.muted_inputs[-1] = -1
        self.in_dim = in_dim
        self.in_volume = np.prod(in_dim)
        self.out_dim = out_dim
        self.out_volume = np.prod(out_dim)
        self.channel_dim = channel_dim
        self.in_nodes = in_nodes
        self.in_node_adapters = nn.ModuleList(
            [self.create_node_adapter(n) for n in self.in_nodes]
        )

    def _register_input(self, in_node, drop_out=None):
        self.in_nodes.append(in_node)
        adaptor = self.create_node_adapter(in_node)
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.uniform_(m.weight)
            if type(m) == nn.Conv2d:
                nn.init.xavier_uniform(m.weight)
        adaptor.apply(init_weights)
        if drop_out is not None:
            adaptor = nn.Sequential(adaptor, nn.Dropout(drop_out))
        self.in_node_adapters.append(adaptor)
        self.muted_inputs = torch.ones(len(self.in_nodes))
        self.muted_inputs[-1] = -1

    def create_node_adapter(self, in_node):
        if in_node.out_dim == self.in_dim:
            return identity
        #TODO configurable noise (that grows with distance?)
        #TODO node may provide different adaptor (convolution?)
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        def conv2d_volume(filters, size, kernel_size, stride):
            s = conv2d_size_out(size, kernel_size, stride)
            return (s**2) * filters

        if len(self.in_dim) == 3 and len(in_node.out_dim) == 3:
            #TODO convolution up or down sample
            dh = in_node.out_dim[2] - self.in_dim[2]
            dw = in_node.out_dim[1] - self.in_dim[1]
            transforms = []
            #adapt size
            if dw or dh:
                transforms.append(Interpolate(self.in_dim[1:]))
            #adapt channels
            #if in_node.out_dim[0] != self.in_dim[0]:
            transforms.append(nn.Conv2d(in_node.out_dim[0], self.in_dim[0], 1, 1))
            transforms.append(nn.BatchNorm2d(self.in_dim[0]))
            return torch.nn.Sequential(*transforms)
        if len(self.in_dim) == 1 and len(in_node.out_dim) == 3:
            #squash
            strides = 1
            kernel_size = 1
            filters = 1
            new_h = conv2d_size_out(in_node.out_dim[1], kernel_size, strides)
            new_volume = int(new_h) ** 2 * filters
            if new_volume:
                return nn.Sequential(
                    nn.Conv2d(in_node.out_dim[0], 1, kernel_size, strides),
                    View(new_volume),
                    nn.Linear(new_volume, self.in_volume),
                    nn.BatchNorm1d(self.in_volume),
                    View(*self.in_dim)
                )

        return nn.Sequential(
            View(in_node.out_volume),
            nn.Linear(in_node.out_volume, self.in_volume),
            nn.BatchNorm1d(self.in_volume),
            View(*self.in_dim)
        )

    def is_input_muted(self, idx):
        return self.muted_inputs[idx] > 0 and idx < len(self.in_nodes) - 1

    def forward(self, x):
        if self.in_nodes:
            x_avg = []
            for i, x_i in enumerate(x):
                if not self.is_input_muted(i):
                    x_i = self.in_node_adapters[i](x_i)
                    assert x_i.shape[1:] == self.in_dim, '%s != %s (from %s)' % (x_i.shape, self.in_dim, x[i].shape)
                    x_avg.append(x_i)
            if len(x_avg) > 1:
                x = torch.sum(torch.stack(x_avg, dim=self.channel_dim), dim=self.channel_dim)
            else:
                x = x_avg[0]
        else:
            assert x.shape
        assert x.shape[1:] == self.in_dim
        #print('Node connect:', x.shape, self.in_dim, self.out_dim)
        active_sensors = filter(lambda e: self.muted_cells[e[0]], enumerate(self.cells))
        active_sensors = list(active_sensors)
        sensor_outputs = []
        for i, f in active_sensors:
            try:
                sensor_outputs.append(f(x))
            except Exception as error:
                print('Failed:', f, f.get_param_dict(), f.in_dim, f.out_dim)
                print(f.get_param_options())
                raise
        matched_outputs = sensor_outputs #list(filter(lambda z: z.shape[1:] == self.out_dim, sensor_outputs))
        assert len(matched_outputs)
        if len(matched_outputs) > 1:
            out = torch.stack(matched_outputs, dim=self.channel_dim)
            out = torch.sum(out, dim=self.channel_dim)
        else:
            out = matched_outputs[0]
        assert out.shape[0] == x.shape[0], str(out.shape)
        assert out.shape[1:] == self.out_dim, '%s != %s' % (out.shape, self.out_dim)
        return out

    def observe(self):
        return torch.FloatTensor([self.in_volume, self.out_volume])

    def actions(self):
        return [self.toggle_cell, self.toggle_input]

    def toggle_cell(self, world, direction):
        cell_index = world.cell_index
        if cell_index < len(self.muted_cells):
            self.muted_cells = torch.ones_like(self.muted_cells)
            self.muted_cells[cell_index] = -1

    def toggle_input(self, world, direction):
        index = world.input_index
        if index < len(self.muted_inputs):
            self.muted_inputs = torch.ones_like(self.muted_inputs)
            self.muted_inputs[index] = -1
            #can't toggle the last input
            self.muted_inputs[len(self.muted_inputs)-1] = -1

class Graph(nn.Module):
    def __init__(self, cell_types, in_dim, out_dim, channel_dim=1):
        super(Graph, self).__init__()
        self.cell_types = cell_types
        self.in_dim = in_dim
        self.in_volume = np.prod(in_dim)
        self.out_dim = out_dim
        self.out_volume = np.prod(out_dim)
        self.channel_dim = channel_dim
        self.nodes = list()
        self.prior_nodes = nn.ModuleList()
        self.output_node = self.create_output_node()

    def create_output_node(self, cell_types=None):
        cell_types = cell_types or self.cell_types
        in_nodes = list(self.prior_nodes)
        node = Node(in_nodes=in_nodes, in_dim=self.out_dim, out_dim=self.out_dim, channel_dim=self.channel_dim, cell_types=cell_types)
        self.nodes.append(node)
        return node

    def create_node(self, out_dim, cell_types=None):
        cell_types = cell_types or self.cell_types
        in_nodes = list(self.prior_nodes)
        if in_nodes:
            in_dim = in_nodes[-1].out_dim
        else:
            in_dim = self.in_dim
        node = Node(in_nodes=in_nodes, in_dim=in_dim, out_dim=out_dim, channel_dim=self.channel_dim, cell_types=cell_types)
        self.prior_nodes.append(node)
        self.nodes.insert(-1, node)
        #priors get spiked with noise
        p = max(.9 - len(self.prior_nodes) / 9, 0.)
        print(p)
        self.output_node._register_input(node, drop_out=p)
        return node

    def pop_last_node(self):
        _s = lambda t: nn.ModuleList(t[:-1])
        self.prior_nodes = _s(self.prior_nodes)
        self.nodes.pop(-2)
        self.output_node.in_nodes = list(self.prior_nodes)
        self.output_node.in_node_adapters = _s(self.output_node.in_node_adapters)
        self.output_node.muted_inputs = self.output_node.muted_inputs[:-1]
        self.output_node.muted_inputs[-1] = -1

    def observe(self):
        return torch.FloatTensor([self.in_volume, self.out_volume, len(self.nodes)])

    def forward(self, x):
        outputs = []
        for node in self.prior_nodes:
            #print('connecting node:', node.in_dim, node)
            if outputs:
                inputs = outputs
            else:
                inputs = x
            outputs.append(node(inputs))
        x = self.output_node(outputs)
        return x


class World(object):
    def __init__(self, graph):
        super(World, self).__init__()
        self.initial_graph = graph
        self.rebuild()

    def rebuild(self):
        self.graph = copy.deepcopy(self.initial_graph)
        self.forked_graph = copy.deepcopy(self.initial_graph)
        self.graph_optimizer = optim.SGD(self.graph.parameters(), lr=0.001, momentum=0.9)
        self.graph_optimizer.zero_grad()
        self.forked_graph_optimizer = optim.SGD(self.forked_graph.parameters(), lr=0.001, momentum=0.9)
        self.forked_graph_optimizer.zero_grad()
        self._graph_loss = 0.0
        self._forked_graph_loss = 0.0
        self.cooldown = 0
        self.node_index = 0
        self.cell_index = 0
        self.param_index = 0
        self.input_index = 0
        self.gas = 1000000.
        self.negative_entropy = 0.

    @property
    def current_node(self):
        return self.graph.nodes[self.node_index]

    @property
    def current_cell(self):
        if len(self.current_node.cells) <= self.cell_index:
            #assert False, 'All nodes should have the same number of cells'
            self.cell_index = 0
        return self.current_node.cells[self.cell_index]

    @property
    def current_input(self):
        return self.graph.nodes[self.input_index]

    def observe(self):
        graph_state = self.graph.observe()
        #reports visible node size
        max_volume = max(self.graph.in_volume, self.graph.out_volume)
        node_state = self.current_node.observe() / max_volume
        #reports the type of cell (one hot)
        cell_state = self.current_cell.observe()
        cell_muted = self.current_node.muted_cells[self.cell_index]
        if self.input_index < len(self.current_node.muted_inputs):
            input_muted = self.current_node.is_input_muted(self.input_index)
        else:
            input_muted = 0.
        param_state = self.current_cell.param_state
        if self.param_index >= param_state.shape[0]:
            assert False, 'This should happen when paging cells'
            self.param_index = 0
        _, p_min, p_max = self.current_cell.get_param_options()[self.param_index]
        #TODO convey overall network shape
        nav_state = torch.FloatTensor([
            self.graph.in_volume / max_volume,
            self.graph.out_volume / max_volume,
            (param_state[self.param_index] - p_min) / p_max,
            cell_muted,
            input_muted,
            self.gas,
            self.cooldown / 20,
            len(self.graph.nodes) / 10,
            self.node_index / len(self.graph.nodes),
            self.cell_index / len(self.graph.cell_types),
            self.param_index / param_state.shape[0],
            self.input_index / len(self.graph.nodes),
        ])
        return torch.cat([nav_state, graph_state, node_state, cell_state])

    def perform_action(self, action_idx, direction):
        '''
        action_idx: [0, n_actions]
        direction: [-1, 1]
        '''
        if self.cooldown:
            self.cooldown -= 1
        actions = self.actions()
        #print('Action:', actions[action_idx], direction)
        return actions[action_idx](self, direction)

    def actions_shape(self):
        actions = self.actions()
        return (len(actions) + 1, )

    def actions(self):
        nav_actions = [self.mov_node, self.mov_cell, self.mov_param, self.mov_input, self.mov_fork, self.mov_add_node, self.mov_pop_node]
        actions = nav_actions + self.current_node.actions() + self.current_cell.actions()
        return actions

    def _move(self, v, direction, _min, _max):
        v += int(direction * 2)
        if v < _min:
            v = _max
        elif v > _max:
            v = _min
        return v

    def mov_node(self, world, direction):
        self.node_index = self._move(self.node_index, direction, 0, len(self.graph.nodes)-1)
        #ensure input is less or equal to node index
        self.mov_input(world, 0)
        self.mov_cell(world, 0)

    def mov_cell(self, world, direction):
        self.cell_index = self._move(self.cell_index, direction, 0, len(self.graph.cell_types)-1)
        #refresh param position
        self.mov_param(world, 0)

    def mov_param(self, world, direction):
        self.param_index = self._move(self.param_index, direction, 0, self.current_cell.param_state.shape[0]-1)

    def mov_input(self, world, direction):
        self.input_index = self._move(self.input_index, direction, 0, self.node_index)

    def mov_fork(self, world, direction):
        if self.cooldown:
            return
        #TODO compare accuracy with validation set?
        #keep_current = self._graph_loss * ( (direction + 2) / 2) < (self._forked_graph_loss * 0.97)
        keep_current = self._graph_loss < (self._forked_graph_loss * 0.9999)
        reward = -1.
        if keep_current:
            reward = 1.
        self.fork_graph(keep_current)
        self.mov_node(world, 0)
        self.cooldown = 20
        return reward

    def mov_add_node(self, world, direction):
        if self.cooldown:
            return
        if len(world.graph.nodes) > 9:
            return
        last_prior = world.graph.prior_nodes[-1]
        prev_dim = last_prior.out_dim
        n = (direction + 1) * 3 + .1 #(.1, 3.1)
        scale = lambda v: int(min(max(n * v, 1), 300))
        if len(prev_dim) == 3:
            wh = max(prev_dim[1] + int(direction*-2), 1)
            out_dim = [scale(prev_dim[0]), wh, wh]
        else:
            out_dim = list(map(scale, prev_dim))
        print('#'*20)
        print('create node', out_dim, n)
        try:
            world.graph.create_node(tuple(out_dim))
        except AssertionError as e:
            print(e)
            return -.1
        self.mov_node(world, 0)
        self.cooldown = 20
        self.graph_optimizer = optim.SGD(self.graph.parameters(), lr=0.001, momentum=0.9)
        #return -len(world.graph.prior_nodes) / 10 - n

    def mov_pop_node(self, world, direction):
        if self.cooldown:
            return
        if len(world.graph.prior_nodes) < 2:
            return
        print('#'*20)
        print('pop last node')
        world.graph.pop_last_node()
        self.mov_node(world, 0)
        self.cooldown = 20
        self.graph_optimizer = optim.SGD(self.graph.parameters(), lr=0.001, momentum=0.9)
        #return len(world.graph.prior_nodes) / 20

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
        self.forked_graph_optimizer = optim.SGD(self.forked_graph.parameters(), lr=0.001, momentum=0.9)
        #copy gradients?
        self.forked_graph_optimizer.load_state_dict(self.graph_optimizer.state_dict())
        self._forked_graph_loss = 0.0
        self._graph_loss = 0.0

    def optimize(self, graph_loss, forked_loss):
        self._graph_loss += graph_loss.item()
        self._forked_graph_loss += forked_loss.item()
        graph_loss.backward()
        self.graph_optimizer.step()
        forked_loss.backward()
        self.forked_graph_optimizer.step()
        #update energy
        volume = sum(map(lambda x: np.prod(x.size()), self.graph.parameters()))
        self.negative_entropy += np.log(volume)
        self.gas -= volume * self._graph_loss

    def draw(self):
        G = nx.OrderedDiGraph()
        for ni, node in enumerate(self.graph.nodes):
            in_node_n = 'node_%s_input' % ni
            out_node_n = 'node_%s_output' % ni
            G.add_node(in_node_n, in_dim=node.in_dim)
            G.add_node(out_node_n, out_dim=node.out_dim)
            for ii, in_node in enumerate(node.in_nodes):
                if not node.is_input_muted(ii):
                    G.add_edge('node_%s_output' % ii, in_node_n)
            for ic, cell in enumerate(node.cells):
                if node.muted_cells[ic] < 0:
                    c_n = 'node_%s_cell_%s' % (ni, cell.__class__.__name__)
                    G.add_node(c_n, **cell.get_param_dict())
                    G.add_edge(in_node_n, c_n)
                    G.add_edge(c_n, out_node_n)
        return G

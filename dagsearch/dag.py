import torch
from torch import nn
from torch import functional as F
import torch.optim as optim
import numpy as np
import random


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


class Connector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Connector, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_volume = np.prod(self.in_dim)
        self.out_volume = np.prod(self.out_dim)
        self.model = self.make_model()

    def make_model(self):
        if self.out_dim == self.in_dim:
            return Identity()
        #TODO configurable noise (that grows with distance?)
        #TODO node may provide different adaptor (convolution?)
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        def conv2d_volume(filters, size, kernel_size, stride):
            s = conv2d_size_out(size, kernel_size, stride)
            return (s**2) * filters

        #transform to out size
        transforms = []
        if len(self.in_dim) == 3:
            if len(self.out_dim) == 3:
                #adapt size
                if self.out_dim[1:] != self.in_dim[1:]:
                    transforms.append(Interpolate(self.out_dim[1:]))
                transforms.append(nn.Conv2d(self.in_dim[0], self.out_dim[0], 1, 1))
            else:
                new_volume = self.in_volume // self.in_dim[0]
                if new_volume != self.in_volume:
                    transforms.extend([
                        nn.Conv2d(self.in_dim[0], 1, 1, 1),
                        View(new_volume),
                        nn.Linear(new_volume, self.out_volume),
                    ])
        else:
            transforms.extend([
                View(self.in_volume),
                nn.Linear(self.in_volume, self.out_volume),
            ])
        #normalize
        if len(self.out_dim) == 3:
            transforms.append(nn.BatchNorm2d(self.out_dim[0]))
        elif len(self.out_dim) == 2:
            transforms.append(nn.BatchNorm1d(self.out_dim[0]))
        elif len(self.out_dim) == 1:
            transforms.append(nn.LayerNorm(self.out_dim))
        transforms.append(View(*self.out_dim))
        return torch.nn.Sequential(*transforms)

    def forward(self, x):
        return self.model(x)


class Node(nn.Module):
    def __init__(self, in_nodes, cell_types, in_dim, out_dim, channel_dim=1):
        super(Node, self).__init__()
        _a = (in_dim, out_dim, channel_dim)
        self.cells = nn.ModuleList([s(*_a) for s in filter(lambda s: s.valid(*_a), cell_types)])
        assert self.cells, str(_a)
        self.muted_cells = torch.ones(len(self.cells))
        self.muted_cells[random.randint(0, len(self.cells)-1)] = -1
        self.muted_inputs = dict()#nn.ModuleDict()
        self.in_dim = in_dim
        self.in_volume = np.prod(in_dim)
        self.out_dim = out_dim
        self.out_volume = np.prod(out_dim)
        self.channel_dim = channel_dim
        self.in_node_adapters = nn.ModuleDict()
        for key, in_node in in_nodes.items():
            self.register_input(key, in_node)
        if in_nodes:
            self.muted_inputs[key] = -1

    def register_input(self, key, in_node_or_dim, drop_out=None):
        if isinstance(in_node_or_dim, Node):
            adaptor = self.create_node_adapter(in_node_or_dim)
        else:
            adaptor = Connector(in_node_or_dim, self.in_dim)
        if drop_out is not None:
            adaptor = nn.Sequential(adaptor, nn.Dropout(drop_out))
        self.in_node_adapters[key] = adaptor
        self.muted_inputs[key] = torch.tensor(-1) if key == 'input' else torch.tensor(1)

    def create_node_adapter(self, in_node):
        a = Connector(in_node.out_dim, self.in_dim)
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.uniform_(m.weight)
            if type(m) == nn.Conv2d:
                nn.init.xavier_uniform(m.weight)
        a.apply(init_weights)
        return a

    def is_input_muted(self, node_id):
        return self.muted_inputs.get(node_id, 1) > 0

    def forward(self, x_dict):
        x_ts = []
        for node_id, a in self.in_node_adapters.items():
            if not self.is_input_muted(node_id):
                x_i = a(x_dict[node_id])
                assert x_i.shape[1:] == self.in_dim, '%s != %s (from %s)' % (x_i.shape, self.in_dim, node_id)
                x_ts.append(x_i)
        if len(x_ts) > 1:
            x = torch.sum(torch.stack(x_ts, dim=self.channel_dim), dim=self.channel_dim)
        else:
            assert len(x_ts), '%s <> %s' % (x_dict.keys(), self.in_node_adapters.keys())
            x = x_ts[0]
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

    def toggle_cell(self, world):
        cell_index = world.cell_index
        if cell_index < len(self.muted_cells):
            self.muted_cells = torch.ones_like(self.muted_cells)
            self.muted_cells[cell_index] = -1

    def toggle_input(self, world):
        key = world.input_key
        if key in self.muted_inputs:
            self.muted_inputs[key] *= 1


class Graph(nn.Module):
    def __init__(self, cell_types, in_dim, channel_dim=1):
        super(Graph, self).__init__()
        self.cell_types = cell_types
        self.in_dim = in_dim
        self.in_volume = np.prod(in_dim)
        self.channel_dim = channel_dim
        self.nodes = nn.ModuleDict()

    def create_node(self, out_dim, cell_types=None, key=None):
        if key is None:
            key = str(len(self.nodes))
        cell_types = cell_types or self.cell_types
        in_nodes = dict(self.nodes)
        if in_nodes:
            in_dim = list(in_nodes.values())[-1].out_dim
        else:
            in_dim = self.in_dim
        node = Node(in_nodes=in_nodes, in_dim=in_dim, out_dim=out_dim, channel_dim=self.channel_dim, cell_types=cell_types)
        self.register_node(key, node)
        return node

    def register_node(self, key, node):
        if not len(self.nodes):
            node.register_input('input', self.in_dim)
        self.nodes[key] = node

    def observe(self):
        return torch.FloatTensor([self.in_volume, len(self.nodes)])

    def forward(self, x, outputs=None):
        if outputs is None:
            outputs = {'input': x}
        else:
            #otherwise we have been suplied input from another network
            assert 'input' in x
        for key, node in self.nodes.items():
            n_x = node(outputs)
            outputs[key] = n_x
        x = n_x
        return x

    @classmethod
    def from_model(cls, model):
        '''
        Converts an existing model into a graph
        '''
        pass


class StackedGraph(Graph):
    '''
    https://arxiv.org/pdf/1706.03256.pdf

    A DAG graph that can add layers,
    freezing the previous layer and connecting the new
    '''
    def __init__(self, make_layer):
        super(StackedGraph, self).__init__()
        self.make_layer = make_layer
        self.stack = list()
        self.expand()

    @classmethod
    def from_graph(cls, graph):
        '''
        Convert an existing graph into a stackable graph,
        each new layer is a copy of the graph but randomized
        '''
        g = copy.deepcopy(graph)
        def make_layer(self, name):
            nodes = list()
            for key, node in g.nodes.items():
                pass
            return nodes
        return cls(make_layer)

    def expand(self):
        name = str(len(self.stack))
        prior_keys = list(self.nodes.keys())[-len(self.stack[-1]):]
        new_nodes = self.make_layer(self, name)
        #new_layer.randomize_state()
        previous_frozen_nodes = list(map(self._freeze, self.stack[-1]))
        self.stack.append(new_nodes)
        #connect nodes
        for key, p_node, t_node in zip(prior_keys, previous_frozen_nodes, new_nodes):
            t_node.register_input(key, p_node)

    def _freeze(self, g):
        for param in g.parameters():
            param.requires_grad = False
        return g

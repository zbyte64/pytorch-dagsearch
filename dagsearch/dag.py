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

    def toggle_cell(self, world):
        cell_index = world.cell_index
        if cell_index < len(self.muted_cells):
            self.muted_cells = torch.ones_like(self.muted_cells)
            self.muted_cells[cell_index] = -1

    def toggle_input(self, world):
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

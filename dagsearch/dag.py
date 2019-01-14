import torch
from torch import nn
from torch import functional as F
import torch.optim as optim
import numpy as np
import random
import copy
from collections import OrderedDict
import networkx


class Identity(nn.Module):
    def forward(self, x):
        return x


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
    '''
    Allows for (residual) connecting of layers with different sizes
    Transformation results should be additive
    '''
    def __init__(self, in_dim, out_dim):
        super(Connector, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_volume = np.prod(self.in_dim)
        self.out_volume = np.prod(self.out_dim)
        self.model = self.make_model()
        self.add_module('model', self.model)
        self._mean, self._std_dev = 0.0, 0.0

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
                elif self.in_volume != self.out_volume:
                    transforms.extend([
                        View(self.in_volume),
                        nn.Linear(self.in_volume, self.out_volume),
                    ])
        else:
            transforms.extend([
                View(self.in_volume),
                nn.Linear(self.in_volume, self.out_volume),
            ])
        transforms.append(View(*self.out_dim))
        #normalize
        if len(self.out_dim) == 3:
            transforms.append(nn.BatchNorm2d(self.out_dim[0]))
        elif len(self.out_dim) == 2:
            transforms.append(nn.BatchNorm1d(self.out_dim[0]))
        elif len(self.out_dim) == 1:
            transforms.append(nn.LayerNorm(self.out_dim))
        return torch.nn.Sequential(*transforms)

    def forward(self, x):
        out = self.model(x)
        self._mean = torch.mean(out).item()
        self._std_dev = torch.std(out).item()
        return out


class HyperCell(nn.Module):
    '''
    Special type of node that enables Efficient Network Architecture Search
    '''
    def __init__(self, cell_types, in_dim, out_dim, channel_dim=1):
        super(HyperCell, self).__init__()
        _a = (in_dim, out_dim, channel_dim)
        self.cells = nn.ModuleList([s(*_a) for s in filter(lambda s: s.valid(*_a), cell_types)])
        assert self.cells, str(_a)
        self.muted_cells = torch.ones(len(self.cells))
        self.muted_cells[random.randint(0, len(self.cells)-1)] = -1
        self.in_dim = in_dim
        self.in_volume = np.prod(in_dim)
        self.out_dim = out_dim
        self.out_volume = np.prod(out_dim)
        self.channel_dim = channel_dim
        self.add_module('cells', self.cells)

    def forward(self, x):
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


class Graph(nn.Module, networkx.DiGraph): #TensorGraph? TorchGraph?
    '''
    Creates a DAG computation of tunable cells that share weights
    Differs from ENAS in that preset cell types have scalable params
    Also allows for adding nodes in reverse order 
    https://arxiv.org/pdf/1802.03268.pdf
    '''
    def __init__(self, cell_types, in_dim, channel_dim=1):
        nn.Module.__init__(self)
        networkx.DiGraph.__init__(self)
        self.cell_types = cell_types
        self.in_dim = in_dim
        self.in_volume = np.prod(in_dim)
        self.channel_dim = channel_dim
        self.tensor_nodes = nn.ModuleDict()
        self.tensor_adaptors = nn.ModuleDict()
        self.add_module('tensor_nodes', self.tensor_nodes)
        self.add_module('tensor_adaptors', self.tensor_adaptors)
        self.add_node('input', out_dim=in_dim, in_dim=(0,))
    
    def named_children(self):
        from itertools import chain
        return chain(self.tensor_nodes.named_children(), self.tensor_adaptors.named_children())

    def create_hypercell(self, in_dim, out_dim=None, cell_types=None, key=None, link_previous=True):
        if key is None:
            key = str(len(self.tensor_nodes))
        if out_dim is None:
            if len(self.tensor_nodes):
                prior_in_node = list(self.tensor_nodes.values())[-1]
                out_dim = prior_in_node.in_dim
            else:
                out_dim = in_dim
        cell_types = cell_types or self.cell_types
        prior_nodes = list(self.tensor_nodes.items())
        node = HyperCell(in_dim=in_dim, out_dim=out_dim, channel_dim=self.channel_dim, cell_types=cell_types)
        self.register_node(key, node, in_dim, out_dim)
        self.register_adaptor('input', key, muteable=False, muted=False)
        if link_previous:
            for i, (k, n) in enumerate(prior_nodes):
                if i == 0:
                    self['input'][k]['muteable'] = True
                    self['input'][k]['muted'] = True
                    self.register_adaptor(key, k, muteable=False, muted=False)
                else:
                    self.register_adaptor(key, k)
        return node

    def register_node(self, key, node, in_dim, out_dim):
        assert key not in self.nodes
        self.add_node(key, in_dim=in_dim, out_dim=out_dim)
        self.tensor_nodes.update({key: node})
    
    def register_adaptor(self, src, to, muteable=True, muted=True):
        key = '%s->%s' % (src, to)
        in_dim = self.nodes[src]['out_dim']
        out_dim = self.nodes[to]['in_dim']
        adaptor = Connector(in_dim, out_dim)
        self.add_edge(src, to, muteable=muteable, muted=muted)
        self.tensor_adaptors.update({key: adaptor})
    
    def is_input_muted(self, src, to):
        return not self.has_edge(src, to) or self[src][to]['muted']

    def forward(self, x):
        assert x is not None
        outputs = {'input': x}
        for key, node in reversed(self.tensor_nodes.items()):
            #print(key, list(self.predecessors(key)))
            assert key not in outputs
            inputs = [self.tensor_adaptors['%s->%s' % (k, key)](outputs[k]) for k in self.predecessors(key) if not self.is_input_muted(k, key)]
            assert len(inputs)
            if len(inputs) > 1:
                x = torch.sum(torch.stack(inputs, dim=self.channel_dim), dim=self.channel_dim)
            else:
                x = inputs[0]
            assert x.shape[1:] == self.nodes[key]['in_dim'], str((x.shape, key, len(inputs)))
            try:
                x = node(x)
            except:
                print(outputs.keys())
                print(key)
                raise
            outputs[key] = x
        return x

    @classmethod
    def from_model(cls, model):
        '''
        Converts an existing model into a graph
        '''
        pass

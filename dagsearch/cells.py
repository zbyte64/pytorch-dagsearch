import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import random

from .env import *


def make_one_hot(labels, num_classes):
    y = torch.eye(num_classes)
    return y[labels]


def pad_to_match(x, out_shape):
    bs = x.shape[0]
    h = x.shape[2]
    w = x.shape[3]
    dh = (out_shape[1] - h)
    dw = (out_shape[2] - w)
    if dh > 0:
        pw = dw // 2
        ph = dh // 2
        x = F.pad(x, (dw-pw, pw, dh-ph, ph))
    elif dh < 0:
        x = nn.functional.interpolate(x, out_shape[1:])
    return x


CELL_TYPES = {}
def register_cell(cell_type):
    CELL_TYPES[cell_type] = len(CELL_TYPES)
    return cell_type


class BaseCell(nn.Module):
    """
    Implements a general strategy

    Define your own forward & get_param_options & __init__
    """
    def __init__(self, in_dim, out_dim, channel_dim):
        super(BaseCell, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channel_dim = channel_dim
        self.param_state = torch.zeros(len(self.get_param_options()))
        for i, (n, _min, _max) in enumerate(self.get_param_options()):
            self.param_state[i] = random.randint(_min, _max)

    @staticmethod
    def valid(in_dim, out_dim, channel_dim):
        '''
        Return whether the args are a valid cell size
        '''
        return True

    def observe(self):
        idx = CELL_TYPES.get(self.__class__, -1)
        return make_one_hot(idx, len(CELL_TYPES))

    def get_param_options(self):
        '''
        Return a list of a triples desribing the params and their ranges.
        '''
        return [] #('name', min, max))

    def get_param_dict(self):
        return {k: float(self.param_state[i]) for i, (k, _n, _x) in enumerate(self.get_param_options())}

    def actions(self):
        return [self.mov_scramble]

    def mov_scramble(self, world):
        '''
        Randomize the weights
        '''
        pass


@register_cell
class LinearCell(BaseCell):
    def __init__(self, in_dim, out_dim, channel_dim):
        super(LinearCell, self).__init__(in_dim, out_dim, channel_dim)
        self.f = nn.Linear(np.prod(in_dim), np.prod(out_dim))
        self.add_module('f', self.f)

    @staticmethod
    def valid(in_dim, out_dim, channel_dim):
        return len(out_dim) == 1 or len(in_dim) == 1

    def get_param_options(self):
        return [('activation', 0, 4)]
    
    def forward(self, x):
        params = self.get_param_dict()
        x = x.view(x.shape[0], -1)
        activation = int(params['activation'])
        a_f = {
            0: torch.relu,
            1: torch.tanh,
            2: torch.sigmoid,
            3: torch.nn.LogSoftmax(dim=1),
            4: lambda x: x,
        }[activation]
        x = a_f(self.f(x))
        return x.view(-1, *self.out_dim)

    def mov_scramble(self, world):
        nn.init.uniform_(self.f.weight)


@register_cell
class Conv2dCell(BaseCell):
    max_kernel_size = 11
    #CONSIDER: we can oversize our kernel and slice down, allowing for agent to change size or stride
    def __init__(self, in_dim, out_dim, channel_dim):
        super(Conv2dCell, self).__init__(in_dim, out_dim, channel_dim)
        self.weights = nn.Parameter(torch.ones((
            out_dim[channel_dim-1],
            in_dim[channel_dim-1],
            self.max_kernel_size, self.max_kernel_size)))
        nn.init.xavier_uniform_(self.weights)

    @staticmethod
    def valid(in_dim, out_dim, channel_dim):
        return len(in_dim) == 3 and len(out_dim) == 3 and in_dim[0] <= out_dim[0] and in_dim[1] >= out_dim[1]

    @staticmethod
    def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        from math import floor
        assert len(h_w) == 2
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
        w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
        return h, w

    def get_param_options(self):
        return [
            ('kernel', 2, self.max_kernel_size),
            ('stride', 1, 4),
            ('dilation', 1, 4),
        ]

    def forward(self, x):
        params = self.get_param_dict()
        kernel_size = int(params['kernel'])
        stride = min(int(params['stride']), kernel_size)
        dilation = int(params['dilation'])
        padding = 0
        h, w = self.conv_output_shape(x.shape[2:4], kernel_size, stride, padding, dilation)
        while h < 1:
            if stride > 1:
                stride -= 1
            elif dilation > 1:
                dilation -= 1
            elif padding > 0:
                padding -= 1
            elif kernel > 1:
                kernel -= 1
            else:
                assert False, 'Cannot fit convolution'
            h, w = self.conv_output_shape(x.shape[2:4], kernel_size, stride, padding, dilation)
        kernel = self.weights
        if kernel_size < self.max_kernel_size:
            kernel = torch.narrow(kernel, 2, 0, kernel_size)
            kernel = torch.narrow(kernel, 3, 0, kernel_size)
        x = torch.relu(F.conv2d(x, kernel, stride=stride, dilation=dilation, padding=padding))
        x = pad_to_match(x, self.out_dim)
        return x

    def mov_scramble(self, world):
        nn.init.xavier_uniform_(self.weights)


@register_cell
class Pooling2dCell(BaseCell):
    max_kernel_size = 5

    @staticmethod
    def valid(in_dim, out_dim, channel_dim):
        return len(in_dim) == 3 and len(out_dim) == 3 and in_dim[channel_dim-1] == out_dim[channel_dim-1] and in_dim[channel_dim-1] <= in_dim[1]

    def get_param_options(self):
        return [
            ('function', 0, 1),
            ('kernel', 2, self.max_kernel_size),
            ('stride', 1, 4),
            ('dilation', 1, 4),
        ]

    def forward(self, x):
        params = self.get_param_dict()
        kernel_size = int(params['kernel'])
        kernel_size = self.out_dim[self.channel_dim-1]
        stride = min(int(params['stride']), kernel_size)
        dilation = int(params['dilation'])
        if params['function'] > 0:
            x = F.max_pool2d(x, kernel_size, stride=stride, dilation=dilation)
        else:
            x = F.avg_pool2d(x, kernel_size, stride=stride)
        x = pad_to_match(x, self.out_dim)
        return x


@register_cell
class DeConv2dCell(BaseCell):
    max_kernel_size = 11
    #CONSIDER: we can oversize our kernel and slice down, allowing for agent to change size or stride
    def __init__(self, in_dim, out_dim, channel_dim):
        super(DeConv2dCell, self).__init__(in_dim, out_dim, channel_dim)
        self.weights = nn.Parameter(torch.ones((
            in_dim[channel_dim-1],
            out_dim[channel_dim-1],
            self.max_kernel_size, self.max_kernel_size)))
        nn.init.xavier_uniform_(self.weights)

    @staticmethod
    def valid(in_dim, out_dim, channel_dim):
        return len(in_dim) == 3 and len(out_dim) == 3 and in_dim[0] > out_dim[0] and in_dim[1] <= out_dim[1]

    def get_param_options(self):
        return [
            ('kernel', 2, self.max_kernel_size),
            ('stride', 1, 5),
            ('dilation', 1, 4),
        ]

    def forward(self, x):
        assert len(x.shape) == 4
        params = self.get_param_dict()
        kernel_size = int(params['kernel'])
        stride = min(int(params['stride']), kernel_size)
        dilation = int(params['dilation'])
        kernel = self.weights.to(device)
        if kernel_size < self.max_kernel_size:
            kernel = torch.narrow(kernel, 2, 0, kernel_size)
            kernel = torch.narrow(kernel, 3, 0, kernel_size)
        x = torch.relu(F.conv_transpose2d(x, kernel, stride=stride, dilation=dilation))
        x = pad_to_match(x, self.out_dim)
        return x

    def mov_scramble(self, world):
        nn.init.xavier_uniform_(self.weights)

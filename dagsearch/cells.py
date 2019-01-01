import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import random


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
        return True

    def observe(self):
        idx = CELL_TYPES.get(self.__class__, -1)
        return make_one_hot(idx, len(CELL_TYPES))

    def get_param_options(self):
        return [] #('name', min, max))

    def get_param_dict(self):
        return {k: float(self.param_state[i]) for i, (k, _n, _x) in enumerate(self.get_param_options())}

    def actions(self):
        return [self.toggle_param]

    def toggle_param(self, world, direction):
        options = self.get_param_options()
        param_index = world.param_index
        (option_name, _min, _max) = options[param_index]
        v = self.param_state[param_index] + direction
        v = max(min(v, _max), _min)
        self.param_state[param_index] = v


@register_cell
class LinearCell(BaseCell):
    def __init__(self, in_dim, out_dim, channel_dim):
        super(LinearCell, self).__init__(in_dim, out_dim, channel_dim)
        self.f = nn.Linear(np.prod(in_dim), np.prod(out_dim))

    @staticmethod
    def valid(in_dim, out_dim, channel_dim):
        return len(out_dim) == 1 or len(in_dim) == 1

    def get_param_options(self):
        return [('activation', 0, 1)]

    def forward(self, x):
        params = self.get_param_dict()
        x = x.view(x.shape[0], -1)
        activation = torch.relu
        if params['activation'] > 0:
            activation = torch.tanh
        x = activation(self.f(x))
        return x.view(-1, *self.out_dim)


@register_cell
class Conv2dCell(BaseCell):
    max_kernel_size = 11
    #CONSIDER: we can oversize our kernel and slice down, allowing for agent to change size or stride
    def __init__(self, in_dim, out_dim, channel_dim):
        super(Conv2dCell, self).__init__(in_dim, out_dim, channel_dim)
        self.weights = torch.ones((
            out_dim[channel_dim-1],
            in_dim[channel_dim-1],
            self.max_kernel_size, self.max_kernel_size), requires_grad=True)
        nn.init.xavier_uniform(self.weights)

    @staticmethod
    def valid(in_dim, out_dim, channel_dim):
        return len(in_dim) == 3 and len(out_dim) == 3 and in_dim[0] <= out_dim[0] and in_dim[1] >= out_dim[1]

    def get_param_options(self):
        max_stride = max(self.in_dim[1] // self.out_dim[1], 1)
        max_kernel_size = min(max(self.in_dim[1] // max_stride - self.out_dim[1], 1), self.max_kernel_size)
        return [
            ('kernel', 1, max_kernel_size),
            ('stride', 1, max_stride),
        ]

    def forward(self, x):
        params = self.get_param_dict()
        kernel_size = int(params['kernel'])
        stride = min(int(params['stride']), kernel_size)
        kernel = self.weights
        if kernel_size < self.max_kernel_size:
            kernel = torch.narrow(kernel, 2, 0, kernel_size)
            kernel = torch.narrow(kernel, 3, 0, kernel_size)
        x = torch.relu(F.conv2d(x, kernel, stride=stride))
        x = pad_to_match(x, self.out_dim)
        return x


@register_cell
class Pooling2dCell(BaseCell):
    max_kernel_size = 5

    @staticmethod
    def valid(in_dim, out_dim, channel_dim):
        return len(in_dim) == 3 and len(out_dim) == 3 and in_dim[channel_dim-1] == out_dim[channel_dim-1] and in_dim[channel_dim-1] <= in_dim[1]

    def get_param_options(self):
        max_stride = max(self.in_dim[1] // self.out_dim[1], 1)
        max_kernel_size = min(max(self.in_dim[1] // max_stride - self.out_dim[1], 1), self.max_kernel_size)
        return [
            ('function', 0, 1),
            ('kernel', 1, min(self.max_kernel_size, max_kernel_size)),
            ('stride', 1, max_stride),
        ]

    def forward(self, x):
        params = self.get_param_dict()
        kernel_size = int(params['kernel'])
        kernel_size = self.out_dim[self.channel_dim-1]
        stride = min(int(params['stride']), kernel_size)
        if params['function'] > 0:
            f = F.max_pool2d
        else:
            f = F.avg_pool2d
        x = f(x, kernel_size, stride=stride)
        x = pad_to_match(x, self.out_dim)
        return x


@register_cell
class DeConv2dCell(BaseCell):
    max_kernel_size = 11
    #CONSIDER: we can oversize our kernel and slice down, allowing for agent to change size or stride
    def __init__(self, in_dim, out_dim, channel_dim):
        super(DeConv2dCell, self).__init__(in_dim, out_dim, channel_dim)
        self.weights = torch.ones((
            in_dim[channel_dim-1],
            out_dim[channel_dim-1],
            self.max_kernel_size, self.max_kernel_size), requires_grad=True)
        nn.init.xavier_uniform(self.weights)

    @staticmethod
    def valid(in_dim, out_dim, channel_dim):
        return len(in_dim) == 3 and len(out_dim) == 3 and in_dim[0] > out_dim[0] and in_dim[1] <= out_dim[1]

    def get_param_options(self):
        return [
            ('kernel', 1, self.max_kernel_size),
            ('stride', 1, 5),
        ]

    def forward(self, x):
        assert len(x.shape) == 4
        params = self.get_param_dict()
        kernel_size = int(params['kernel'])
        stride = min(int(params['stride']), kernel_size)
        kernel = self.weights
        if kernel_size < self.max_kernel_size:
            kernel = torch.narrow(kernel, 2, 0, kernel_size)
            kernel = torch.narrow(kernel, 3, 0, kernel_size)
        x = torch.relu(F.conv_transpose2d(x, kernel, stride=stride))
        x = pad_to_match(x, self.out_dim)
        return x

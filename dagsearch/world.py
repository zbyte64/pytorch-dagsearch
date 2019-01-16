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
from collections import OrderedDict

from .env import *
from .scoreboard import Scoreboard


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
        if isinstance(labels, int):
            return y[labels]
        return y[labels.type(torch.int64)]
    return f


class World(object):
    def __init__(self, graph, test_dataloader, valid_dataloader, loss_fn, 
    initial_gas=600, ratcheting=True, add_threshold=.7):
        super(World, self).__init__()
        self.initial_graph = graph
        self.initial_gas = initial_gas
        self.criterion = loss_fn
        self.test_data = inf_data(test_dataloader)
        self.valid_data = inf_data(valid_dataloader)
        self.summary = SummaryWriter()
        self.rebuild()
        #self.summary.add_graph(self.graph, next(self.test_data)[0])
        self.one_hot_node = one_hot(20)
        self.one_hot_param = one_hot(10)
        self.ratcheting = ratcheting
        self.add_threshold = add_threshold
        self.scoreboard = Scoreboard(self.criterion, self.valid_data, top_k=10)

    def rebuild(self):
        self.graph = copy.deepcopy(self.initial_graph)
        self.forked_graph = copy.deepcopy(self.initial_graph)
        self.graph_optimizer = optim.SGD(self.graph.parameters(), lr=0.1, momentum=0.9)
        self.forked_graph_optimizer = optim.SGD(self.forked_graph.parameters(), lr=0.0001, momentum=0.9)
        self._graph_loss = 0.0
        self._forked_graph_loss = 0.0
        self.cooldown = 40
        self.node_index = 0
        self.cell_index = 0
        self.param_index = 0
        self.input_index = -1
        self.param_state = torch.zeros(5)
        self.gas = self.initial_gas
        self.current_loss = None
        self.lowest_loss = None
        self.initial_loss = None
        self.current_bench = None
        self.ticks = 0

    @property
    def active_nodes(self):
        return self.graph.tensor_nodes

    @property
    def input_nodes(self):
        return list(self.graph.predecessors(self.node_key))

    @property
    def node_key(self):
        return str(self.node_index)

    @property
    def current_node(self):
        return self.active_nodes[self.node_key]

    @property
    def current_cell(self):
        if not hasattr(self.current_node, 'cells'):
            return None
        if len(self.current_node.cells) <= self.cell_index:
            #assert False, 'All nodes should have the same number of cells'
            self.cell_index = 0
        return self.current_node.cells[self.cell_index]

    @property
    def current_input(self):
        return self.graph.nodes[self.input_key]
    
    @property
    def input_key(self):
        return self.input_nodes[self.input_index]

    def get_param_options(self):
        options = [
            ('new_cell_scale1', -4, 4),
            ('new_cell_scale2', -4, 4),
            ('learning_rate', 0, 8),
            ('initializer', 0, 3),
            ('training_steps', 0, 9),
        ]
        options.extend(self.current_cell.get_param_options())
        return options

    def get_param_state(self):
        return torch.cat((self.param_state, self.current_cell.param_state))
    
    def observe(self):
        #reports visible node size
        current_node = self.graph.nodes[self.node_key]
        current_input = self.graph.nodes[self.input_key]
        current_adaptor = self.graph.tensor_adaptors['%s->%s' % (self.input_key, self.node_key)]
        node_stats = lambda n: torch.stack([
            _c(len(n['in_dim'])),
            _c(np.prod(n['in_dim'])),
            _c(len(n['out_dim'])),
            _c(np.prod(n['out_dim'])),
        ])
        max_volume = self.graph.in_volume
        #reports the type of cell (one hot)
        if self.current_cell:
            cell_state = self.current_cell.observe()
            cell_muted = self.current_node.muted_cells[self.cell_index]
        else:
            cell_state = torch.zeros(len(self.graph.cell_types))
            cell_muted = 1.
        input_muted = 1. if self.graph.is_input_muted(self.input_key, self.node_key) else 0.
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
            _c(self.current_loss or 0.),
            _c(self.lowest_loss or 0.),
            _c(self._graph_loss),
            _c(self._forked_graph_loss),
            _c(self.ticks),
            self.gas / self.initial_gas,
            self.cooldown / 100,
            _c(len(self.graph.tensor_nodes)),
            _c(len(self.graph.in_dim)),
            _c(self.graph.in_volume),
            _c(current_adaptor._std_dev),
            _c(current_adaptor._mean),
            
        ], dtype=torch.float32)
        return torch.cat([
            nav_state, 
            self.one_hot_node(self.node_index),
            self.one_hot_param(self.param_index),
            self.one_hot_param(param_state[self.param_index] - p_min),
            self.one_hot_node(self.input_index),
            node_stats(current_node),
            node_stats(current_input),
            cell_state
        ]).detach()

    def perform_action(self, action_idx):
        '''
        action_idx: [0, n_actions]
        direction: [-1, 1]
        '''
        assert self.gas > 0, 'Please reset environment'
        actions = self.actions()
        #print('Action:', actions[action_idx], direction)
        self.gas -= .015
        r = actions[action_idx](self)
        if r is None:
            r = 0.
            #idle cost
            self.gas -= .15
        if self.gas <= 0:
            self.scoreboard.record(self.graph)
            #add final score
            if self.current_loss is not None:
                r += (self.initial_loss - self.current_loss) / self.initial_loss
            else:
                r -= 10.
        r = torch.tanh(torch.tensor(r)).item()
        info = {
            'loss': self.current_loss or 0.,
            'gas': self.gas,
            'reward': r
        }
        return r, info

    def actions_shape(self):
        actions = self.actions()
        return (len(actions), )

    @lru_cache()
    def actions(self):
        a = [self.mov_param_up, self.mov_param_down, self.mov_add_node, self.mov_fork, 
            self.mov_randomize_input_adaptor, self.mov_randomize_cell, self.mov_toggle_input,
            self.mov_train, self.mov_toggle_cell]
        for f in [self.page_node, self.page_cell, self.page_param, self.page_input]:
            a.append(partial(f, direction=-1))
            a.append(partial(f, direction=1))
        return a
    
    def mov_toggle_input(self, world):
        src = self.input_key
        to = self.node_key
        if self.graph.has_edge(src, to) and self.graph[src][to]['muteable']:
            self.graph[src][to]['muted'] = not self.graph[src][to]['muted']

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
        m = len(self.input_nodes) - 1
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
            reward = -self.current_loss - 1
        if self.ratcheting or keep_current:
            self.fork_graph(keep_current)
            self.page_node(world, 0)
        self.scoreboard.record(self.graph)
        self._forked_graph_loss = 0.0
        self._graph_loss = 0.0
        self.cooldown = 10
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
                return -1.
            if self.gas <= 0:
                break
        g_loss /= iterations
        f_loss /= iterations
        reward = None
        if self.lowest_loss is None:
            self.lowest_loss = g_loss
            self.initial_loss = g_loss
            self.current_bench = g_loss
        else:
            #scale loss relative to initial loss
            l_delta = (self.lowest_loss - g_loss) / self.initial_loss
            reward = (self.initial_loss - self.current_loss * 0.98) / self.initial_loss
            #special rewards for new low loss
            if l_delta > 0:
                reward += l_delta
                self.lowest_loss = g_loss
                #may earn back 100% of initial gas if loss drops 100%
                self.gas += l_delta * self.initial_gas
        self.current_loss = g_loss
        return reward

    def draw(self):
        return self.graph
        G = nx.OrderedDiGraph()
        for ni, node in reversed(self.graph.nodes.items()):
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

    def mov_add_node(self, world):
        if self.current_bench is None or (self.current_loss / self.current_bench) > self.add_threshold:
            return
        self.scoreboard.record(self.graph)
        self.current_bench = self.current_loss
        size1, size2 = self.param_state[0:2]
        link_from = self.graph.nodes[self.input_key]
        link_to = self.graph.nodes[self.node_key]
        print('#'*20)
        print('add_node', size1, size2, link_from, link_to)
        in_dim = link_from['out_dim']
        out_dim = link_to['in_dim']
        in_volume = np.prod(in_dim)
        if len(in_dim) == 1: 
            #1D
            if size1 > 0: 
                # upscale
                in_dim = (int(in_volume * (size1 ** .5)), )
            else:
                #downscale
                in_dim = (int(max(in_volume * (2 ** size1), 1)), )
        else: #2D
            if size1 > 0: 
                # upscale
                c = int(in_dim[0] * (size1 ** .5))
            else:
                c = int(max(in_dim[0] * (2 ** size1), 1))
                #downscale
            wh = int(max(in_dim[1] * (2 ** size2), 1))
            in_dim = (c, wh, wh)
        new_index = len(self.graph.tensor_nodes)
        new_key = str(new_index)
        new_node = self.graph.create_hypercell(in_dim, out_dim, key=new_key)
        new_node.apply(self.init_weights)
        self.graph[new_key][self.node_key]['muted'] = False
        self.graph_optimizer = optim.SGD(self.graph.parameters(), lr=0.1, momentum=0.9)
        self.cooldown = 20
        self.node_index = new_index
        self.page_node(world, 0)
        
    def mov_randomize_input_adaptor(self, world):
        key = '%s->%s' % (self.input_key, self.node_key)
        if key in world.graph.tensor_adaptors:
            a = world.graph.tensor_adaptors[key]
            a.apply(self.init_weights)
    
    def mov_randomize_cell(self, world):
        '''
        Randomize the weights
        '''
        cell = self.current_cell
        if cell is not None:
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
        
    def mov_train(self, world):
        iterations = int(self.param_state[4] + 3) ** 2
        if self.cooldown > 0:
            self.cooldown -= iterations
        return self.train(iterations)
    
    def mov_toggle_cell(self, world):
        node = self.current_node
        if not hasattr(node, 'cells'):
            return
        cell_index = world.cell_index
        if cell_index < len(node.muted_cells):
            node.muted_cells = torch.ones_like(node.muted_cells)
            node.muted_cells[cell_index] = -1
        
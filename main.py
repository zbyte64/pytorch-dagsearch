import os
from dagsearch.dag import Graph
from dagsearch.world import World
from dagsearch.cells import CELL_TYPES
from dagsearch.dag_env import DagSearchEnv
from dagsearch.drqn import Trainer
from torchvision import datasets, transforms

import networkx as nx
import matplotlib.pyplot as plt

from torch import nn
import torch
from torchviz import make_dot, make_dot_from_trace

cell_types = list(CELL_TYPES.keys())


in_dim = (1, 28, 28)
out_dim = (10,)
batch_size = 32
final_image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])
data = datasets.MNIST('./MNIST', download=True, transform=final_image_transform, train=True)
data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=batch_size,
                                          shuffle=True,)
validata = datasets.MNIST('./MNIST', download=True, transform=final_image_transform, train=False)
validata_loader = torch.utils.data.DataLoader(validata,
                                          batch_size=batch_size,
                                          shuffle=True,)

g = Graph(cell_types, in_dim)
g.create_node((28*28,))
g.create_node((500,))
g.create_node((256,))
'''
g.create_node((4, 28, 28))
g.create_node((4, 28, 28))
g.create_node((8, 20, 20))
g.create_node((18, 8, 8))
g.create_node((20,))
'''
g.create_node(out_dim)
x, _ = next(iter(data_loader))
#dot = make_dot(g(x), params=dict(g.named_parameters()))
#dot.view()
world = World(g, data_loader, validata_loader, nn.CrossEntropyLoss(), initial_gas=30)

print(world.actions())
print(world.observe())

env = DagSearchEnv(world)
#env.render()
trainer = Trainer(world, env)
if os.path.exists('./trainer.pth'):
    trainer.policy_net.load_state_dict(torch.load('./trainer.pth'))
trainer.train(10000)
env.render()
print(trainer.score_board)
torch.save(trainer.policy_net.state_dict(), './trainer.pth')

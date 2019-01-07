import os
from dagsearch.dag import Graph
from dagsearch.world import World
from dagsearch.cells import CELL_TYPES
from dagsearch.dag_env import DagSearchEnv
from dagsearch.drqn import Trainer
from torchvision import datasets, transforms

from torch import nn
import torch
#from torchviz import make_dot, make_dot_from_trace

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

g = Graph(cell_types, in_dim, channel_dim=1)
for s in reversed([
     (1, 5, 5),
     out_dim
    ]):
    g.create_node(s)
x, _ = next(iter(data_loader))

world = World(g, data_loader, validata_loader, nn.CrossEntropyLoss(), initial_gas=60*5)
#dot = make_dot(world.graph(x), params=dict(world.graph.named_parameters()))
#dot.view()
#exit()
env = DagSearchEnv(world)
#env.render()
trainer = Trainer(world, env)
import copy
if os.path.exists('./trainer.pth'):
    trainer.policy_net.load_state_dict(torch.load('./trainer.pth'))
#trainer.train(5)
#env.render()
while True:
    trainer.train(1000)
    #env.render()
    print('saving...')
    torch.save(trainer.policy_net.state_dict(), './trainer.pth')

from dagsearch.dag import Graph, World
from dagsearch.cells import CELL_TYPES
from dagsearch.dag_env import DagSearchEnv
from torchvision import datasets, transforms

import networkx as nx
import matplotlib.pyplot as plt

from torch import nn
import torch

cell_types = list(CELL_TYPES.keys())


in_dim = (1, 28, 28)
out_dim = (10,)
batch_size = 32
final_image_transform = transforms.Compose([
    transforms.ToTensor(),
])
data = datasets.MNIST('./MNIST', download=True, transform=final_image_transform)
data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=batch_size,
                                          shuffle=True,)

g = Graph(cell_types, in_dim, out_dim)
g.create_node((4, 28, 28))
world = World(g)

print(world.actions())
print(world.observe())

env = DagSearchEnv(world, data_loader, nn.CrossEntropyLoss())


for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

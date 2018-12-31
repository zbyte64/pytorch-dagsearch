from dagsearch.dag import Graph, World
from dagsearch.cells import CELL_TYPES
from dagsearch.dqn import Trainer
from torchvision import datasets, transforms

from torch import nn
import torch

cell_types = CELL_TYPES.keys()


in_dim = (1, 28, 28)
out_dim = (10,)
batch_size = 8
final_image_transform = transforms.Compose([
    transforms.ToTensor(),
])
data = datasets.MNIST('./MNIST', download=True, transform=final_image_transform)
data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=batch_size,
                                          shuffle=True,)

g = Graph(cell_types, in_dim, out_dim)
g.create_node((4, 28, 28))
g.create_node((4, 28, 28))
g.create_node((8, 20, 20))
g.create_node((8, 20, 20))
g.create_node((16, 12, 12))
g.create_node((16, 12, 12))
g.create_node((32, 4, 4))
g.create_node((32, 4, 4))
world = World(g)

print(world.actions())
print(world.observe())

t = Trainer(world)
t.train(data_loader)

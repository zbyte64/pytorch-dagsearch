import os
from dagsearch.dag import Graph
from dagsearch.world import World
from dagsearch.cells import CELL_TYPES
from dagsearch.dag_env import DagSearchEnv
from dagsearch.drqn import Trainer
from dagsearch.utils import inf_data
from dagsearch.env import *
from torchvision import datasets, transforms

from torch import nn
import torch
import networkx as nx
import matplotlib.pyplot as plt
#from torchviz import make_dot, make_dot_from_trace

cell_types = list(CELL_TYPES.keys())


def env_from_dataloader(dataloader, n_classes):
    data_iter = inf_data(dataloader)
    sample = next(data_iter)
    x, y = sample
    in_dim = x.shape[1:]
    out_dim = y.shape[1:]
    criterion = nn.CrossEntropyLoss()
    g = Graph(cell_types, in_dim, channel_dim=1)
    g.create_hypercell((n_classes,))
    g.create_hypercell((1, 5, 5))
    def sample_loss(graph):
        x, y = next(data_iter)
        vx = x.to(device)
        vy = y.to(device)
        py = graph(vx)
        graph_loss = criterion(py, vy)
        return graph_loss
    world = World(g, sample_loss=sample_loss, initial_gas=30, max_gas=600)
    env = DagSearchEnv(world)
    return env


def env_for_metatrain(envs):
    #TODO 
    world_size = envs[0].world.observe().shape[0]
    #include memory embeding 
    in_dim = (world_size*3,)
    out_dim = (len(envs[0].world.actions()), )
    g = Graph(cell_types, in_dim, channel_dim=1)
    g.create_hypercell(out_dim)
    g.create_hypercell((world_size, ))
    def sample_loss(graph):
        #TODO huber loss from memory
        #populate memory
        for env in envs:
            #TODO select action, iterate
            pass
        pass
    world = World(g, sample_loss=sample_loss, initial_gas=30, max_gas=600)
    env = DagSearchEnv(world)
    return env


batch_size = 32
final_image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])
data = datasets.FashionMNIST('./FashionMNIST', download=True, transform=final_image_transform, train=True)
data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=batch_size,
                                          shuffle=True,)

#TODO CIFAR10, EMNIST, VHSN
env = env_from_dataloader(data_loader, n_classes=10)
trainer = Trainer(env)
import copy
if os.path.exists('./trainer.pth'):
    trainer.policy_net.load_state_dict(torch.load('./trainer.pth'))
if os.path.exists('./embeding.pth'):
    trainer.embeding.load_state_dict(torch.load('./embeding.pth'))
#trainer.train(5)
#env.render()
print(trainer.action_size)
while True:
    trainer.train(10000)
    #env.render()
    print('saving...')
    torch.save(trainer.policy_net.state_dict(), './trainer.pth')
    torch.save(trainer.embeding.state_dict(), './embeding.pth')
    '''
    l, g = world.scoreboard.leaders[0]
    labels=dict((n,'%s %s' % (n, ['%s: %s' % (k,v) for k, v in d.items()])) for n,d in g.nodes(data=True))
    nx.draw(g, node_size=100, labels=labels)
    #plt.subplot(400)
    print('Loss:', l)
    plt.show()
    '''

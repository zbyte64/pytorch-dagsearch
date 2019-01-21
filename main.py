import os
from dagsearch.dag import Graph
from dagsearch.world import World
from dagsearch.cells import CELL_TYPES
from dagsearch.dag_env import DagSearchEnv
from dagsearch.memory_embeding import SessionMemory, MemoryEmbed
from dagsearch.agents import Agent, MetaAgent
from dagsearch.utils import inf_data
from dagsearch.env import *
from torchvision import datasets, transforms

from torch import nn
import torch.optim as optim
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
    g = Graph(cell_types, in_dim, channel_dim=1).to(device)
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


def meta_agent(envs, memory, embedding):
    #TODO 
    world_size = envs[0].world.observe().shape[0]
    action_size = len(envs[0].world.actions())
    #include memory embeding 
    in_dim = (world_size*3,)
    out_dim = (action_size, )
    g = Graph(cell_types, in_dim, channel_dim=1).to(device)
    g.create_hypercell(out_dim)
    g.create_hypercell((world_size, ))
    policy_net = g
    def sample_loss(graph):
        #our loss is based on agent performance
        return ma.child_agents[0].sample_loss()
    world = World(g, sample_loss=sample_loss, initial_gas=30, max_gas=600)
    env = DagSearchEnv(world)
    ma = MetaAgent(env, memory, embeding, policy_net, child_envs=envs)
    return ma


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

world_size = env.world.observe().shape[0]
action_size = len(env.world.actions())
embeding = MemoryEmbed(world_size, action_size).to(device)
embeding_optimizer = optim.RMSprop(embeding.parameters())
memory = SessionMemory(100)

def optimize_memory():
    #optimize embed space
    embeding_optimizer.zero_grad()
    e_loss = embeding.sample_loss(memory)
    if len(e_loss):
        e_loss = torch.mean(torch.stack(e_loss))
        e_loss.backward()
        embeding_optimizer.step()

trainer = meta_agent([env], memory, embeding)
import copy
if os.path.exists('./trainer.pth'):
    trainer.policy_net.load_state_dict(torch.load('./trainer.pth'))
if os.path.exists('./embeding.pth'):
    trainer.embeding.load_state_dict(torch.load('./embeding.pth'))
#trainer.train(5)
#env.render()
print(list(trainer.policy_net._modules.keys()))
while True:
    trainer.tick_for(1000)
    optimize_memory()
    #TODO upsert if trainer.sample_loss > trainer.world.scoreboard.leaders
    #env.render()
    #print('saving...')
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

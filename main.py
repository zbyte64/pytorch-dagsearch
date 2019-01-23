import os
from dagsearch.dag import Graph
from dagsearch.world import World
from dagsearch.cells import CELL_TYPES
from dagsearch.dag_env import DagSearchEnv
from dagsearch.memory_embeding import SessionMemory, MemoryEmbed
from dagsearch.agents import Agent, Trainer
from dagsearch.utils import inf_data
from dagsearch.env import *
from torchvision import datasets, transforms
import copy

from torch import nn
import torch.optim as optim
import torch
import networkx as nx
import matplotlib.pyplot as plt
#from torchviz import make_dot, make_dot_from_trace

cell_types = list(CELL_TYPES.keys())


def env_from_dataloader(dataloader, n_classes):
    '''
    Given a classification task/dataloader, 
    returns an OpenAI gym environment for ENAS
    '''
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


def meta_trainer(envs, memory, embedding):
    world_size = envs[0].world.observe().shape[0]
    action_size = len(envs[0].world.actions())
    #include memory embeding 
    in_dim = (world_size*3,)
    out_dim = (action_size, )
    policy_net = Graph(cell_types, in_dim, channel_dim=1).to(device)
    policy_net.create_hypercell(out_dim)
    policy_net.create_hypercell((world_size, ))
    def sample_loss(graph):
        c_trainer = Trainer(graph, [c_a], target_net=trainer.target_net)
        return c_trainer.sample_loss()
    c_policy = copy.deepcopy(policy_net)
    world = World(c_policy, sample_loss=sample_loss, initial_gas=30, max_gas=600)
    meta_env = DagSearchEnv(world)
    c_a = Agent(meta_env, memory, embedding, policy_net)
    agents = [Agent(env, memory, embeding, policy_net) for env in envs]
    agents.append(c_a)
    trainer = Trainer(policy_net, agents)
    return trainer


batch_size = 32
final_image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])
envs = list()
for data, n_classes in [
    (datasets.FashionMNIST('./FashionMNIST', download=True, transform=final_image_transform, train=True), 10),
    (datasets.CIFAR10('./CIFAR10', download=True, transform=final_image_transform, train=True), 10),
    #(datasets.EMNIST('./EMNIST', download=True, transform=final_image_transform, train=True, split='balanced'), 47),
    (datasets.SVHN('./SVHN', download=True, transform=final_image_transform, split='train'), 10),
]:
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=batch_size,
                                              shuffle=True,)
    envs.append(env_from_dataloader(data_loader, n_classes=n_classes))

world_size = envs[0].world.observe().shape[0]
action_size = len(envs[0].world.actions())
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
        for param in embeding.parameters():
            param.grad.data.clamp_(-1, 1)
        embeding_optimizer.step()
        print('Memory Loss: %.4f' % e_loss.item())

trainer = meta_trainer(envs, memory, embeding)
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
    c_loss = envs[-1].world._sample_loss()
    m_loss = trainer.sample_loss()
    if c_loss < m_loss * .9:
        trainer.set_policy(copy.deepcopy(envs[-1].world.graph))
    else:
        best_policy_net = copy.deepcopy(trainer.policy_net)
        envs[-1].world.initial_graph = best_policy_net
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

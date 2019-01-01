from gym import Env, spaces
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class DagSearchEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, world, data_loader, loss_fn):
        self.world = world
        self.data_loader = data_loader
        self.criterion = loss_fn
        self.action_space = spaces.Tuple((
            spaces.Discrete(len(world.actions())),
            spaces.Box(low=-1., high=-1, shape=(2,), dtype=np.float32)
        ))
        self.observation_space = spaces.Discrete(world.observe().shape[0])

    def next_batch(self):
        if not hasattr(self, '_ds_iter'):
            self._ds_iter = iter(self.data_loader)
        try:
            return next(self._ds_iter)
        except StopIteration:
            self._ds_iter = iter(self.data_loader)
            return next(self._ds_iter)

    def train(self, iterations=1):
        f_loss = 0.
        g_loss = 0.
        for i in range(iterations):
            batch = self.next_batch()
            x, y = batch
            forked_loss = self.criterion(self.world.forked_graph(x), y)
            graph_loss = self.criterion(self.world.graph(x), y)
            self.world.optimize(graph_loss, forked_loss)
            f_loss += forked_loss
            g_loss += graph_loss
        return (g_loss, f_loss)


    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        reward = self._reward(action)
        ob = self.observe()
        episode_over = self.world.gas <= 0
        return ob, reward, episode_over, {}

    def reset(self):
        self.world.rebuild()
        return self._observe()

    def observe(self):
        return self.world.observe()

    def render(self, mode='human', close=False):
        if close: return

        g = self.world.draw()
        labels=dict((n,'%s %s' % (n, ['%s: %s' % (k,v) for k, v in d.items()])) for n,d in g.nodes(data=True))
        nx.draw(g, node_size=100, labels=labels)
        #plt.subplot(400)
        plt.show()
        return g

    def _reward(self, action):
        (a, d) = action
        #TODO two direction support
        d = d[0]
        r = self.world.perform_action(a, d) or 0.
        graph_loss, forked_loss = self.train()
        return r + (forked_loss - graph_loss)

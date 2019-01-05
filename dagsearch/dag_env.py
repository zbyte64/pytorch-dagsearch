from gym import Env, spaces
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time


class DagSearchEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, world):
        self.world = world
        self.action_space = spaces.Discrete(len(world.actions()))
        self.observation_space = spaces.Discrete(world.observe().shape[0])
        self._last_loss = None
        self._lowest_loss = None

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
        assert self.world.gas > 0
        reward, info = self._reward(action)
        ob = self.observe()
        episode_over = self.world.gas <= 0
        return ob, reward, episode_over, info

    def reset(self):
        self.world.rebuild()
        self._last_loss = None
        self._lowest_loss = None
        return self.observe()

    def observe(self):
        return self.world.observe().view(1, -1)

    def render(self, mode='human', close=False):
        if close: return

        g = self.world.draw()
        labels=dict((n,'%s %s' % (n, ['%s: %s' % (k,v) for k, v in d.items()])) for n,d in g.nodes(data=True))
        nx.draw(g, node_size=100, labels=labels)
        #plt.subplot(400)
        plt.show()
        return g

    def _reward(self, action):
        r = self.world.perform_action(action) or 0.
        graph_loss, forked_loss = self.world.train()
        if self._lowest_loss is None:
            self._lowest_loss = graph_loss
        elif self._lowest_loss > graph_loss:
            l_delta = self._lowest_loss - graph_loss
            self._lowest_loss = graph_loss
            self.world.gas += l_delta * (self.world.initial_gas ** 2)
        delta_loss = 0.
        if self._last_loss is not None:
            delta_loss = self._last_loss - graph_loss * 0.999999
        self._last_loss = graph_loss
        reward = r + delta_loss
        if reward > 0:
            reward *= (forked_loss / graph_loss)
        if self.world.gas < 0:
            #add final score
            reward -= graph_loss
        reward = np.tanh(reward)
        info = {
            'delta_loss': delta_loss,
            'loss': graph_loss,
            'gas': self.world.gas,
        }
        return reward, info

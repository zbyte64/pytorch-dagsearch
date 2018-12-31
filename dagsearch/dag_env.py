from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from gym import Env

class DagSearchEnv(Env)
    #metadata = {'render.modes': ['human']}

    def __init__(self, world, model=None, optimizer=None, device=None, data_loader=None, loss_fn=nn.MSELoss):
        self.world = world
        self.data_loader = data_loader
        self.model = model or world
        self.optimizer = optimizer #SGD(world.parameters(), lr=lr, momentum=momentum)
        self.trainer = create_supervised_trainer(self.model, optimizer, loss_fn, device=device)

    def train(self, epochs=1):
        return self.trainer.run(self.data_loader, max_epochs=epochs)

    def _step(self, action):
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
        ob = self._observe()
        episode_over = False
        return ob, reward, episode_over, {}

    def _reset(self):
        pass

    def _observe(self):
        return self.world.observe()

    def _render(self, mode='human', close=False):
        if close: return

        g = self.world.draw()
        labels=dict((n,'%s %s' % (n, ['%s: %s' % (k,v) for k, v in d.items()])) for n,d in g.nodes(data=True))
        nx.draw(g, node_size=100, labels=labels)
        #plt.subplot(400)
        plt.show()
        return g

    def _reward(self, action):
        self.world.perform_action(action)
        loss = self.train()
        return 0

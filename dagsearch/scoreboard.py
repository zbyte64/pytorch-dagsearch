import torch
import heapq
import copy

from .env import *


class Scoreboard(object):
    def __init__(self, criterion, top_k):
        self.criterion = criterion
        self.top_k = top_k
        self.leaders = list()
    
    def record(self, model, n_trials=5):
        model = copy.deepcopy(model).to(device)
        with torch.no_grad():
            t_loss = 0.
            for i in range(n_trials):
                loss = self.criterion(model)
                if loss is None:
                    continue
                t_loss += loss.cpu().item()
            if t_loss == 0. :
                return
            self.leaders.append((t_loss/n_trials, model))
            self.leaders = heapq.nsmallest(self.top_k, self.leaders, key=lambda x: x[0])
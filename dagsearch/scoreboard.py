import torch
import heapq
import copy

from .env import *


class Scoreboard(object):
    def __init__(self, criterion, valid_data, top_k):
        self.criterion = criterion
        self.valid_data = valid_data
        self.top_k = top_k
        self.leaders = list()
    
    def record(self, model, n_trials=5):
        model = copy.deepcopy(model).to(device)
        with torch.no_grad():
            t_loss = 0.
            for i in range(n_trials):
                x, y = next(self.valid_data)
                x, y = x.to(device), y.to(device)
                py = model(x)
                loss = self.criterion(py, y).cpu().item()
                t_loss += loss
            self.leaders.append((t_loss/n_trials, model))
            self.leaders = heapq.nsmallest(self.top_k, self.leaders, key=lambda x: x[0])
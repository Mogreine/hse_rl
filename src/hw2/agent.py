import random
import numpy as np
import os
import torch
from .train import Net


class Agent:
    def __init__(self):
        self.model = Net(8, 4)
        self.model.load_state_dict(torch.load(__file__[:-8] + "/agent.pt"))
        self.model.cuda()
        self.model.eval()

    def act(self, state):
        with torch.no_grad():
            x = torch.from_numpy(np.array(state)).cuda()
            preds = self.model(x)
            action = torch.argmax(preds)
            return action.item()

    def reset(self):
        pass

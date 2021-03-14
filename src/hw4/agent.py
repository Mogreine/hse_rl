import random
import numpy as np
import os
import torch

from .train import Actor

SEED = 65537
DEVICE = torch.device("cpu")
torch.manual_seed(SEED)


class Agent:
    def __init__(self):
        self.model = Actor(28, 8)
        self.model.load_state_dict(torch.load(__file__[:-8] + "/agent.pt", map_location=DEVICE))
        self.model = self.model.to(DEVICE)
        self.model.eval()

    def act(self, state):
        with torch.no_grad():
            state = torch.unsqueeze(torch.tensor(np.array(state)).float(), 0).to(DEVICE)
            action, _, _ = self.model.act(state)
        return action[0].numpy()

    def reset(self):
        pass

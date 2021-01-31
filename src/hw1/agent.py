import random
import numpy as np
import os
from .train import transform_state


class Agent:
    def __init__(self):
        # self.Qfun = np.load(__file__[:-8] + "agent.npz")
        self.Qfun = np.load(__file__[:-8] + "/agent.npy")

    def act(self, state):
        state = transform_state(state)
        a = np.argmax(self.Qfun[state])
        return a

    def reset(self):
        pass

import random
import numpy as np
import os
from .train import transform_state

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

SEED = 65537
rs = RandomState(MT19937(SeedSequence(SEED)))


class Agent:
    def __init__(self):
        self.Qfun = np.load(__file__[:-8] + "/agent.npy")
        # self.Qfun = np.load(__file__[:-14] + "/agent.npy")
        self.eps = 0.1

    def act(self, state):
        state = transform_state(state)
        if rs.random() < self.eps:
            a = rs.randint(0, 3)
        else:
            a = np.argmax(self.Qfun[state])
        return a

    def reset(self):
        pass

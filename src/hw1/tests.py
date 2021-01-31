from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import numpy as np

from src.hw1.train import evaluate_policy
from src.hw1.agent import Agent
from src.hw1.agent_sarsa import Agent as AgentSarsa

if __name__ == "__main__":
    agent = AgentSarsa()
    rewards = evaluate_policy(agent, 5)
    mean, std = np.mean(rewards), np.std(rewards)
    print(f"Reward mean: {mean}, Reward std: {std}")

from gym import make
import numpy as np
import torch
import copy
from collections import deque
import random
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

import plotly.graph_objs as go
import plotly

SEED = 65537
rs = RandomState(MT19937(SeedSequence(SEED)))

GRID_SIZE_X = 40
GRID_SIZE_Y = 40


# Simple discretization
def transform_state(state):
    state = (np.array(state) + np.array((1.2, 0.07))) / np.array((1.8, 0.14))
    x = min(int(state[0] * GRID_SIZE_X), GRID_SIZE_X - 1)
    y = min(int(state[1] * GRID_SIZE_Y), GRID_SIZE_Y - 1)
    return x + GRID_SIZE_X * y


class QLearning:
    def __init__(self, state_dim, action_dim, alpha=0.9, gamma=0.9, lr_decay=1.0):
        self.Qfun = np.zeros((state_dim, action_dim)) + 2.
        self.alpha = alpha
        self.gamma = gamma
        self.best_score = -200
        self.lr_decay = lr_decay

    def update(self, transition):
        state, action, next_state, reward, done = transition
        # Q(s, a) = (1 - lr) * Q(s, a) + lr * (r + gamma * max_a(Q(s')) - Q(s, a))
        self.Qfun[state][action] = self.Qfun[state][action] * (1 - self.lr_decay * self.alpha)\
                                   + self.lr_decay * self.alpha\
                                   * (reward + self.gamma * np.amax(self.Qfun[next_state]) - self.Qfun[state][action])

    def act(self, state):
        a = np.argmax(self.Qfun[state])
        return a

    def save(self, path, score):
        if self.best_score < score:
            self.best_score = score
            np.save(path, self.Qfun)


def evaluate_policy(agent, episodes=5):
    env = make("MountainCar-v0")
    env.seed(SEED)
    env.action_space.seed(SEED)
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            action = agent.act(transform_state(state))
            state, reward, done, _ = env.step(action)
            # reward += abs(state[1]) / 0.07
            total_reward += reward
        returns.append(total_reward)
    return returns


def evaluate_policy_mine(agent, episodes=5):
    env = make("MountainCar-v0")
    env.seed(SEED)
    env.action_space.seed(SEED)
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            action = agent.act(transform_state(state))
            state, reward, done, _ = env.step(action)
            pos, vel = state
            reward += abs(vel) / 0.07
            total_reward += reward
        returns.append(total_reward)
    return returns


def plot(data):
    def gen_color():
        return {
            'color': f'rgba({np.random.randint(0, 255)},'
                     f' {np.random.randint(0, 255)},'
                     f' {np.random.randint(0, 255)},'
                     f' 0.8)'
        }

    traces = []
    for method_name, method_data in data.items():
        traces.append(
            go.Scatter(
                x=method_data[0],
                y=method_data[1],
                mode='lines',
                marker=gen_color(),
                name=method_name
            )
        )

    layout = go.Layout(
        xaxis=dict(title="transitions", zeroline=False),
        yaxis=dict(title='$reward$', zeroline=False)
    )

    fig = go.Figure(data=traces, layout=layout)
    plotly.offline.iplot(fig)


if __name__ == "__main__":
    env = make("MountainCar-v0")
    env.seed(SEED)
    env.action_space.seed(SEED)

    eps_decay = 0.6
    lr = 0.2
    lr_decay = 0.92
    gamma = 0.7

    ql = QLearning(state_dim=GRID_SIZE_X * GRID_SIZE_Y, action_dim=3, alpha=lr, gamma=gamma, lr_decay=lr_decay)
    eps = 0.2
    transitions = 2000000
    trajectory = []
    state = transform_state(env.reset())
    rewards_hist = []
    x = []
    for i in range(transitions):
        total_reward = 0
        steps = 0
        eps *= eps_decay
        # env.render()

        # Epsilon-greedy policy
        if rs.random() < eps:
            action = env.action_space.sample()
        else:
            action = ql.act(state)

        next_state, reward, done, _ = env.step(action)
        reward += abs(next_state[1]) / 0.07
        next_state = transform_state(next_state)

        trajectory.append((state, action, next_state, reward, done))

        if done:
            for transition in reversed(trajectory):
                ql.update(transition)
            trajectory = []
            eps = 0.1
        state = next_state if not done else transform_state(env.reset())

        if (i + 1) % (transitions // 100) == 0:
            rewards1 = evaluate_policy_mine(ql, 5)
            rewards2 = evaluate_policy(ql, 5)
            mean, std = np.mean(rewards1), np.std(rewards1)
            print(f"Mine Step: {i + 1}, Reward mean: {mean}, Reward std: {std}")

            mean, std = np.mean(rewards2), np.std(rewards2)
            print(f"Initial Step: {i + 1}, Reward mean: {mean}, Reward std: {std}")
            ql.save('agent', mean)

            x.append(i)
            rewards_hist.append(mean)

    data = {
        'q-learning': (x, rewards_hist)
    }
    plot(data)

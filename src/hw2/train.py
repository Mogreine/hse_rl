from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque
import copy
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

GAMMA = 0.99
INITIAL_STEPS = 4000
TRANSITIONS = 500000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BUFFER_SIZE = 10_000
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
l2 = 1e-5

TAU = 0.005
GRADIENT_CLIP = 5
HIDDEN_DIM = 1024

SEED = 65537
rs = RandomState(MT19937(SeedSequence(SEED)))
torch.manual_seed(SEED)


class Net(nn.Module):
    def __init__(self, state_dim, action_dim, n_hidden_layers=5):
        super().__init__()
        hidden_layers = [nn.Linear(state_dim, HIDDEN_DIM), nn.LeakyReLU()]
        for _ in range(n_hidden_layers):
            hidden_layers += [nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.LeakyReLU()]
        hidden_layers += [nn.Linear(HIDDEN_DIM, action_dim)]
        self.layers = nn.Sequential(*hidden_layers)

    def forward(self, x):
        return self.layers(x)


class DQN:
    def __init__(self, state_dim, action_dim, buffer_size=1000, gamma=0.99, verbose=False):
        self.steps = 0
        self.net = Net(state_dim, action_dim).cuda()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.net_target = Net(state_dim, action_dim).cuda()
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = BATCH_SIZE
        self.verbose = verbose
        self.gamma = gamma
        self.best_r = -1e8

    def consume_transition(self, transition):
        self.buffer.append(transition)

    def sample_batch(self):
        ix = rs.randint(0, len(self.buffer), size=self.batch_size)
        batch = [self.buffer[ind] for ind in ix]
        return batch

    def train_step(self, batch, actions, y):
        # Use batch to update DQN's network.
        y_hat = self.net(batch)
        y_hat = y_hat[torch.arange(y_hat.shape[0]), actions]

        loss = F.mse_loss(y_hat, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.net.parameters(), GRADIENT_CLIP)
        self.optimizer.step()

        self.update_target_network()

        if self.verbose:
            print(f'loss: {loss.item()}')

    def update_target_network(self):
        # self.net_target.load_state_dict(self.net.state_dict())
        for target_param, local_param in zip(self.net_target.parameters(), self.net.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

    def act(self, state, target=False):
        with torch.no_grad():
            self.net = self.net.eval()
            state = torch.from_numpy(np.array(state)).cuda()
            preds = self.net(state)
            action = torch.argmax(preds)
            self.net = self.net.train()
        return action.item()

    def parse_batch(self, batch_trans):
        arrs = zip(*batch_trans)
        train_batch, actions, target_batch, rewards, dones = map(lambda arr: torch.from_numpy(np.array(arr)).cuda(),
                                                                 arrs)

        with torch.no_grad():
            target_preds = self.net_target(target_batch)
            target_preds = torch.amax(target_preds, dim=1)

            target_preds[dones] = 0
            target_preds *= self.gamma
            target_preds += rewards

        y = target_preds
        return train_batch, actions, y

    def update(self, transition):
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            batch, actions, y = self.parse_batch(batch)
            self.train_step(batch, actions, y)
        # if self.steps % STEPS_PER_TARGET_UPDATE == 0:
        #     self.update_target_network()
        self.steps += 1

    def save(self, r):
        if self.best_r < r:
            torch.save(self.net.state_dict(), "agent.pt")
            self.best_r = r


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    env.seed(SEED)
    env.action_space.seed(SEED)
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    env = make("LunarLander-v2")
    env.seed(SEED)
    env.action_space.seed(SEED)
    load = False
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
              buffer_size=BUFFER_SIZE, verbose=False)
    if load:
        model = torch.load('agent.pkl')
        dqn.net = copy.deepcopy(model).cuda()
        dqn.net_target = copy.deepcopy(model).cuda()
        dqn.net_target.eval()
    eps_max = 0.1
    eps_min = 0.01
    state = env.reset()

    for _ in range(INITIAL_STEPS):
        steps = 0
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

    for i in range(TRANSITIONS):
        steps = 0
        eps = eps_max - (eps_max - eps_min) * i / TRANSITIONS
        # Epsilon-greedy policy
        if rs.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(dqn, 5)
            mean, std = np.mean(rewards), np.std(rewards)
            print(f"Step: {i + 1}, Reward mean: {mean}, Reward std: {std}")
            dqn.save(mean)

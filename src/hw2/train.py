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
INITIAL_STEPS = 1024
TRANSITIONS = 500000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 5e-4

SEED = 65537
rs = RandomState(MT19937(SeedSequence(SEED)))


class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, state_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(state_dim * 4, action_dim)
        )

    def forward(self, x):
        return self.layers(x)


class DQN:
    def __init__(self, state_dim, action_dim, buffer_size=1000, gamma=0.99, verbose=False):
        self.steps = 0  # Do not change
        self.net = Net(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.net_target = Net(state_dim, action_dim)
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = BATCH_SIZE
        self.verbose = verbose
        self.gamma = gamma

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.buffer.append(transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        ix = np.random.randint(0, len(self.buffer), size=self.batch_size)
        batch = [self.buffer[ind] for ind in ix]
        return batch

    def train_step(self, batch, actions, y):
        # Use batch to update DQN's network.
        y_hat = self.net(batch)
        y_hat = y_hat[torch.arange(y_hat.shape[0]), actions]

        loss = F.mse_loss(y_hat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.verbose:
            print(f'loss: {loss.item()}')

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or
        # assign a values of network parameters via PyTorch methods.
        self.net_target = copy.deepcopy(self.net_target)
        self.net_target = self.net_target.eval()

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        with torch.no_grad():
            state = torch.from_numpy(np.array(state))
            preds = self.net(state)
            action = torch.argmax(preds)
        return action.item()

    def parse_batch(self, batch_trans):
        states_fr, actions, states_to, rewards, dones = zip(*batch_trans)
        train_batch = torch.from_numpy(np.array(states_fr))
        actions = torch.from_numpy(np.array(actions)).long()
        target_batch = torch.from_numpy(np.array(states_to))
        rewards = torch.from_numpy(np.array(rewards))
        dones = torch.from_numpy(np.array(dones))

        with torch.no_grad():
            target_preds = self.net_target(target_batch)
            target_preds = torch.amax(target_preds, dim=1)

            target_preds[dones] = 0
            target_preds *= self.gamma
            target_preds += rewards

        y = target_preds
        return train_batch, actions, y

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            batch, actions, y = self.parse_batch(batch)
            self.train_step(batch, actions, y)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.net, "agent.pkl")


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
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
              buffer_size=INITIAL_STEPS * 2, verbose=False)
    eps = 0.1
    state = env.reset()

    for _ in range(INITIAL_STEPS):
        steps = 0
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

    for i in range(TRANSITIONS):
        steps = 0

        # Epsilon-greedy policy
        if np.random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(dqn, 5)
            print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            dqn.save()

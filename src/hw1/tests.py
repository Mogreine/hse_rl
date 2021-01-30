import gym

env = gym.make('MountainCar-v0')
env.reset()

for _ in range(1000):
    env.render()
    action = 2  # env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(reward)

env.close()

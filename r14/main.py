import gym
import time

env = gym.make('CartPole-v0')

### show the bounds
# print(env.observation_space.high)
# print(env.observation_space.low)

for i_episode in range(1):
    observation = env.reset()
    for t in range(1000):
        env.render()

        ### slow down the rendering
        time.sleep(0.1)

        ### keyboard control
        # input("Press Enter to continue...")

        ### sample an action: env.action_space.sample()
        ### specify an action using 0,1,....; Gym may not tell you which number means which action
        observation, reward, done, info = env.step(env.action_space.sample())

        # if done:
        #     print("Episode finished after {} timesteps".format(t + 1))
        #     break
env.close()
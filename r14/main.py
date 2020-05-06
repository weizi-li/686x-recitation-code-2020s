# CartPole v0:
# https://github.com/openai/gym/wiki/CartPole-v0

import gym
import time
import math
import random
import numpy as np

env = gym.make('CartPole-v0')

# show the bounds
# print(env.observation_space.high)
# print(env.observation_space.low)

# no buckets for position and velocity
# 6 buckets for angle, and 3 buckets for angular velocity
ob_buckets = (1, 1, 6, 3)
num_actions = env.action_space.n

# initialize the Q value table
Q_table = np.zeros(ob_buckets + (num_actions,))
#print(Q_table.shape)

# set the bounds for observations
ob_bounds = list(zip(env.observation_space.low, env.observation_space.high))
ob_bounds[1] = [-0.5, 0.5]  # changed from inf for velocity
ob_bounds[3] = [-math.radians(50), math.radians(50)]  # changed from inf for angular velocity

# hyperparameters, can be optimized using grid search
min_explore_rate = 0.01
min_learning_rate = 0.1
discount = 0.99

# task parameters
max_episodes = 1000
max_time_steps = 250
solved_time = 199
num_success_trials_current = 0
num_success_trials_to_end = 100  # Gym requires 100 consecutive successful to trials
render_flag = False  # slow down the rendering

def select_action(ob, er):
    if random.random() < er:
        action = env.action_space.sample()  # smaller than the exploration rate
    else:
        action = np.argmax(Q_table[ob])
    return action


# decrease the exploration rate as time advances
def select_er(x):
    return max(min_explore_rate, min(1, 1.0 - math.log10((x + 1) / 25)))


# decrease the learning rate as time advances
def select_lr(x):
    return max(min_learning_rate, min(0.5, 1.0 - math.log10((x + 1) / 25)))


def discretize_ob(ob):
    bucket_indices = []

    for i in range(len(ob)):
        if ob[i] <= ob_bounds[i][0]:  # smaller than the lower bound, assign the bucket index 0
            bucket_index = 0
        elif ob[i] >= ob_bounds[i][1]:  # greater than the upper bound, assign the bucket index (max)
            bucket_index = ob_buckets[i] - 1
        else:
            bound_width = ob_bounds[i][1] - ob_bounds[i][0]
            offset = (ob_buckets[i] - 1) * ob_bounds[i][0] / bound_width
            scaling = (ob_buckets[i] - 1) / bound_width
            bucket_index = int(round(scaling * ob[i] - offset))

        bucket_indices.append(bucket_index)
    return tuple(bucket_indices)


for e in range(max_episodes):
    # get exploration and learning rates for each episode
    er = select_er(e)
    lr = select_lr(e)
    print('Episode %d starting...' % e)
    print('Explore rate: %f' % er)
    print('Learning rate: %f' % lr)

    # get the initial observation of the environment
    ob_env = env.reset()

    # discretize observations into their indices
    start_ob = discretize_ob(ob_env)
    previous_ob = start_ob

    # start to learn
    for t in range(max_time_steps):
        # render the environment
        if render_flag:
            print("time step: %d" % t)
            env.render()
            time.sleep(0.05)

        # select an action according epsilon-greedy algorithm
        act = select_action(previous_ob, er)

        # execute one step with the selected action, collect info
        ob_env, reward, done, _ = env.step(act)

        # discretize the continuous observation from the environment
        ob = discretize_ob(ob_env)

        # retrieve the best Q value
        best_Q = np.amax(Q_table[ob])

        # update the Q value table
        Q_table[previous_ob + (act,)] += \
            lr * (reward + discount * best_Q - Q_table[previous_ob + (act,)])

        # print info for diagnosis
        # print('Time step: %d' % t)
        # print('Selected action: %d' % act)
        # print('Current observation: %s' % str(ob))
        # print('Best Q: %f' % best_Q)
        # print('Reward: %f' % reward)

        # the episode is finished, this trial can be either successful or failed
        if done:
            print('Episode %d finished after %f time steps' % (e, t))
            if t >= solved_time:  # this trial is successful
                num_success_trials_current += 1
            else:  # this trail is failed
                num_success_trials_current = 0
            break

        # update previous_ob so that we can use it to further update the Q value table
        previous_ob = ob

    print('Successful trials so far: %d' % num_success_trials_current)
    print("\n")

    # render the environment during the last successful trial
    if num_success_trials_current > num_success_trials_to_end-1:
        render_flag = True

    # if we have solved the CartPole problem, stop.
    if num_success_trials_current > num_success_trials_to_end:
        break

# for i_episode in range(1):
#     observation = env.reset()
#     for t in range(1000):
#         env.render()
#
#         ### slow down the rendering
#         time.sleep(0.1)
#
#         ### keyboard control
#         # input("Press Enter to continue...")
#
#         ### sample an action: env.action_space.sample()
#         ### specify an action using 0,1,....; Gym may not tell you which number means which action
#         observation, reward, done, info = env.step(env.action_space.sample())
#
#         # if done:
#         #     print("Episode finished after {} timesteps".format(t + 1))
#         #     break


env.close()

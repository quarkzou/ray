import gym
from gym.spaces import Discrete, Box
import numpy as np

s = Box(0, 1, shape=(3, 3), dtype=np.int8)
print(s.sample())
print(s.sample())
print(s.sample())
print(s.sample())
# print(s)

# env = gym.make('CartPole-v0') # 创建环境
#
# observe = env.reset()
#
# print(env.action_space)
# print(env.observation_space)
#
# env.step(0) # 0表示向左，1表示向右
# print(env.action_space.sample())
# print(env.observation_space.sample())

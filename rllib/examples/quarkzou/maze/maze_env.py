import gym
from gym.spaces import Discrete, Box
from ray.rllib.env.env_context import EnvContext
import numpy as np
import random

MAZE_SIZE = 4
MAZE_SHAPE = (4, 4)
TRAP1_POS = np.array([1, 1])
TRAP2_POS = np.array([2, 2])
EXIT_POS = np.array([2, 1])
# TREASURE_POS = np.array([2, 3])
TREASURE_POS = np.array([2, 3])

TREASURE_VALUE = 20.0
TRAP_VALUE = -10.0
EXIT_VALUE = 10.0
POS_VALUE = 1.0


class QuarkMaze(gym.Env):
    def __init__(self, config: EnvContext):
        # 0:up, 1:down, 2:left, 3:right
        self.action_space = Discrete(4)
        self.observation_space = Box(0, MAZE_SIZE - 1, shape=(2,), dtype=np.int8)
        self.cur_pos = np.array([0, 0])
        self.seed((config.worker_index + 1) * (config.num_workers + 1))

    def reset(self):
        self.cur_pos = np.array([0, 0])
        return self.cur_pos

    def step(self, action):
        done = False
        reward = 0.0

        # 0:up, 1:down, 2:left, 3:right
        # if action == 0 and self.cur_pos[1] > 0:
        #     self.cur_pos[1] = self.cur_pos[1] - 1
        # elif action == 1 and self.cur_pos[1] < MAZE_SIZE - 1:
        #     self.cur_pos[1] = self.cur_pos[1] + 1
        # elif action == 2 and self.cur_pos[0] > 0:
        #     self.cur_pos[0] = self.cur_pos[0] - 1
        # elif action == 3 and self.cur_pos[0] < MAZE_SIZE - 1:
        #     self.cur_pos[0] = self.cur_pos[0] + 1
        if action == 0:
            if self.cur_pos[1] > 0:
                self.cur_pos[1] = self.cur_pos[1] - 1
            else:
                reward = -1.0
        elif action == 1:
            if self.cur_pos[1] < MAZE_SIZE - 1:
                self.cur_pos[1] = self.cur_pos[1] + 1
            else:
                reward = -1.0
        elif action == 2:
            if self.cur_pos[0] > 0:
                self.cur_pos[0] = self.cur_pos[0] - 1
            else:
                reward = -1.0
        elif action == 3:
            if self.cur_pos[0] < MAZE_SIZE - 1:
                self.cur_pos[0] = self.cur_pos[0] + 1
            else:
                reward = -1.0

        if (self.cur_pos == TRAP1_POS).all() or (self.cur_pos == TRAP2_POS).all():
            done = True
            reward = TRAP_VALUE
        elif (self.cur_pos == EXIT_POS).all():
            done = True
            reward = EXIT_VALUE

        return self.cur_pos, reward, done, {}

    def seed(self, seed=None):
        random.seed(seed)


class QuarkMaze2(gym.Env):
    def __init__(self, config: EnvContext):
        # 0:up, 1:down, 2:left, 3:right
        self.action_space = Discrete(4)
        self.observation_space = Box(TRAP_VALUE, EXIT_VALUE, shape=MAZE_SHAPE, dtype=np.float32)
        self.cur_pos = np.array([0, 0])
        self.seed((config.worker_index + 1) * (config.num_workers + 1))

    def create_obs(self):
        obs = np.zeros(MAZE_SHAPE)
        obs[TRAP1_POS[0]][TRAP1_POS[1]] = TRAP_VALUE
        obs[TRAP2_POS[0]][TRAP2_POS[1]] = TRAP_VALUE
        obs[EXIT_POS[0]][EXIT_POS[1]] = EXIT_VALUE
        obs[self.cur_pos[0]][self.cur_pos[1]] = POS_VALUE
        return obs

    def reset(self):
        self.cur_pos = np.array([0, 0])
        return self.create_obs()

    def step(self, action):
        done = False
        reward = 0.0

        # 0:up, 1:down, 2:left, 3:right
        # if action == 0 and self.cur_pos[1] > 0:
        #     self.cur_pos[1] = self.cur_pos[1] - 1
        # elif action == 1 and self.cur_pos[1] < MAZE_SIZE - 1:
        #     self.cur_pos[1] = self.cur_pos[1] + 1
        # elif action == 2 and self.cur_pos[0] > 0:
        #     self.cur_pos[0] = self.cur_pos[0] - 1
        # elif action == 3 and self.cur_pos[0] < MAZE_SIZE - 1:
        #     self.cur_pos[0] = self.cur_pos[0] + 1
        if action == 0:
            if self.cur_pos[0] > 0:
                self.cur_pos[0] = self.cur_pos[0] - 1
            else:
                reward = -1.0
        elif action == 1:
            if self.cur_pos[0] < MAZE_SIZE - 1:
                self.cur_pos[0] = self.cur_pos[0] + 1
            else:
                reward = -1.0
        elif action == 2:
            if self.cur_pos[1] > 0:
                self.cur_pos[1] = self.cur_pos[1] - 1
            else:
                reward = -1.0
        elif action == 3:
            if self.cur_pos[1] < MAZE_SIZE - 1:
                self.cur_pos[1] = self.cur_pos[1] + 1
            else:
                reward = -1.0

        if (self.cur_pos == TRAP1_POS).all() or (self.cur_pos == TRAP2_POS).all():
            done = True
            reward = TRAP_VALUE
        elif (self.cur_pos == EXIT_POS).all():
            done = True
            reward = EXIT_VALUE

        return self.create_obs(), reward, done, {}

    def seed(self, seed=None):
        random.seed(seed)

# 加入宝藏、PPO明显比DQN好，DQN在局部无限抖动
class QuarkMaze3(gym.Env):
    def __init__(self, config: EnvContext):
        # 0:up, 1:down, 2:left, 3:right
        self.action_space = Discrete(4)
        self.observation_space = Box(TRAP_VALUE, TREASURE_VALUE, shape=MAZE_SHAPE, dtype=np.float32)
        self.cur_pos = np.array([0, 0])
        self.seed((config.worker_index + 1) * (config.num_workers + 1))
        self.treasure_status = True
        self.epoch_reward = 0

    def create_obs(self):
        obs = np.zeros(MAZE_SHAPE)
        obs[TRAP1_POS[0]][TRAP1_POS[1]] = TRAP_VALUE
        obs[TRAP2_POS[0]][TRAP2_POS[1]] = TRAP_VALUE
        obs[EXIT_POS[0]][EXIT_POS[1]] = EXIT_VALUE
        if self.treasure_status:
            obs[TREASURE_POS[0]][TREASURE_POS[1]] = TREASURE_VALUE
        obs[self.cur_pos[0]][self.cur_pos[1]] = POS_VALUE
        return obs

    def reset(self):
        self.cur_pos = np.array([0, 0])
        self.treasure_status = True
        return self.create_obs()

    def step(self, action):
        done = False
        reward = 0.0

        # 0:up, 1:down, 2:left, 3:right
        # if action == 0 and self.cur_pos[1] > 0:
        #     self.cur_pos[1] = self.cur_pos[1] - 1
        # elif action == 1 and self.cur_pos[1] < MAZE_SIZE - 1:
        #     self.cur_pos[1] = self.cur_pos[1] + 1
        # elif action == 2 and self.cur_pos[0] > 0:
        #     self.cur_pos[0] = self.cur_pos[0] - 1
        # elif action == 3 and self.cur_pos[0] < MAZE_SIZE - 1:
        #     self.cur_pos[0] = self.cur_pos[0] + 1
        if action == 0:
            if self.cur_pos[0] > 0:
                self.cur_pos[0] = self.cur_pos[0] - 1
            else:
                reward = -1.0
        elif action == 1:
            if self.cur_pos[0] < MAZE_SIZE - 1:
                self.cur_pos[0] = self.cur_pos[0] + 1
            else:
                reward = -1.0
        elif action == 2:
            if self.cur_pos[1] > 0:
                self.cur_pos[1] = self.cur_pos[1] - 1
            else:
                reward = -1.0
        elif action == 3:
            if self.cur_pos[1] < MAZE_SIZE - 1:
                self.cur_pos[1] = self.cur_pos[1] + 1
            else:
                reward = -1.0

        if (self.cur_pos == TREASURE_POS).all() and self.treasure_status:
            self.treasure_status = False
            reward = TREASURE_VALUE
        elif (self.cur_pos == TRAP1_POS).all() or (self.cur_pos == TRAP2_POS).all():
            done = True
            reward = TRAP_VALUE
        elif (self.cur_pos == EXIT_POS).all():
            done = True
            reward = EXIT_VALUE

        self.epoch_reward += reward
        return self.create_obs(), reward, done, {}

    def seed(self, seed=None):
        random.seed(seed)


def test_env():
    env_config = EnvContext({}, 0, num_workers=1)
    env = QuarkMaze3(env_config)
    obs = env.reset()
    print(obs)
    obs, reward, done, info = env.step(1)
    print(obs, reward, done)
    obs, reward, done, info = env.step(1)
    print(obs, reward, done)


if __name__ == '__main__':
    test_env()

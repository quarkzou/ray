import gym
from gym.spaces import Discrete, Box
from ray.rllib.env.env_context import EnvContext
import numpy as np
import random
from ray import tune
import ray.rllib.agents.dqn as dqn

MAZE_SIZE = 4
MAZE_SHAPE = (4, 4)
TRAP1_POS = np.array([1, 1])
TRAP2_POS = np.array([2, 2])
EXIT_POS = np.array([2, 1])

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


def train():
    analysis = tune.run(
        "DQN",
        # stop={"training_iteration": 20},
        stop={"episode_reward_mean": 9.5},
        config={
            "env": QuarkMaze2,
            "framework": "tf2",
            "eager_tracing": True,
            "num_gpus": 0,
            "num_workers": 1,
        },
        # checkpoint_freq=2,
        checkpoint_at_end=True,
    )
    last_checkpoint1 = analysis.get_last_checkpoint()
    print(last_checkpoint1)
    return last_checkpoint1


def predict(last_checkpoint1):
    if last_checkpoint1 is None:
        last_checkpoint1 = "/Users/quarkzou/ray_results/DQN/DQN_QuarkMaze2_ce9a7_00000_0_2022-03-23_11-10-35/checkpoint_000020/checkpoint-20"

    config = dqn.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    config["framework"] = "tf2"
    config["eager_tracing"] = True

    agent = dqn.DQNTrainer(config=config, env=QuarkMaze2)
    agent.restore(last_checkpoint1)

    env_config = EnvContext({}, 0, num_workers=1)
    env = QuarkMaze2(env_config)
    obs = env.reset()
    done = False
    while not done:
        # env.render()
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        print(obs, action, done)

    env.close()


def test_env():
    env_config = EnvContext({}, 0, num_workers=1)
    env = QuarkMaze2(env_config)
    obs = env.reset()
    print(obs)
    obs, reward, done, info = env.step(1)
    print(obs, reward, done)


def main():
    # test_env()
    last_checkpoint1 = train()
    predict(last_checkpoint1)


if __name__ == '__main__':
    main()

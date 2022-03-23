import gym
from gym.spaces import Discrete, Box
from ray.rllib.env.env_context import EnvContext
import numpy as np
import random
from ray import tune
import ray.rllib.agents.dqn as dqn


MAZE_SIZE = 4
TRAP1_POS = np.array([1, 1])
TRAP2_POS = np.array([2, 2])
EXIT_POS = np.array([2, 1])

TRAP_VALUE = -10.0
EXIT_VALUE = 10


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


def train():
    analysis = tune.run(
        "DQN",
        stop={"training_iteration": 20},
        config={
            "env": QuarkMaze,
            "framework": "tf2",
            "eager_tracing": True,
            "num_gpus": 0,
            "num_workers": 4,
        },
        # checkpoint_freq=2,
        checkpoint_at_end=True,
    )


def predict():
    last_checkpoint1 = "/Users/quarkzou/ray_results/DQN/DQN_QuarkMaze_e28d0_00000_0_2022-03-23_10-35-21/checkpoint_000020/checkpoint-20"

    config = dqn.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 4
    config["framework"] = "tf2"
    config["eager_tracing"] = True

    agent = dqn.DQNTrainer(config=config, env=QuarkMaze)
    agent.restore(last_checkpoint1)

    env_config = EnvContext({}, 0, num_workers=1)
    env = QuarkMaze(env_config)
    obs = env.reset()
    done = False
    while not done:
        # env.render()
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        print(action, obs, done)

    env.close()


def main():
    # env_config = EnvContext({}, 0, num_workers=1)
    # env = QuarkMaze(env_config)
    # train()
    predict()


if __name__ == '__main__':
    main()

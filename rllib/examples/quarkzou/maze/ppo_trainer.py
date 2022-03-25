from ray.rllib.env.env_context import EnvContext
from ray import tune
import ray.rllib.agents.ppo as ppo
from maze_env import QuarkMaze3
import maze_env_4
from ray.rllib.policy.sample_batch import SampleBatch
import tensorflow as tf
import numpy as np


def train():
    analysis = tune.run(
        "PPO",
        # stop={"training_iteration": 20},
        stop={"episode_reward_mean": 29},
        config={
            "env": maze_env_4.QuarkMaze4,
            "framework": "tf2",
            "eager_tracing": True,
            "num_gpus": 0,
            "num_workers": 4,
            "lr": 0.001,
        },
        # checkpoint_freq=2,
        checkpoint_at_end=True,
    )

    last_checkpoint1 = analysis.get_last_checkpoint()
    print(last_checkpoint1)
    return last_checkpoint1


def predict(last_checkpoint1):
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    config["framework"] = "tf2"
    config["eager_tracing"] = True

    agent = ppo.PPOTrainer(config=config, env=maze_env_4.QuarkMaze4)
    agent.restore(last_checkpoint1)

    env_config = EnvContext({}, 0, num_workers=1)
    env = maze_env_4.QuarkMaze4(env_config)
    obs = env.reset()
    done = False
    while not done:
        # env.render()
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        print(obs.reshape((maze_env_4.MAZE_SIZE, maze_env_4.MAZE_SIZE)), action, done)

    env.close()


def model_policy(last_checkpoint1):
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    config["framework"] = "tf2"
    config["eager_tracing"] = True

    agent = ppo.PPOTrainer(config=config, env=maze_env_4.QuarkMaze4)
    agent.restore(last_checkpoint1)

    env_config = EnvContext({}, 0, num_workers=1)
    env = maze_env_4.QuarkMaze4(env_config)
    obs = env.reset()
    obs1 = env.step(1)[0]
    obs2 = env.step(1)[0]
    obs3 = env.step(1)[0]

    policy = agent.get_policy()
    w = policy.get_weights()
    logits, _ = policy.model.from_batch(SampleBatch({"obs": [obs, obs1, obs2, obs3]}))
    print(logits)
    dist = policy.dist_class(logits, policy.model)
    print(dist)
    print(dist.sample())
    print(policy.model.value_function())

    # policy.model.base_model.summary()


def main():
    # test_env()
    last_checkpoint1 = train()
    predict(last_checkpoint1)
    # last_checkpoint1 = "/Users/quarkzou/ray_results/PPO/PPO_QuarkMaze4_e961d_00000_0_2022-03-25_13-03-29/checkpoint_000012/checkpoint-12"
    # model_policy(last_checkpoint1)


if __name__ == '__main__':
    main()

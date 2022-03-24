from ray.rllib.env.env_context import EnvContext
from ray import tune
import ray.rllib.agents.ppo as ppo
from maze_env import QuarkMaze3
import tensorflow as tf


def train():
    analysis = tune.run(
        "PPO",
        # stop={"training_iteration": 20},
        stop={"episode_reward_mean": 29},
        config={
            "env": QuarkMaze3,
            "framework": "tf2",
            "eager_tracing": True,
            "num_gpus": 0,
            "num_workers": 4,
            "gamma": 0.90,
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

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 1
    config["num_workers"] = 1
    config["framework"] = "tf2"
    config["eager_tracing"] = True

    agent = ppo.PPOTrainer(config=config, env=QuarkMaze3)
    agent.restore(last_checkpoint1)

    env_config = EnvContext({}, 0, num_workers=1)
    env = QuarkMaze3(env_config)
    obs = env.reset()
    done = False
    while not done:
        # env.render()
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        print(obs, action, done)

    env.close()



def main():
    # test_env()
    last_checkpoint1 = train()
    predict(last_checkpoint1)


if __name__ == '__main__':
    main()

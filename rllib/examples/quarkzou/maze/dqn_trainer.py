from ray.rllib.env.env_context import EnvContext
from ray import tune
import ray.rllib.agents.dqn as dqn
from maze_env import QuarkMaze3
import maze_env_4
from ray.rllib.policy.sample_batch import SampleBatch


# 自定义停止迭代逻辑，平均reward > 28 或者训练次数 > 20
def stopper(trial_id, result):
    # return result["episode_reward_mean"] > 28 or result["training_iteration"] >= 20
    return result["episode_reward_mean"] > 28


def train():
    analysis = tune.run(
        "DQN",
        stop=stopper,
        config={
            "env": maze_env_4.QuarkMaze4,
            "framework": "tf2",
            "eager_tracing": True,
            "num_gpus": 0,
            "num_workers": 2,
            # "lr": tune.grid_search([0.01, 0.001, 0.0001]),
            "lr": 0.001,
        },
        # checkpoint_freq=2,
        checkpoint_at_end=True,
    )
    last_checkpoint1 = analysis.get_last_checkpoint()
    print(last_checkpoint1)
    return last_checkpoint1


def predict(last_checkpoint1):
    config = dqn.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    config["framework"] = "tf2"
    config["eager_tracing"] = True

    agent = dqn.DQNTrainer(config=config, env=maze_env_4.QuarkMaze4)
    agent.restore(last_checkpoint1)

    env_config = EnvContext({}, 0, num_workers=1)
    env = maze_env_4.QuarkMaze4(env_config)
    obs = env.reset()
    done = False
    while not done:
        # env.render()
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        print(obs[:maze_env_4.MAZE_SIZE * maze_env_4.MAZE_SIZE].reshape((maze_env_4.MAZE_SIZE, maze_env_4.MAZE_SIZE)),
              action, done)

    env.close()


def model_policy(last_checkpoint1):
    config = dqn.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    config["framework"] = "tf2"
    config["eager_tracing"] = True

    agent = dqn.DQNTrainer(config=config, env=maze_env_4.QuarkMaze4)
    agent.restore(last_checkpoint1)

    env_config = EnvContext({}, 0, num_workers=1)
    env = maze_env_4.QuarkMaze4(env_config)
    obs = env.reset()
    obs1 = env.step(1)[0]
    obs2 = env.step(1)[0]
    obs3 = env.step(1)[0]

    policy = agent.get_policy()
    model = policy.model
    # print(model.variables())
    # w = policy.get_weights()
    logits, _ = model.from_batch(SampleBatch({"obs": [obs, obs1, obs2, obs3]}))
    print(logits)
    dist = policy.dist_class(logits, model)
    print(dist)
    print(dist.sample())
    print(model.value_function())
    model.base_model.summary()


def main():
    # test_env()
    # last_checkpoint1 = train()
    # predict(last_checkpoint1)
    last_checkpoint1 = "/Users/quarkzou/ray_results/DQN/DQN_QuarkMaze4_e1e74_00000_0_2022-03-25_17-20-59/checkpoint_000022/checkpoint-22"
    # predict(last_checkpoint1)
    model_policy(last_checkpoint1)


if __name__ == '__main__':
    main()

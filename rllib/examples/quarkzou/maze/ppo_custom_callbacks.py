from ray.rllib.env.env_context import EnvContext
from ray import tune
import ray.rllib.agents.ppo as ppo
from maze_env import QuarkMaze3
import maze_env_4
from ray.rllib.policy.sample_batch import SampleBatch
import tensorflow as tf
import numpy as np
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.env import BaseEnv
from typing import Dict
from ray.rllib.policy import Policy


class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.model = FullyConnectedNetwork(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))


    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        pass

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.sampler.sample_collector.multiple_episodes_in_batch:
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )
        pole_angle = np.mean(episode.user_data["pole_angles"])
        print(
            "episode {} (env-idx={}) ended with length {} and pole "
            "angles {}".format(
                episode.episode_id, env_index, episode.length, pole_angle
            )
        )
        episode.custom_metrics["pole_angle"] = pole_angle
        episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
        print("returned sample batch of size {}".format(samples.count))

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        print(
            "trainer.train() result: {} -> {} episodes".format(
                trainer, result["episodes_this_iter"]
            )
        )
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True

    def on_learn_on_batch(
        self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    ) -> None:
        print(
            "policy.learn_on_batch() result: {} -> sum actions: {}".format(
                policy, result["sum_actions_in_train_batch"]
            )
        )

    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: Episode,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, SampleBatch],
        **kwargs
    ):
        print("postprocessed {} steps".format(postprocessed_batch.count))


# 自定义停止迭代逻辑，平均reward > 29 或者训练次数 > 20
def stopper(trial_id, result):
    return result["episode_reward_mean"] > 29 or result["training_iteration"] >= 20




def train():

    analysis = tune.run(
        "PPO",
        # stop={"training_iteration": 20},
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
        print(obs[:maze_env_4.MAZE_SIZE * maze_env_4.MAZE_SIZE].reshape((maze_env_4.MAZE_SIZE, maze_env_4.MAZE_SIZE)),
              action, done)

    env.close()


def model_policy(last_checkpoint1):
    ModelCatalog.register_custom_model(
        "my_model", CustomModel
    )

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    config["framework"] = "tf2"
    config["eager_tracing"] = True
    config["model"] = { "custom_model": "my_model", "vf_share_layers": True, }

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

    policy.model.model.base_model.summary()


def main():
    # test_env()
    # last_checkpoint1 = train()
    # predict(last_checkpoint1)
    last_checkpoint1 = "/root/ray_results/PPO/PPO_QuarkMaze4_eb972_00000_0_2022-03-25_20-48-50/checkpoint_000020/checkpoint-20"
    # predict(last_checkpoint1)
    model_policy(last_checkpoint1)


if __name__ == '__main__':
    main()

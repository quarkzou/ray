import gym
import ray
from ray import tune
import ray.rllib.agents.ppo as ppo

ray.init()

# tune.run(
#     "PPO",
#     stop={"episode_reward_mean": 100},
#     config={
#         "env": "CartPole-v0",
#         "num_gpus": 0,
#         "num_workers": 1,
#         "lr": tune.grid_search([0.01, 0.001, 0.0001]),
#     },
# )

# tune.run(
#     "PPO",
#     stop={"episode_reward_mean": 30},
#     config={
#         "env": "CartPole-v0",
#         "framework": "tf2",
#         "eager_tracing": True,
#         "num_gpus": 0,
#         "num_workers": 1,
#         "lr": tune.grid_search([0.01, 0.001, 0.0001]),
#     },
# )

# analysis = tune.run(
#     "PPO",
#     stop={"episode_reward_mean": 200},
#     config={
#         "env": "CartPole-v0",
#         "framework": "tf2",
#         "eager_tracing": True,
#         "num_gpus": 0,
#         "num_workers": 4,
#     },
#     # checkpoint_freq=2,
#     checkpoint_at_end=True,
# )

# list of lists: one list per checkpoint; each checkpoint list contains
# 1st the path, 2nd the metric value
# checkpoints = analysis.get_trial_checkpoints_paths(
#     trial=analysis.get_best_trial("episode_reward_mean", mode="max"),
#     metric="episode_reward_mean")
#
# # or simply get the last checkpoint (with highest "training_iteration")
# last_checkpoint1 = analysis.get_last_checkpoint()
# print(last_checkpoint1)
last_checkpoint1 = "/Users/quark/ray_results/PPO/PPO_CartPole-v0_6b19c_00000_0_2022-03-19_11-05-23/checkpoint_000014/checkpoint-14"
# last_checkpoint1 = "/Users/quark/ray_results/PPO/PPO_CartPole-v0_2f235_00000_0_2022-03-19_10-20-45/checkpoint_000002/checkpoint-2"

config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 4
config["framework"] = "tf2"
config["eager_tracing"] = True

agent = ppo.PPOTrainer(config=config, env="CartPole-v0")
agent.restore(last_checkpoint1)

env = gym.make('CartPole-v0')
obs = env.reset()
done = False
while True:
    env.render()
    action = agent.compute_action(obs)
    obs, reward, done, info = env.step(action)
    print(done)

env.close()

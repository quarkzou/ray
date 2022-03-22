from ray.rllib.agents.ppo import PPOTrainer

trainer = PPOTrainer(env="CartPole-v0", config={"framework": "tf2", "num_workers": 0})

policy = trainer.get_policy()
w = policy.get_weights()
# print(w)

s = policy.model.base_model.summary()
print(s)

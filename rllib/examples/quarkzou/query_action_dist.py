from ray.rllib.agents.ppo import PPOTrainer
import numpy as np

trainer = PPOTrainer(env="CartPole-v0", config={"framework": "tf2", "num_workers": 0})

policy = trainer.get_policy()
w = policy.get_weights()
logits, _ = policy.model.from_batch({"obs": np.array([[0.1, 0.2, 0.3, 0.4]])})
# print(w)

s = policy.model.base_model.summary()
print(s)

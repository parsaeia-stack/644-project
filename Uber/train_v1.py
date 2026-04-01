from stable_baselines3 import PPO
from environment_v1 import UberDialogueEnvV1

env = UberDialogueEnvV1()
#model = PPO("MlpPolicy", env, verbose=1)
#odel.learn(total_timesteps=5)
model = PPO("MlpPolicy", env, verbose=1, n_steps=128)
model.learn(total_timesteps=256)
model.save("uber_policy_llm_v1")
print("Training done!")
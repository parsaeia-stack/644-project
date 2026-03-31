from stable_baselines3 import PPO
from environment import UberDialogueEnv

# Create the environment
env = UberDialogueEnv()

# Create the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train for 100,000 steps
model.learn(total_timesteps=100000)

# Save the trained model
model.save("uber_policy_sparse")
print("Training done!")
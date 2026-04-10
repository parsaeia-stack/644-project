from stable_baselines3 import PPO
from environment import UberDialogueEnv
from logging_callback import LoggingCallback
from stable_baselines3.common.monitor import Monitor


env = Monitor(UberDialogueEnv())
model = PPO("MlpPolicy", env, verbose=1, n_steps=2048)
logging_cb = LoggingCallback(f"logs_sparse.csv")
model.learn(total_timesteps=200000, callback=logging_cb)
model.save(f"uber_policy_sparse")
print(f"Training done!")
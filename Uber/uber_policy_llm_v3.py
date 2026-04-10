from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from environment import UberDialogueEnv
from llm_reward import get_llm_rewards
from uber_policy_llm_v2 import LLMRewardCallback
from logging_callback import LoggingCallback


class HybridLLMRewardCallback(LLMRewardCallback):
    """
    This callback extends LLMRewardCallback to add additional logging
    or custom behavior if needed.
    """
    def _on_step(self):
        for i, done in enumerate(self.locals["dones"]):
            if done:
                env = self.training_env.envs[i].unwrapped
                timeout = env.turn >= env.max_turns
                if not timeout:
                    self.episode_count += 1
                    history = env.history
                    if len(history) == 0:
                        continue
                    try:
                        scores = get_llm_rewards(history)
                    except Exception as e:
                        print(f"LLM call failed: {e}, skipping episode")
                        continue
                    if len(scores) != len(history):
                        continue
                    buffer = self.model.rollout_buffer
                    total = buffer.pos
                    n_turns = len(scores)
                    for j, score in enumerate(scores):
                        idx = (total - n_turns + j) % buffer.buffer_size
                        buffer.rewards[idx] = score
                    if self.episode_count % 10 == 0:
                        print(f"Episode {self.episode_count} — LLM scores: {[round(s,2) for s in scores]}")
        return True




env = Monitor(UberDialogueEnv())

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1, n_steps=2048)

# Train with LLM reward callback
logging_cb = LoggingCallback(f"logs_v3.csv")
hybrid_cb = HybridLLMRewardCallback()
model.learn(total_timesteps=200000, callback=CallbackList([hybrid_cb, logging_cb]))


# Save
model.save(f"uber_policy_llm_v3")
print(f"Training done!")
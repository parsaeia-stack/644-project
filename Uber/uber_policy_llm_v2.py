from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from environment import UberDialogueEnv
from llm_reward import get_llm_rewards
from logging_callback import LoggingCallback

class LLMRewardCallback(BaseCallback):
    """
    This callback runs after each episode and replaces
    the sparse rewards in the trajectory with LLM scores.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self):
        # Check if any environment just finished an episode
        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_count += 1
                env = self.training_env.envs[i].unwrapped

                # Get the episode history
                history = env.history

                if len(history) == 0:
                    continue

                # Get LLM scores for this episode
                try:
                    scores = get_llm_rewards(history)
                except Exception as e:
                    print(f"LLM call failed: {e}, skipping episode")
                    continue

                # Make sure we got the right number of scores
                if len(scores) != len(history):
                    print(f"Score mismatch: {len(scores)} scores for {len(history)} turns, skipping")
                    continue

                # Replace rewards in the rollout buffer with LLM scores
                buffer = self.model.rollout_buffer
                total = buffer.pos  # how many steps are in the buffer right now
                n_turns = len(scores)

                # The last n_turns steps in the buffer belong to this episode
                for j, score in enumerate(scores):
                    idx = (total - n_turns + j) % buffer.buffer_size
                    buffer.rewards[idx] = score

                if self.episode_count % 10 == 0:
                    print(f"Episode {self.episode_count} — LLM scores: {[round(s,2) for s in scores]}")

        return True



env = Monitor(UberDialogueEnv())

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1, n_steps=2048)


logging_cb = LoggingCallback(f"logs_v2.csv")
llm_cb = LLMRewardCallback()
# Train with LLM reward callback
model.learn(total_timesteps=200000, callback=CallbackList([llm_cb, logging_cb]))

# Save
model.save(f"uber_policy_llm_v2")
print(f"Training done!")
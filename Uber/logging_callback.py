import csv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean


class LoggingCallback(BaseCallback):
    def __init__(self, log_file, verbose=0):
        super().__init__(verbose)
        self.log_file = log_file
        self.file = open(log_file, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["timesteps", "ep_rew_mean", "ep_len_mean"])

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        # Same source as SB3's dump_logs(); logger keys are not set until after this callback.
        buf = self.model.ep_info_buffer
        if buf is not None and len(buf) > 0 and len(buf[0]) > 0:
            self.writer.writerow(
                [
                    self.num_timesteps,
                    safe_mean([ep_info["r"] for ep_info in buf]),
                    safe_mean([ep_info["l"] for ep_info in buf]),
                ]
            )
            self.file.flush()

    def _on_training_end(self):
        self.file.close()

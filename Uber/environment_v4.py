from environment import UberDialogueEnv
from reward_model import RewardModelPredictor


class UberDialogueEnvV4(UberDialogueEnv):
    """
    Version 4: distilled hybrid reward model.

    A learned reward model predicts dense turn-level rewards from transition
    features, and the environment adds a terminal success bonus to preserve a
    strong task-completion signal.
    """

    def __init__(self, reward_model_path="reward_model_v4.pt", terminal_success_bonus=500.0):
        super().__init__()
        self.reward_predictor = RewardModelPredictor(reward_model_path)
        self.terminal_success_bonus = float(terminal_success_bonus)
        self.reward_strategy = "distilled_hybrid_reward_model"

    def step(self, action):
        obs, sparse_reward, terminated, truncated, info = super().step(action)
        del sparse_reward

        last_turn = self.history[-1]
        predicted_reward = self.reward_predictor.predict(
            state_before=last_turn["state"],
            action=action,
            user_response=last_turn["user_response"],
            state_after=obs.tolist(),
        )

        success_bonus = self.terminal_success_bonus if self._is_done() else 0.0
        reward = float(predicted_reward + success_bonus)

        info = dict(info)
        info["reward_source"] = self.reward_strategy
        info["predicted_reward"] = float(predicted_reward)
        info["success_bonus"] = float(success_bonus)
        info["used_reward"] = reward

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    env = UberDialogueEnvV4()
    obs, _ = env.reset()
    print("Version 4 reward strategy:", env.reward_strategy)
    print("Initial state:", obs)

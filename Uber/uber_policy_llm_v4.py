import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from environment_v4 import UberDialogueEnvV4
from logging_callback import LoggingCallback


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO with the distilled hybrid reward model (Version 4)."
    )
    parser.add_argument("--reward-model-path", default="reward_model_v4.pt")
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-path", default="uber_policy_llm_v4")
    parser.add_argument("--log-file", default="logs_v4.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--terminal-success-bonus", type=float, default=500.0)
    return parser.parse_args()


def main():
    args = parse_args()
    batch_size = args.batch_size or min(64, args.n_steps)
    env = Monitor(
        UberDialogueEnvV4(
            reward_model_path=args.reward_model_path,
            terminal_success_bonus=args.terminal_success_bonus,
        )
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=batch_size,
        seed=args.seed,
    )
    logging_cb = LoggingCallback(args.log_file)
    model.learn(total_timesteps=args.total_timesteps, callback=logging_cb)
    model.save(args.save_path)
    print(f"Training done! Saved model to {args.save_path}")


if __name__ == "__main__":
    main()

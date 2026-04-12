import argparse
import json
import random
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from environment import UberDialogueEnv
from llm_reward import get_llm_rewards


DEFAULT_POLICIES = ["random", "uber_policy_sparse", "uber_policy_llm_v2", "uber_policy_llm_v3"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect offline labels for the V4 distilled reward model."
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=DEFAULT_POLICIES,
        help="Policies to sample from. Use 'random' or saved PPO model names/paths.",
    )
    parser.add_argument("--episodes-per-policy", type=int, default=50)
    parser.add_argument("--output-path", default="v4_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-sparse-fallback",
        action="store_true",
        help="If an LLM scoring call fails on a successful episode, store sparse labels instead of skipping the episode.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def resolve_policy(policy_name):
    if policy_name == "random":
        return "random", None

    path = Path(policy_name)
    if not path.exists():
        path = path.with_suffix(".zip")
    if not path.exists():
        raise FileNotFoundError(f"Could not find policy: {policy_name}")

    return path.stem, PPO.load(str(path), device="cpu")


def choose_action(policy_name, policy, obs):
    if policy_name == "random":
        return None
    action, _ = policy.predict(obs, deterministic=True)
    return int(action)


def build_failure_labels(length):
    return [-5.0] * length


def collect_episode(env, policy_name, policy):
    obs, _ = env.reset()
    done = False
    truncated = False
    transitions = []

    while not (done or truncated):
        action = choose_action(policy_name, policy, obs)
        if action is None:
            action = env.action_space.sample()

        state_before = obs.tolist()
        next_obs, reward, done, truncated, _ = env.step(action)
        last_turn = env.history[-1]
        transitions.append(
            {
                "turn": last_turn["turn"],
                "state_before": state_before,
                "action": int(action),
                "action_name": last_turn["action"],
                "user_response": last_turn["user_response"],
                "state_after": next_obs.tolist(),
                "sparse_reward": float(reward),
            }
        )
        obs = next_obs

    success = env._is_done()
    history = [dict(item) for item in env.history]
    return transitions, history, success


def main():
    args = parse_args()
    set_seed(args.seed)
    output_path = Path(args.output_path)
    env = UberDialogueEnv()

    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for policy_offset, policy_name in enumerate(args.policies):
            resolved_name, policy = resolve_policy(policy_name)
            for episode_idx in range(args.episodes_per_policy):
                transitions, history, success = collect_episode(env, resolved_name, policy)
                if not transitions:
                    continue

                if success:
                    try:
                        labels = [float(value) for value in get_llm_rewards(history)]
                    except Exception as exc:
                        if not args.allow_sparse_fallback:
                            print(
                                f"Skipping successful episode from {resolved_name} "
                                f"because LLM scoring failed: {exc}"
                            )
                            skipped += 1
                            continue
                        labels = build_failure_labels(len(transitions))
                else:
                    labels = build_failure_labels(len(transitions))

                if len(labels) != len(transitions):
                    print(
                        f"Skipping episode from {resolved_name} due to label mismatch: "
                        f"{len(labels)} labels for {len(transitions)} transitions"
                    )
                    skipped += 1
                    continue

                reward_source = "llm_success" if success else "rule_failure"
                for transition, label in zip(transitions, labels):
                    record = {
                        "policy": resolved_name,
                        "episode_index": episode_idx,
                        "episode_success": success,
                        "reward_source": reward_source,
                        "turn": transition["turn"],
                        "state_before": transition["state_before"],
                        "action": transition["action"],
                        "action_name": transition["action_name"],
                        "user_response": transition["user_response"],
                        "state_after": transition["state_after"],
                        "label_reward": float(label),
                        "sparse_reward": transition["sparse_reward"],
                    }
                    handle.write(json.dumps(record) + "\n")
                    written += 1

    print(f"Wrote {written} labeled transitions to {output_path.resolve()}")
    print(f"Skipped episodes: {skipped}")


if __name__ == "__main__":
    main()

from stable_baselines3 import PPO
from environment import UberDialogueEnv
import numpy as np
from pathlib import Path

def evaluate(model, env, n_episodes=500, eval_seed=42):
    successes = 0
    total_length = 0

    for i in range(n_episodes):
        obs, _ = env.reset(seed=eval_seed + i)
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)

        if env._is_done():
            successes += 1
        total_length += env.turn

    success_rate = successes / n_episodes * 100
    avg_length = total_length / n_episodes
    timeout_rate = (n_episodes - successes) / n_episodes * 100

    return success_rate, avg_length, timeout_rate


model_specs = [
    ("Sparse", "uber_policy_sparse"),
    ("Version 2 (LLM whole)", "uber_policy_llm_v2"),
    ("Version 3 (Hybrid)", "uber_policy_llm_v3"),
]

if Path("uber_policy_llm_v4.zip").exists():
    model_specs.append(("Version 4 (Distilled Hybrid)", "uber_policy_llm_v4"))

for name, model_path in model_specs:
    env = UberDialogueEnv()
    model = PPO.load(model_path, env=env)
    sr, al, tr = evaluate(model, env)

    print(f"\n--- {name} ---")
    print(f"Success Rate: {sr:.1f}%")
    print(f"Average Dialogue Length: {al:.1f} turns")
    print(f"Timeout Rate: {tr:.1f}%")

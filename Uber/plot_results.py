import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_single(prefix):
    df = pd.read_csv(f"logs_{prefix}.csv")
    return df["timesteps"].values, df["ep_rew_mean"].values, df["ep_len_mean"].values

sparse_steps, sparse_rew, sparse_len = load_single("sparse")
v2_steps, v2_rew, v2_len = load_single("v2")
v3_steps, v3_rew, v3_len = load_single("v3")
has_v4 = Path("logs_v4.csv").exists()
if has_v4:
    v4_steps, v4_rew, v4_len = load_single("v4")

# Plot reward
plt.figure(figsize=(12, 6))
plt.plot(sparse_steps, sparse_rew, label="Sparse", color="blue", linewidth=2)
plt.plot(v2_steps, v2_rew, label="LLM Whole Conversation", color="green", linewidth=2, linestyle="--")
plt.plot(v3_steps, v3_rew, label="Hybrid", color="orange", linewidth=2, linestyle=":")
if has_v4:
    plt.plot(v4_steps, v4_rew, label="V4 Distilled Hybrid", color="red", linewidth=2, linestyle="-.")
plt.xlabel("Training Steps", fontsize=12)
plt.ylabel("Mean Episode Reward", fontsize=12)
plt.title("Learning Curve — Episode Reward", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_reward.png", dpi=150)
plt.show()

# Plot length
plt.figure(figsize=(12, 6))
plt.plot(sparse_steps, sparse_len, label="Sparse", color="blue", linewidth=2)
plt.plot(v2_steps, v2_len, label="LLM Whole Conversation", color="green", linewidth=2, linestyle="--")
plt.plot(v3_steps, v3_len, label="Hybrid", color="orange", linewidth=2, linestyle=":")
if has_v4:
    plt.plot(v4_steps, v4_len, label="V4 Distilled Hybrid", color="red", linewidth=2, linestyle="-.")
plt.xlabel("Training Steps", fontsize=12)
plt.ylabel("Mean Episode Length (turns)", fontsize=12)
plt.title("Learning Curve — Dialogue Length", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_length.png", dpi=150)
plt.show()

print("Plots saved!")

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from environment import ACTIONS
from simulated_user import SLOT_VALUES


SLOTS = list(SLOT_VALUES.keys())
USER_RESPONSES = (
    [f"provide_{slot}" for slot in SLOTS]
    + [f"provide_wrong_{slot}" for slot in SLOTS]
    + [f"confirm_pos_{slot}" for slot in SLOTS]
    + [f"confirm_neg_{slot}" for slot in SLOTS]
    + ["irrelevant"]
)

ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}
USER_RESPONSE_TO_INDEX = {response: idx for idx, response in enumerate(USER_RESPONSES)}

STATE_DIM = 12
ACTION_DIM = len(ACTIONS)
USER_RESPONSE_DIM = len(USER_RESPONSES)
INPUT_DIM = STATE_DIM + ACTION_DIM + USER_RESPONSE_DIM + STATE_DIM


def encode_transition(state_before, action, user_response, state_after):
    before = np.asarray(state_before, dtype=np.float32)
    after = np.asarray(state_after, dtype=np.float32)
    if before.shape != (STATE_DIM,) or after.shape != (STATE_DIM,):
        raise ValueError(
            f"Expected state vectors of length {STATE_DIM}, got "
            f"{before.shape} and {after.shape}."
        )

    if isinstance(action, str):
        action_index = ACTION_TO_INDEX[action]
    else:
        action_index = int(action)
    if action_index < 0 or action_index >= ACTION_DIM:
        raise ValueError(f"Action index out of range: {action_index}")

    response_index = USER_RESPONSE_TO_INDEX.get(user_response)
    if response_index is None:
        raise ValueError(f"Unknown user response: {user_response}")

    action_one_hot = np.zeros(ACTION_DIM, dtype=np.float32)
    action_one_hot[action_index] = 1.0

    response_one_hot = np.zeros(USER_RESPONSE_DIM, dtype=np.float32)
    response_one_hot[response_index] = 1.0

    return np.concatenate([before, action_one_hot, response_one_hot, after]).astype(
        np.float32
    )


class RewardMLP(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def save_reward_model(path, model, hidden_dim, target_mean, target_std):
    payload = {
        "state_dict": model.state_dict(),
        "hidden_dim": hidden_dim,
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "input_dim": INPUT_DIM,
        "actions": ACTIONS,
        "user_responses": USER_RESPONSES,
    }
    torch.save(payload, path)


def load_reward_model(path, device="cpu"):
    payload = torch.load(path, map_location=device)
    model = RewardMLP(
        input_dim=payload.get("input_dim", INPUT_DIM),
        hidden_dim=payload["hidden_dim"],
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


class RewardModelPredictor:
    def __init__(self, model_path, device="cpu"):
        self.model_path = Path(model_path)
        self.device = device
        self.model, self.metadata = load_reward_model(self.model_path, device=device)
        self.target_mean = float(self.metadata.get("target_mean", 0.0))
        self.target_std = float(self.metadata.get("target_std", 1.0))
        if self.target_std == 0:
            self.target_std = 1.0

    def predict(self, state_before, action, user_response, state_after):
        features = encode_transition(state_before, action, user_response, state_after)
        tensor = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            normalized_reward = float(self.model(tensor).item())
        return normalized_reward * self.target_std + self.target_mean


def load_jsonl_dataset(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

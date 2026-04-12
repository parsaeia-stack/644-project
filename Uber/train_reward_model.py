import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from reward_model import (
    RewardMLP,
    encode_transition,
    load_jsonl_dataset,
    save_reward_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the offline reward model used by Version 4."
    )
    parser.add_argument("--dataset-path", default="v4_dataset.jsonl")
    parser.add_argument("--model-out", default="reward_model_v4.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_tensors(records):
    features = []
    labels = []
    for record in records:
        features.append(
            encode_transition(
                state_before=record["state_before"],
                action=record["action"],
                user_response=record["user_response"],
                state_after=record["state_after"],
            )
        )
        labels.append(float(record["label_reward"]))

    x = torch.tensor(np.asarray(features), dtype=torch.float32)
    y = torch.tensor(np.asarray(labels), dtype=torch.float32)
    return x, y


def split_dataset(x, y, val_split, seed):
    indices = list(range(len(x)))
    random.Random(seed).shuffle(indices)

    val_count = max(1, int(len(indices) * val_split))
    train_indices = indices[val_count:]
    val_indices = indices[:val_count]

    if not train_indices:
        raise ValueError("Dataset too small after validation split.")

    train_x = x[train_indices]
    train_y = y[train_indices]
    val_x = x[val_indices]
    val_y = y[val_indices]
    return train_x, train_y, val_x, val_y


def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            predictions = model(batch_x)
            losses.append(float(criterion(predictions, batch_y).item()))
    return sum(losses) / max(1, len(losses))


def main():
    args = parse_args()
    set_seed(args.seed)

    records = load_jsonl_dataset(args.dataset_path)
    if not records:
        raise ValueError(f"No records found in {args.dataset_path}")

    x, y = build_tensors(records)
    train_x, train_y, val_x, val_y = split_dataset(x, y, args.val_split, args.seed)

    target_mean = float(train_y.mean().item())
    target_std = float(train_y.std().item())
    if target_std == 0:
        target_std = 1.0

    train_y_norm = (train_y - target_mean) / target_std
    val_y_norm = (val_y - target_mean) / target_std

    train_loader = DataLoader(
        TensorDataset(train_x, train_y_norm),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y_norm),
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RewardMLP(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.SmoothL1Loss()

    best_val_loss = None
    best_state_dict = None

    print(
        f"Training reward model on {len(train_x)} train transitions and "
        f"{len(val_x)} validation transitions."
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    if best_state_dict is None:
        raise RuntimeError("Reward model training did not produce a checkpoint.")

    model.load_state_dict(best_state_dict)
    save_reward_model(
        path=Path(args.model_out),
        model=model.cpu(),
        hidden_dim=args.hidden_dim,
        target_mean=target_mean,
        target_std=target_std,
    )

    print(f"Saved reward model to {Path(args.model_out).resolve()}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

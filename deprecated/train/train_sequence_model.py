#!/usr/bin/env python3
"""Train a Transformer-based activity classifier over event sequences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.data import load_events
from model.sequence_dataset import SequenceSamples, build_sequence_samples
from model.sequence_model import SequenceModelConfig, SensorSequenceModel
from utils.profiling import Timer


class ActivitySequenceDataset(Dataset):
    def __init__(self, samples: SequenceSamples, indices: np.ndarray):
        self.samples = samples
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base = self.indices[idx]
        return {
            "sensor": torch.from_numpy(self.samples.sensor_seq[base]).long(),
            "state": torch.from_numpy(self.samples.state_seq[base]).long(),
            "value_type": torch.from_numpy(self.samples.value_type_seq[base]).long(),
            "numeric": torch.from_numpy(self.samples.numeric_seq[base]).float(),
            "numeric_mask": torch.from_numpy(self.samples.numeric_mask_seq[base]).float(),
            "time": torch.from_numpy(self.samples.time_features_seq[base]).float(),
            "label": torch.tensor(self.samples.labels[base], dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer activity classifier.")
    parser.add_argument("--events-csv", type=Path, default=Path("data/processed/events.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint/activity_transformer.pt"))
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--sensor-embed-dim", type=int, default=64)
    parser.add_argument("--state-embed-dim", type=int, default=16)
    parser.add_argument("--value-type-embed-dim", type=int, default=8)
    parser.add_argument("--numeric-feature-dim", type=int, default=16)
    parser.add_argument("--time-feature-dim", type=int, default=16)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument(
        "--sensor-embedding-checkpoint",
        type=Path,
        default=None,
        help="Optional skip-gram checkpoint (.pt or .npz) for sensor embeddings.",
    )
    parser.add_argument("--freeze-sensor-embedding", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    timers: Dict[str, float] = {}

    with Timer("load_events_sec") as t:
        events = load_events(args.events_csv)
    timers["load_events_sec"] = t.stop()

    with Timer("build_sequences_sec") as t:
        samples = build_sequence_samples(
            events, window_size=args.window_size, max_samples=args.max_samples
        )
    timers["build_sequences_sec"] = t.stop()

    num_samples = len(samples.labels)
    train_end = max(1, int(num_samples * args.train_ratio))
    val_end = max(train_end + 1, int(num_samples * (args.train_ratio + args.val_ratio)))
    val_end = min(val_end, num_samples - 1)
    train_indices = np.arange(0, train_end)
    val_indices = np.arange(train_end, val_end)
    test_indices = np.arange(val_end, num_samples)

    train_dataset = ActivitySequenceDataset(samples, train_indices)
    val_dataset = ActivitySequenceDataset(samples, val_indices)
    test_dataset = ActivitySequenceDataset(samples, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sensor_embedding_tensor = None
    if args.sensor_embedding_checkpoint is not None:
        sensor_embedding_tensor = load_sensor_embeddings(
            args.sensor_embedding_checkpoint, samples.sensor_vocab
        )
        embed_dim = sensor_embedding_tensor.shape[1]
        if embed_dim != args.sensor_embed_dim:
            print(
                f"Overriding sensor embedding dim to {embed_dim} to match checkpoint "
                f"(was {args.sensor_embed_dim})"
            )
            args.sensor_embed_dim = embed_dim

    config = SequenceModelConfig(
        sensor_embed_dim=args.sensor_embed_dim,
        state_embed_dim=args.state_embed_dim,
        value_type_embed_dim=args.value_type_embed_dim,
        numeric_feature_dim=args.numeric_feature_dim,
        time_feature_dim=args.time_feature_dim,
        model_dim=args.model_dim,
        ff_dim=args.ff_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    model = SensorSequenceModel(
        num_sensors=len(samples.sensor_vocab),
        num_states=len(samples.state_vocab),
        num_value_types=len(samples.value_type_vocab),
        num_classes=len(samples.label_names),
        window_size=args.window_size,
        config=config,
        sensor_embedding_init=sensor_embedding_tensor,
    ).to(device)
    if args.freeze_sensor_embedding and sensor_embedding_tensor is not None:
        model.sensor_emb.weight.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    class_weights = compute_class_weights(samples.labels, len(samples.label_names))
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    best_state = None
    best_val_score = -float("inf")
    epoch_logs: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_clip=args.grad_clip,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, samples.label_names)

        epoch_time = time.perf_counter() - epoch_start

        epoch_logs.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "epoch_time_sec": epoch_time,
            }
        )

        if val_metrics["macro_f1"] > best_val_score:
            best_val_score = val_metrics["macro_f1"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_macro_f1": best_val_score,
            }
            save_checkpoint(args.checkpoint, model, samples, config, best_state, args)
            print(
                f"[Epoch {epoch}] New best macro F1: {best_val_score:.4f} "
                f"(accuracy={val_metrics['accuracy']:.3f}, time={epoch_time:.1f}s)"
            )
        else:
            print(
                f"[Epoch {epoch}] train_loss={train_loss:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"time={epoch_time:.1f}s"
            )

    if best_state is None:
        save_checkpoint(args.checkpoint, model, samples, config, None, args)
    else:
        model.load_state_dict(best_state["model_state_dict"])

    train_metrics = evaluate(model, eval_train_loader, criterion, device, samples.label_names)
    val_metrics = evaluate(model, val_loader, criterion, device, samples.label_names)
    test_metrics = evaluate(model, test_loader, criterion, device, samples.label_names)

    timers["total_train_sec"] = sum(log.get("epoch_time_sec", 0.0) for log in epoch_logs)

    metrics_path = args.checkpoint.with_suffix(".metrics.json")
    metrics_payload = {
        "train": summarize_metrics(train_metrics),
        "val": summarize_metrics(val_metrics),
        "test": summarize_metrics(test_metrics),
        "epochs": epoch_logs,
        "timers": timers,
        "config": {
            "window_size": args.window_size,
            "model": config.__dict__,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "sensor_embedding_checkpoint": str(args.sensor_embedding_checkpoint)
            if args.sensor_embedding_checkpoint
            else None,
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    print(f"Wrote metrics to {metrics_path}")
    return 0


def train_one_epoch(
    model: SensorSequenceModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch["sensor"].to(device),
            batch["state"].to(device),
            batch["value_type"].to(device),
            batch["numeric"].to(device),
            batch["numeric_mask"].to(device),
            batch["time"].to(device),
        )
        labels = batch["label"].to(device)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
    return total_loss / max(1, total_samples)


def evaluate(
    model: SensorSequenceModel,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    label_names: List[str],
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    preds: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["sensor"].to(device),
                batch["state"].to(device),
                batch["value_type"].to(device),
                batch["numeric"].to(device),
                batch["numeric_mask"].to(device),
                batch["time"].to(device),
            )
            labels = batch["label"].to(device)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    if total_samples == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "report": "No samples.",
        }

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels_all)
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    labels = list(range(len(label_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        zero_division=0,
    )
    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
    }


def summarize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "loss": metrics.get("loss", 0.0),
        "accuracy": metrics.get("accuracy", 0.0),
        "macro_f1": metrics.get("macro_f1", 0.0),
        "weighted_f1": metrics.get("weighted_f1", 0.0),
    }


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor | None:
    counts = np.bincount(labels, minlength=num_classes)
    valid = counts > 0
    if not valid.any():
        return None
    total = counts[valid].sum()
    weights = np.zeros_like(counts, dtype=np.float32)
    weights[valid] = total / counts[valid]
    weights = weights / weights[valid].mean()
    return torch.tensor(weights, dtype=torch.float32)


def save_checkpoint(
    checkpoint_path: Path,
    model: SensorSequenceModel,
    samples: SequenceSamples,
    config: SequenceModelConfig,
    state_dict_bundle: Dict[str, Any] | None,
    args: argparse.Namespace,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "window_size": samples.window_size,
        "sensor_vocab": samples.sensor_vocab,
        "state_vocab": samples.state_vocab,
        "value_type_vocab": samples.value_type_vocab,
        "label_names": samples.label_names,
        "numeric_mean": samples.numeric_mean,
        "numeric_std": samples.numeric_std,
        "train_args": vars(args),
    }
    if state_dict_bundle:
        payload["training_state"] = state_dict_bundle
    torch.save(payload, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def load_sensor_embeddings(checkpoint_path: Path, sensor_vocab: List[str]) -> torch.Tensor:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Sensor embedding checkpoint not found: {checkpoint_path}")

    if path.suffix == ".pt":
        data = torch.load(path, map_location="cpu")
        embeddings = data["embeddings"]
        if isinstance(embeddings, torch.Tensor):
            embedding_matrix = embeddings.cpu().numpy()
        else:
            embedding_matrix = np.asarray(embeddings, dtype=np.float32)
        vocab = data.get("sensor_vocab")
    elif path.suffix == ".npz":
        npz = np.load(path, allow_pickle=True)
        embedding_matrix = npz["embeddings"]
        vocab = npz["sensor_vocab"].tolist()
    else:
        raise ValueError(f"Unsupported embedding checkpoint format: {path}")

    vocab_index = {name: idx for idx, name in enumerate(vocab)}
    emb_dim = embedding_matrix.shape[1]
    aligned = np.random.normal(scale=0.01, size=(len(sensor_vocab), emb_dim)).astype(np.float32)
    hits = 0
    for i, sensor in enumerate(sensor_vocab):
        idx = vocab_index.get(sensor)
        if idx is not None:
            aligned[i] = embedding_matrix[idx]
            hits += 1
    print(f"Loaded sensor embeddings ({hits}/{len(sensor_vocab)} matched).")
    return torch.from_numpy(aligned)


if __name__ == "__main__":
    raise SystemExit(main())

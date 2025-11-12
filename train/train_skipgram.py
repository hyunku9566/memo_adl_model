#!/usr/bin/env python3
"""Train skip-gram embeddings over the sensor event stream."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.data import load_events
from model.skipgram import SkipGramConfig, build_token_sequence, extract_embeddings, train_skipgram
from utils.profiling import Timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train skip-gram sensor embeddings.")
    parser.add_argument(
        "--events-csv",
        type=Path,
        default=Path("data/processed/events.csv"),
        help="Path to the normalized events CSV.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoint/sensor_embeddings.pt"),
        help="Destination for the trained embeddings (PyTorch .pt).",
    )
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--context-size", type=int, default=5)
    parser.add_argument("--negatives", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional cap on tokens used.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timers = {}

    print(f"Loading events from {args.events_csv} ...")
    with Timer("load_events") as t:
        events = load_events(args.events_csv)
    timers["load_events_sec"] = t.stop()

    print("Building token sequence ...")
    with Timer("build_tokens") as t:
        tokens = build_token_sequence(events, max_tokens=args.max_tokens)
    timers["build_tokens_sec"] = t.stop()

    print("Training skip-gram embeddings ...")
    config = SkipGramConfig(
        embedding_dim=args.embedding_dim,
        context_size=args.context_size,
        negatives=args.negatives,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    with Timer("train_skipgram_sec") as t:
        model, stats = train_skipgram(tokens, vocab_size=len(events.sensor_vocab), config=config)
    timers["train_skipgram_sec"] = t.stop()

    embeddings = extract_embeddings(model)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch  # type: ignore
    except ImportError:
        torch = None  # type: ignore

    if torch is not None and hasattr(embeddings, "detach"):
        embeddings_tensor = embeddings.detach().cpu()
        embeddings_np = embeddings_tensor.numpy()
    else:
        embeddings_tensor = None
        embeddings_np = np.array(embeddings, dtype=np.float32)

    payload = {
        "embeddings": embeddings_tensor if embeddings_tensor is not None else embeddings_np,
        "sensor_vocab": events.sensor_vocab,
        "config": config.__dict__,
        "stats": [stat.__dict__ for stat in stats],
    }

    if torch is not None:
        torch.save(payload, checkpoint_path)
        print(f"Saved embeddings to {checkpoint_path}")
    else:
        np.savez(
            checkpoint_path.with_suffix(".npz"),
            embeddings=embeddings_np,
            sensor_vocab=np.array(events.sensor_vocab, dtype=object),
            config=np.array([config.__dict__], dtype=object),
            stats=np.array([stat.__dict__ for stat in stats], dtype=object),
        )
        print(
            "PyTorch not available – wrote NumPy checkpoint "
            f"to {checkpoint_path.with_suffix('.npz')}"
        )

    metrics_path = checkpoint_path.with_suffix(".metrics.json")
    metrics = {"timers": timers, "epochs": [stat.__dict__ for stat in stats]}
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote training metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

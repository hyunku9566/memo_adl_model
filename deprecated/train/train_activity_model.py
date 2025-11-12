#!/usr/bin/env python3
"""Train the baseline activity classifier on processed sensor events."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.activity_model import ModelConfig, evaluate_model, train_activity_model
from model.data import load_events
from model.features import build_feature_set
from utils.profiling import Timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a gradient boosted activity classifier.")
    parser.add_argument(
        "--events-csv",
        type=Path,
        default=Path("data/processed/events.csv"),
        help="Path to the normalized events CSV.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoint/activity_model.joblib"),
        help="Output path for the trained model artifact.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=25,
        help="Number of most recent events to use as context for each label.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of labeled samples (for quick experiments).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split fraction.")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split fraction (remainder goes to test).",
    )
    parser.add_argument("--alpha", type=float, default=1e-4, help="L2 regularization strength.")
    parser.add_argument("--max-iter", type=int, default=60, help="Number of SGD epochs.")
    parser.add_argument("--tol", type=float, default=1e-4, help="Early-stopping tolerance.")
    parser.add_argument(
        "--n-iter-no-change",
        type=int,
        default=5,
        help="Patience (in epochs) before early stopping.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def split_dataset(
    X: np.ndarray, y: np.ndarray, train_ratio: float, val_ratio: float
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    n_samples = len(y)
    train_end = max(1, int(n_samples * train_ratio))
    val_end = max(train_end + 1, int(n_samples * (train_ratio + val_ratio)))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def metrics_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "samples": metrics["samples"],
        "report": metrics["report"],
    }


def main() -> int:
    args = parse_args()
    timers = {}

    print(f"Loading events from {args.events_csv} ...")
    with Timer("load_events") as t:
        events = load_events(args.events_csv)
    timers["load_events_sec"] = t.stop()

    with Timer("build_features") as t:
        feature_set, feature_meta = build_feature_set(
            events, window_size=args.window_size, max_samples=args.max_samples
        )
    timers["build_features_sec"] = t.stop()

    print(
        f"Prepared {len(feature_set.labels):,} labeled samples "
        f"with feature dim {feature_meta.feature_dim} (window={feature_meta.window_size})."
    )

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(
        feature_set.features, feature_set.labels, args.train_ratio, args.val_ratio
    )
    print(
        "Split sizes: "
        f"train={len(y_train):,}, val={len(y_val):,}, test={len(y_test):,}"
    )

    config = ModelConfig(
        alpha=args.alpha,
        max_iter=args.max_iter,
        tol=args.tol,
        n_iter_no_change=args.n_iter_no_change,
        random_state=args.random_state,
    )
    print(
        "Training multinomial logistic regression "
        f"(alpha={config.alpha}, max_iter={config.max_iter}, tol={config.tol}) ..."
    )
    train_start = time.perf_counter()
    clf = train_activity_model(X_train, y_train, config)
    timers["train_model_sec"] = time.perf_counter() - train_start

    train_metrics = evaluate_model(clf, X_train, y_train, feature_meta.label_names)
    val_metrics = evaluate_model(clf, X_val, y_val, feature_meta.label_names)
    test_metrics = evaluate_model(clf, X_test, y_test, feature_meta.label_names)

    print(f"Train accuracy: {train_metrics['accuracy']:.3f}, macro F1: {train_metrics['macro_f1']:.3f}")
    print(f"Val accuracy:   {val_metrics['accuracy']:.3f}, macro F1: {val_metrics['macro_f1']:.3f}")
    print(f"Test accuracy:  {test_metrics['accuracy']:.3f}, macro F1: {test_metrics['macro_f1']:.3f}")

    print("\nValidation classification report:")
    print(val_metrics["report"])

    print("\nTest classification report:")
    print(test_metrics["report"])

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": clf,
        "metadata": {
            "window_size": feature_meta.window_size,
            "feature_dim": feature_meta.feature_dim,
            "sensor_vocab": feature_meta.sensor_vocab,
            "state_vocab": feature_meta.state_vocab,
            "value_type_vocab": feature_meta.value_type_vocab,
            "label_names": feature_meta.label_names,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
        },
    }
    with Timer("save_checkpoint") as t:
        joblib.dump(artifact, args.checkpoint)
    timers["save_checkpoint_sec"] = t.stop()
    print(f"Saved model artifact to {args.checkpoint}")

    metrics_payload = {
        "train": metrics_summary(train_metrics),
        "val": metrics_summary(val_metrics),
        "test": metrics_summary(test_metrics),
        "timers": timers,
    }
    metrics_path = args.checkpoint.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    print(f"Wrote metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

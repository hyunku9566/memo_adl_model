from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score


@dataclass
class ModelConfig:
    alpha: float = 1e-4
    max_iter: int = 60
    tol: float = 1e-4
    n_iter_no_change: int = 5
    random_state: int = 42


def train_activity_model(
    X: np.ndarray,
    y: np.ndarray,
    config: ModelConfig,
) -> SGDClassifier:
    """Fit a multinomial logistic regression via SGD."""

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=config.alpha,
        max_iter=config.max_iter,
        tol=config.tol,
        n_iter_no_change=config.n_iter_no_change,
        class_weight="balanced",
        learning_rate="optimal",
        random_state=config.random_state,
    )
    clf.fit(X, y)
    return clf


def evaluate_model(
    clf: SGDClassifier,
    X: np.ndarray,
    y: np.ndarray,
    label_names: List[str],
) -> Dict[str, float]:
    """Compute accuracy and F1 metrics plus a short classification report."""

    if X.size == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "samples": 0,
            "report": "No samples available for evaluation.",
        }

    preds = clf.predict(X)
    accuracy = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro")
    weighted_f1 = f1_score(y, preds, average="weighted")
    labels = list(range(len(label_names)))
    report = classification_report(
        y,
        preds,
        labels=labels,
        target_names=label_names,
        zero_division=0,
        output_dict=False,
    )
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "samples": int(len(y)),
        "report": report,
    }

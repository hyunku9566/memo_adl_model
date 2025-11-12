from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - torch is optional
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    TORCH_AVAILABLE = False

from model.data import EventData


@dataclass
class SkipGramConfig:
    embedding_dim: int = 64
    context_size: int = 4
    negatives: int = 5
    batch_size: int = 4096
    epochs: int = 5
    learning_rate: float = 0.01
    max_tokens: int | None = None
    seed: int = 42


@dataclass
class SkipGramStats:
    epoch: int
    loss: float
    steps: int
    pairs_processed: int
    elapsed_sec: float


if TORCH_AVAILABLE:

    class SkipGramModel(nn.Module):
        def __init__(self, vocab_size: int, embedding_dim: int) -> None:
            super().__init__()
            self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
            nn.init.uniform_(self.input_embeddings.weight, -0.5 / embedding_dim, 0.5 / embedding_dim)
            nn.init.zeros_(self.output_embeddings.weight)

        def forward(
            self, center_ids: torch.Tensor, context_ids: torch.Tensor, negative_ids: torch.Tensor
        ) -> torch.Tensor:
            center_vecs = self.input_embeddings(center_ids)  # (batch, dim)
            pos_vecs = self.output_embeddings(context_ids)  # (batch, dim)
            pos_score = torch.sum(center_vecs * pos_vecs, dim=1)
            pos_loss = F.logsigmoid(pos_score)

            neg_vecs = self.output_embeddings(negative_ids)  # (batch, negatives, dim)
            neg_score = torch.bmm(neg_vecs, center_vecs.unsqueeze(2)).squeeze(2)
            neg_loss = F.logsigmoid(-neg_score).sum(dim=1)

            loss = -(pos_loss + neg_loss).mean()
            return loss

else:  # pragma: no cover - only used for type checking when torch is absent

    class SkipGramModel:  # type: ignore
        pass


@dataclass
class SkipGramNumpyModel:
    """Lightweight numpy-based skip-gram used when torch is unavailable."""

    input_embeddings: np.ndarray
    output_embeddings: np.ndarray


def build_token_sequence(events: EventData, max_tokens: int | None = None) -> List[int]:
    tokens = events.sensor_ids
    if max_tokens is not None:
        return tokens[:max_tokens]
    return tokens


def _context_window(tokens: Sequence[int], idx: int, radius: int) -> List[int]:
    start = max(0, idx - radius)
    end = min(len(tokens), idx + radius + 1)
    context = []
    for j in range(start, end):
        if j == idx:
            continue
        context.append(tokens[j])
    return context


def _build_unigram_probs(tokens: Sequence[int], vocab_size: int, power: float = 0.75) -> np.ndarray:
    counts = np.zeros(vocab_size, dtype=np.float64)
    for token in tokens:
        counts[token] += 1
    probs = np.power(counts, power)
    total = probs.sum()
    if total == 0:
        probs[:] = 1.0 / max(1, vocab_size)
    else:
        probs /= total
    return probs.astype(np.float32)


def train_skipgram(
    tokens: Sequence[int],
    vocab_size: int,
    config: SkipGramConfig,
    device: Any | None = None,
) -> Tuple[object, List[SkipGramStats]]:
    if len(tokens) == 0:
        raise ValueError("Token sequence is empty; cannot train skip-gram embeddings.")

    if TORCH_AVAILABLE:
        return _train_skipgram_torch(tokens, vocab_size, config, device)
    return _train_skipgram_numpy(tokens, vocab_size, config)


def _train_skipgram_torch(
    tokens: Sequence[int],
    vocab_size: int,
    config: SkipGramConfig,
    device: Any | None = None,
) -> Tuple[SkipGramModel, List[SkipGramStats]]:
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SkipGramModel(vocab_size, config.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    unigram_probs = _build_unigram_probs(tokens, vocab_size)
    unigram_table = torch.tensor(unigram_probs, dtype=torch.float32, device=device)

    stats: List[SkipGramStats] = []
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        running_loss = 0.0
        steps = 0
        pairs_processed = 0

        batch_centers: List[int] = []
        batch_contexts: List[int] = []

        for idx, center in enumerate(tokens):
            contexts = _context_window(tokens, idx, config.context_size)
            if not contexts:
                continue
            context = random.choice(contexts)
            batch_centers.append(center)
            batch_contexts.append(context)

            if len(batch_centers) == config.batch_size:
                loss = _train_batch_torch(
                    model,
                    optimizer,
                    batch_centers,
                    batch_contexts,
                    unigram_table,
                    config.negatives,
                    device,
                )
                running_loss += loss
                steps += 1
                pairs_processed += len(batch_centers)
                batch_centers.clear()
                batch_contexts.clear()

        if batch_centers:
            loss = _train_batch_torch(
                model,
                optimizer,
                batch_centers,
                batch_contexts,
                unigram_table,
                config.negatives,
                device,
            )
            running_loss += loss
            steps += 1
            pairs_processed += len(batch_centers)

        elapsed = time.perf_counter() - epoch_start
        avg_loss = running_loss / max(1, steps)
        stats.append(
            SkipGramStats(
                epoch=epoch,
                loss=avg_loss,
                steps=steps,
                pairs_processed=pairs_processed,
                elapsed_sec=elapsed,
            )
        )
        throughput = pairs_processed / elapsed if elapsed > 0 else float("inf")
        print(
            f"Epoch {epoch}/{config.epochs}: loss={avg_loss:.4f}, "
            f"pairs={pairs_processed:,}, time={elapsed:.1f}s, throughput={throughput:,.0f} pairs/s"
        )

    return model, stats


def _train_batch_torch(
    model: SkipGramModel,
    optimizer: torch.optim.Optimizer,
    centers: List[int],
    contexts: List[int],
    unigram_table: torch.Tensor,
    negatives: int,
    device: Any,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    center_ids = torch.tensor(centers, dtype=torch.long, device=device)
    context_ids = torch.tensor(contexts, dtype=torch.long, device=device)
    neg_ids = torch.multinomial(unigram_table, len(centers) * negatives, replacement=True)
    neg_ids = neg_ids.to(device).view(len(centers), negatives)
    loss = model(center_ids, context_ids, neg_ids)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def _train_skipgram_numpy(
    tokens: Sequence[int],
    vocab_size: int,
    config: SkipGramConfig,
) -> Tuple[SkipGramNumpyModel, List[SkipGramStats]]:
    rng = np.random.default_rng(config.seed)
    random.seed(config.seed)

    input_embeds = rng.uniform(
        low=-0.5 / config.embedding_dim,
        high=0.5 / config.embedding_dim,
        size=(vocab_size, config.embedding_dim),
    ).astype(np.float32)
    output_embeds = np.zeros((vocab_size, config.embedding_dim), dtype=np.float32)
    model = SkipGramNumpyModel(input_embeddings=input_embeds, output_embeddings=output_embeds)

    unigram_probs = _build_unigram_probs(tokens, vocab_size)
    stats: List[SkipGramStats] = []

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        running_loss = 0.0
        steps = 0
        pairs_processed = 0

        for idx, center in enumerate(tokens):
            contexts = _context_window(tokens, idx, config.context_size)
            if not contexts:
                continue
            context = random.choice(contexts)
            negatives = rng.choice(vocab_size, size=config.negatives, p=unigram_probs, replace=True)
            loss = _train_pair_numpy(model, center, context, negatives, config.learning_rate)
            running_loss += loss
            steps += 1
            pairs_processed += 1

        elapsed = time.perf_counter() - epoch_start
        avg_loss = running_loss / max(1, steps)
        stats.append(
            SkipGramStats(
                epoch=epoch,
                loss=avg_loss,
                steps=steps,
                pairs_processed=pairs_processed,
                elapsed_sec=elapsed,
            )
        )
        throughput = pairs_processed / elapsed if elapsed > 0 else float("inf")
        print(
            f"[CPU] Epoch {epoch}/{config.epochs}: loss={avg_loss:.4f}, "
            f"pairs={pairs_processed:,}, time={elapsed:.1f}s, throughput={throughput:,.0f} pairs/s"
        )

    return model, stats


def _sigmoid(x: float | np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x: float | np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _train_pair_numpy(
    model: SkipGramNumpyModel,
    center_idx: int,
    context_idx: int,
    negative_indices: np.ndarray,
    learning_rate: float,
) -> float:
    center_vec = model.input_embeddings[center_idx]
    context_vec = model.output_embeddings[context_idx]
    center_base = center_vec.copy()
    context_base = context_vec.copy()

    pos_score = float(np.dot(center_base, context_base))
    pos_sigmoid = float(_sigmoid(pos_score))
    pos_grad_center = (pos_sigmoid - 1.0) * context_base
    pos_grad_context = (pos_sigmoid - 1.0) * center_base
    loss = float(_softplus(-pos_score))

    center_grad = pos_grad_center
    model.output_embeddings[context_idx] = context_vec - learning_rate * pos_grad_context

    for neg_idx in negative_indices:
        neg_vec = model.output_embeddings[neg_idx]
        neg_base = neg_vec.copy()
        neg_score = float(np.dot(center_base, neg_base))
        neg_sigmoid = float(_sigmoid(neg_score))
        center_grad += neg_sigmoid * neg_base
        grad_neg = neg_sigmoid * center_base
        model.output_embeddings[neg_idx] = neg_vec - learning_rate * grad_neg
        loss += float(_softplus(neg_score))

    model.input_embeddings[center_idx] = center_vec - learning_rate * center_grad
    return loss


def extract_embeddings(model: object):
    if TORCH_AVAILABLE and isinstance(model, SkipGramModel):
        return model.input_embeddings.weight.data.detach().cpu()
    if isinstance(model, SkipGramNumpyModel):
        return np.array(model.input_embeddings, copy=True)
    raise TypeError("Unsupported skip-gram model type for extraction.")

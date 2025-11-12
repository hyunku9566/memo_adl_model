from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SequenceModelConfig:
    sensor_embed_dim: int = 64
    state_embed_dim: int = 16
    value_type_embed_dim: int = 8
    numeric_feature_dim: int = 16
    time_feature_dim: int = 16
    model_dim: int = 128
    ff_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.2


class SensorSequenceModel(nn.Module):
    """Transformer encoder over sliding-window sensor sequences."""

    def __init__(
        self,
        num_sensors: int,
        num_states: int,
        num_value_types: int,
        num_classes: int,
        window_size: int,
        config: SequenceModelConfig,
        sensor_embedding_init: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.config = config

        self.sensor_emb = nn.Embedding(num_sensors, config.sensor_embed_dim)
        if sensor_embedding_init is not None:
            if sensor_embedding_init.shape[1] != config.sensor_embed_dim:
                raise ValueError(
                    f"Sensor embedding dim mismatch: checkpoint={sensor_embedding_init.shape[1]}, "
                    f"expected={config.sensor_embed_dim}"
                )
            with torch.no_grad():
                self.sensor_emb.weight.copy_(sensor_embedding_init)

        self.state_emb = nn.Embedding(num_states, config.state_embed_dim)
        self.value_type_emb = nn.Embedding(num_value_types, config.value_type_embed_dim)

        self.numeric_proj = nn.Linear(2, config.numeric_feature_dim)
        self.time_proj = nn.Linear(4, config.time_feature_dim)

        concat_dim = (
            config.sensor_embed_dim
            + config.state_embed_dim
            + config.value_type_embed_dim
            + config.numeric_feature_dim
            + config.time_feature_dim
        )
        self.pre_proj = nn.Linear(concat_dim, config.model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.positional = nn.Parameter(torch.zeros(1, window_size, config.model_dim))

        self.classifier = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, num_classes),
        )

    def forward(
        self,
        sensor_ids: torch.Tensor,
        state_ids: torch.Tensor,
        value_type_ids: torch.Tensor,
        numeric_values: torch.Tensor,
        numeric_mask: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        sensor_vec = self.sensor_emb(sensor_ids)
        state_vec = self.state_emb(state_ids)
        value_type_vec = self.value_type_emb(value_type_ids)

        numeric_input = torch.stack([numeric_values, numeric_mask], dim=-1)
        numeric_vec = F.gelu(self.numeric_proj(numeric_input))

        time_vec = F.gelu(self.time_proj(time_features))

        concat = torch.cat([sensor_vec, state_vec, value_type_vec, numeric_vec, time_vec], dim=-1)
        x = self.pre_proj(concat)
        x = x + self.positional
        x = self.encoder(x)

        last_state = x[:, -1, :]
        logits = self.classifier(last_state)
        return logits

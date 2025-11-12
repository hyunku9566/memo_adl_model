"""
EMA-Attention 기반 Adaptive Decay Memory 모델

핵심 개념:
- 시간 간격이 멀수록 자동으로 잊음 (지수감쇠)
- 이동·정지 상태에 따라 잊는 속도를 다르게 학습
- Learnable decay rate λ(t,i) = f_θ(movement_i, speed_i, X_i)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import math


# ============================================================================
# 유틸: 시퀀스 마스크
# ============================================================================
def lengths_to_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """
    lengths: (B,) 실제 시퀀스 길이
    max_len: 최대 길이 (None이면 lengths.max())
    반환: (B, T) Boolean mask (True=valid, False=padding)
    """
    B = lengths.size(0)
    T = int(max_len or lengths.max().item())
    arange = torch.arange(T, device=lengths.device).unsqueeze(0).expand(B, T)
    return arange < lengths.unsqueeze(1)


# ============================================================================
# TCN 블록 (Temporal Convolutional Network)
# ============================================================================
class TCNBlock(nn.Module):
    """
    Dilated convolution 기반 시간 인코더
    Residual connection 포함
    """
    def __init__(self, in_ch: int, out_ch: int, ks: int = 3, dil: int = 1, drop: float = 0.1):
        super().__init__()
        pad = (ks - 1) * dil
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, ks, padding=pad, dilation=dil),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv1d(out_ch, out_ch, ks, padding=pad, dilation=dil),
            nn.ReLU(),
            nn.Dropout(drop),
        )
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        """
        y = self.net(x)
        # Causal trim to align length
        if y.size(-1) > x.size(-1):
            y = y[..., :x.size(-1)]
        return y + self.res(x)


# ============================================================================
# Adaptive Decay Attention (핵심 모듈)
# ============================================================================
class AdaptiveDecayAttention(nn.Module):
    """
    시간 감쇠를 고려한 적응형 어텐션 메커니즘
    
    score_{t,i} = (q_t·k_i/√d) - λ_{t,i}·Δt_{t,i}
    λ_{t,i} = Softplus(MLP([x_i, speed_i, move_i]))
    
    Args:
        hidden: 숨겨진 차원 (model_dim)
        cond_dim: 조건 특징 차원 (speed, movement 등)
        heads: 멀티헤드 수
        dropout: 드롭아웃 확률
        lambda_floor: λ의 최소값 (수치 안정성)
    """
    def __init__(
        self,
        hidden: int,
        cond_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
        lambda_floor: float = 0.0,
    ):
        super().__init__()
        self.h = heads
        self.dk = hidden // heads
        assert hidden % heads == 0, f"hidden ({hidden}) must be divisible by heads ({heads})"

        # Q, K, V 프로젝션
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

        # λ(decay rate) 계산용 MLP
        # 입력: 키 위치의 조건 특징 (speed, movement 등)
        self.lambda_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, heads),  # head별 λ 값
        )
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(dropout)
        self.lambda_floor = lambda_floor

    def forward(
        self,
        x: torch.Tensor,
        cond_feat: torch.Tensor,
        delta_t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, H) - 시퀀스 특징
            cond_feat: (B, T, C) - 조건 특징 (speed, movement 등)
            delta_t: (B, T, T) - 쿼리-키 간 시간 차이 (≥0)
            mask: (B, T) - 유효 타임스텝 마스크 (True=valid)
        
        Returns:
            seq_out: (B, T, H) - 시퀀스별 어텐션 출력
            pooled: (B, H) - 풀링된 표현
            attn: (B, h, T, T) - 어텐션 가중치
        """
        B, T, H = x.shape

        # Q, K, V 프로젝션
        q = self.q_proj(x).view(B, T, self.h, self.dk).transpose(1, 2)  # (B, h, T, dk)
        k = self.k_proj(x).view(B, T, self.h, self.dk).transpose(1, 2)  # (B, h, T, dk)
        v = self.v_proj(x).view(B, T, self.h, self.dk).transpose(1, 2)  # (B, h, T, dk)

        # 기본 attention score: s_{t,i} = q_t·k_i / √d
        scores = torch.einsum('bhtd,bhsd->bhts', q, k) / (self.dk ** 0.5)  # (B, h, T, T)

        # Adaptive decay rate λ 계산
        # λ_{t,i}는 키 위치 i의 조건 특징을 기반으로 계산
        lam_raw = self.lambda_mlp(cond_feat)  # (B, T, h)
        lam = self.softplus(lam_raw) + self.lambda_floor  # (B, T, h) > 0
        lam = lam.permute(0, 2, 1).unsqueeze(2)  # (B, h, 1, T)

        # 시간 감쇠 적용: score = score - λ * Δt
        # delta_t: (B, T, T) -> (B, 1, T, T)
        dt = delta_t.unsqueeze(1)  # (B, 1, T, T)
        scores = scores - lam * dt  # Head별 적응형 감쇠

        # 패딩 마스크 처리
        if mask is not None:  # (B, T)
            m = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            scores = scores.masked_fill(~m, float('-inf'))

        # Softmax
        attn = F.softmax(scores, dim=-1)  # (B, h, T, T)
        attn = self.dropout(attn)

        # 가중합: c_t = Σ_i α_{t,i} v_i
        ctx = torch.einsum('bhts,bhsd->bhtd', attn, v)  # (B, h, T, dk)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, H)  # (B, T, H)
        seq_out = self.out_proj(ctx)  # (B, T, H)

        # 시간 풀링: 마스크를 고려한 가중 평균
        if mask is not None:
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)  # (B, 1, 1)
            pooled = (seq_out * mask.unsqueeze(-1)).sum(dim=1) / denom.squeeze(-1)  # (B, H)
        else:
            pooled = seq_out.mean(dim=1)  # (B, H)

        return seq_out, pooled, attn


# ============================================================================
# EMA (Exponential Moving Average) 계산
# ============================================================================
class EMACalculator(nn.Module):
    """
    EMA 기반 특징 평활화
    """
    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) - 입력 시퀀스
            mask: (B, T) - 유효 타임스텝 마스크
        
        Returns:
            ema: (B, T, D) - EMA 평활화된 시퀀스
        """
        B, T, D = x.shape
        ema = torch.zeros_like(x)

        # 첫 번째 스텝
        ema[:, 0] = x[:, 0]

        # 나머지 스텝들
        for t in range(1, T):
            if mask is not None:
                # 유효 스텝만 업데이트
                valid = mask[:, t].float().unsqueeze(-1)  # (B, 1)
                ema[:, t] = (1 - self.alpha) * ema[:, t - 1] + self.alpha * x[:, t]
                ema[:, t] = ema[:, t] * valid + ema[:, t - 1] * (1 - valid)
            else:
                ema[:, t] = (1 - self.alpha) * ema[:, t - 1] + self.alpha * x[:, t]

        return ema


# ============================================================================
# 전체 모델
# ============================================================================
class EMAAdaptiveDecayModel(nn.Module):
    """
    EMA-Attention 기반 Adaptive Decay Memory 모델
    
    구조:
    1. Feature embedding & EMA smoothing
    2. Temporal encoding (TCN)
    3. Adaptive Decay Attention (핵심)
    4. Classification head
    """
    def __init__(
        self,
        num_sensors: int,
        num_states: int,
        num_value_types: int,
        num_classes: int = 5,
        sensor_embed_dim: int = 64,
        state_embed_dim: int = 16,
        value_type_embed_dim: int = 8,
        hidden: int = 128,
        heads: int = 4,
        num_tcn_layers: int = 3,
        cond_dim: int = 8,
        dropout: float = 0.1,
        ema_alpha: float = 0.2,
    ):
        super().__init__()
        self.hidden = hidden
        self.num_classes = num_classes

        # Embedding layers (one-hot 대신 사용)
        self.sensor_emb = nn.Embedding(num_sensors, sensor_embed_dim)
        self.state_emb = nn.Embedding(num_states, state_embed_dim)
        self.value_type_emb = nn.Embedding(num_value_types, value_type_embed_dim)
        
        # Numeric projection
        self.numeric_proj = nn.Linear(2, 16)  # numeric + mask
        self.time_proj = nn.Linear(4, 16)  # time features
        
        # Total feature dim after concat
        concat_dim = sensor_embed_dim + state_embed_dim + value_type_embed_dim + 16 + 16
        
        # Feature projection
        self.feat_proj = nn.Linear(concat_dim, hidden)

        # EMA calculator
        self.ema_calc = EMACalculator(alpha=ema_alpha)

        # TCN 백본 (시간적 인코더)
        tcn_layers = []
        for i in range(num_tcn_layers):
            dil = 2 ** i
            tcn_layers.append(TCNBlock(hidden, hidden, ks=3, dil=dil, drop=dropout))
        self.tcn = nn.Sequential(*tcn_layers)

        # Adaptive Decay Attention (핵심)
        self.decay_attn = AdaptiveDecayAttention(
            hidden=hidden,
            cond_dim=cond_dim,
            heads=heads,
            dropout=dropout,
            lambda_floor=0.0,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(
        self,
        sensor_ids: torch.Tensor,
        state_ids: torch.Tensor,
        value_type_ids: torch.Tensor,
        numeric: torch.Tensor,
        numeric_mask: torch.Tensor,
        time_features: torch.Tensor,
        cond_feat: torch.Tensor,
        delta_t: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            sensor_ids: (B, T) - 센서 ID
            state_ids: (B, T) - 상태 ID
            value_type_ids: (B, T) - 값 타입 ID
            numeric: (B, T) - 수치값
            numeric_mask: (B, T) - 수치값 마스크
            time_features: (B, T, 4) - 시간 특징
            cond_feat: (B, T, C) - 조건 특징 (speed, movement 등)
            delta_t: (B, T, T) - 시간 차이 행렬
            lengths: (B,) - 실제 시퀀스 길이
        
        Returns:
            logits: (B, num_classes) - 분류 로짓
            extras: 디버깅/시각화용 중간값들
        """
        # 마스크 생성
        mask = lengths_to_mask(lengths, sensor_ids.size(1)) if lengths is not None else None

        # 1. Embedding
        sensor_vec = self.sensor_emb(sensor_ids)  # (B, T, sensor_embed_dim)
        state_vec = self.state_emb(state_ids)  # (B, T, state_embed_dim)
        value_type_vec = self.value_type_emb(value_type_ids)  # (B, T, value_type_embed_dim)
        
        # 2. Numeric & Time projection
        numeric_input = torch.stack([numeric, numeric_mask], dim=-1)  # (B, T, 2)
        numeric_vec = F.gelu(self.numeric_proj(numeric_input))  # (B, T, 16)
        time_vec = F.gelu(self.time_proj(time_features))  # (B, T, 16)
        
        # 3. Concatenate all features
        X = torch.cat([sensor_vec, state_vec, value_type_vec, numeric_vec, time_vec], dim=-1)
        
        # 4. Feature projection
        h = self.feat_proj(X)  # (B, T, H)

        # 5. EMA smoothing (선택적)
        h_ema = self.ema_calc(h, mask)  # (B, T, H)

        # 6. TCN 인코딩
        h = self.tcn(h.transpose(1, 2)).transpose(1, 2)  # (B, T, H)

        # 7. Adaptive Decay Attention
        seq_out, pooled, attn = self.decay_attn(h, cond_feat, delta_t, mask)

        # 5. Classification
        logits = self.classifier(pooled)  # (B, num_classes)

        # 부가 정보 (디버깅, 시각화용)
        extras = {
            'attn': attn,  # (B, h, T, T)
            'seq_out': seq_out,  # (B, T, H)
            'pooled': pooled,  # (B, H)
        }

        return logits, extras


# ============================================================================
# 모델 설정 데이터클래스
# ============================================================================
@dataclass
class AdaptiveDecayConfig:
    """모델 설정값"""
    feat_in: int = 114  # 입력 특징 차원
    num_classes: int = 5
    hidden: int = 128
    heads: int = 4
    num_tcn_layers: int = 3
    cond_dim: int = 8  # 조건 특징 차원 (speed, movement 등)
    dropout: float = 0.1
    ema_alpha: float = 0.2

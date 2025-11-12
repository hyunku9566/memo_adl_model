"""
EMA-Attention Based Adaptive Decay Memory Model
================================================

핵심: 시간 간격(Δt)과 이동 상태에 따라 메모리 감쇠율(λ)을 학습적으로 조정하는 어텐션
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 유틸: 시퀀스 마스크 ----------
def lengths_to_mask(lengths, max_len=None):
    """
    시퀀스 길이를 boolean mask로 변환
    
    Args:
        lengths: (B,) 각 샘플의 유효 길이
        max_len: 최대 길이 (None이면 lengths.max() 사용)
    
    Returns:
        (B, T) boolean mask, True=유효, False=패딩
    """
    B = lengths.size(0)
    T = int(max_len or lengths.max().item())
    arange = torch.arange(T, device=lengths.device).unsqueeze(0).expand(B, T)
    return arange < lengths.unsqueeze(1)  # (B, T)


# ---------- TCN (Temporal Convolutional Network) 블록 ----------
class TCNBlock(nn.Module):
    """
    Dilated causal convolution block
    - 시간적 패턴을 병렬로 추출
    - Dilation으로 receptive field 확장
    """
    def __init__(self, in_ch, out_ch, ks=3, dil=1, drop=0.1):
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
        
    def forward(self, x):  # x: (B, C, T)
        y = self.net(x)
        # Causal trim to align length
        if y.size(-1) > x.size(-1):
            y = y[..., :x.size(-1)]
        return y + self.res(x)


# ---------- Adaptive Decay Attention ----------
class AdaptiveDecayAttention(nn.Module):
    """
    시간 감쇠를 포함한 어텐션
    
    핵심 아이디어:
    1. 기본 어텐션 점수: score = q·k / √d
    2. 시간 감쇠 적용: score -= λ * Δt
    3. λ는 키의 조건 특징(속도, 이동 여부 등)으로부터 학습
    
    효과:
    - 오래된 기억일수록 자동 감쇠
    - 이동 중/정지 시 다른 감쇠율 적용
    """
    def __init__(self, hidden, cond_dim, heads=4, dropout=0.1, lambda_floor=0.0):
        super().__init__()
        self.h = heads
        self.dk = hidden // heads
        assert hidden % heads == 0, f"hidden ({hidden}) must be divisible by heads ({heads})"

        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

        # λ를 만드는 작은 MLP (키의 조건 특징 기반)
        self.lambda_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, heads),   # head별 λ
        )
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(dropout)
        self.lambda_floor = lambda_floor  # 최소 감쇠량

    def forward(self, x, cond_feat, delta_t, mask=None):
        """
        Args:
            x: (B, T, H) 입력 hidden states
            cond_feat: (B, T, C) 조건 특징 (speed, movement_flag 등)
            delta_t: (B, T, T) 쿼리-키 간 시간 차이 (>=0)
            mask: (B, T) 유효 타임스텝 마스크
            
        Returns:
            out: (B, T, H) 어텐션 출력
            pooled: (B, H) 시간 풀링된 출력
            attn: (B, h, T, T) 어텐션 가중치
        """
        B, T, H = x.shape
        
        # Q, K, V 투영
        q = self.q_proj(x).view(B, T, self.h, self.dk).transpose(1, 2)  # (B, h, T, dk)
        k = self.k_proj(x).view(B, T, self.h, self.dk).transpose(1, 2)  # (B, h, T, dk)
        v = self.v_proj(x).view(B, T, self.h, self.dk).transpose(1, 2)  # (B, h, T, dk)

        # 기본 어텐션 점수: q·k / √d
        scores = torch.einsum('bhtd,bhsd->bhts', q, k) / (self.dk ** 0.5)  # (B, h, T, T)

        # λ 계산: 키 타임스텝의 조건 특징으로부터
        lam_raw = self.lambda_mlp(cond_feat)  # (B, T, h)
        lam = self.softplus(lam_raw) + self.lambda_floor  # (B, T, h)
        lam = lam.permute(0, 2, 1).unsqueeze(2)  # (B, h, 1, T)

        # 시간 감쇠 적용: score -= λ * Δt
        dt = delta_t.unsqueeze(1)  # (B, 1, T, T)
        scores = scores - lam * dt  # (B, h, T, T)

        # 마스크 처리 (패딩 영역은 -inf로)
        if mask is not None:  # (B, T)
            m = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            scores = scores.masked_fill(~m, float('-inf'))

        # Softmax 어텐션
        attn = F.softmax(scores, dim=-1)  # (B, h, T, T)
        attn = self.dropout(attn)
        
        # 컨텍스트 계산
        ctx = torch.einsum('bhts,bhsd->bhtd', attn, v)  # (B, h, T, dk)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, H)  # (B, T, H)
        out = self.out_proj(ctx)  # (B, T, H)

        # 시간 풀링 (마스크 고려한 평균)
        if mask is not None:
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)  # (B, 1, 1)
            pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / denom.squeeze(-1)  # (B, H)
        else:
            pooled = out.mean(dim=1)  # (B, H)

        return out, pooled, attn


# ---------- 전체 모델 ----------
class EMAAdaptiveDecayModel(nn.Module):
    """
    EMA-Attention 기반 Adaptive Decay Memory 모델
    
    입력:
    - X: (B, T, F_in) 결합 특징 [X_frame | X_ema | X_vel | X_emb]
    - cond_feat: (B, T, C) 조건 특징 (λ 학습용)
    - delta_t: (B, T, T) 시간 차이
    - lengths: (B,) 유효 길이
    
    구조:
    1. 입력 투영: F_in → H
    2. TCN 백본: 시간적 패턴 추출 (dilation=1,2,4)
    3. Adaptive Decay Attention: 시간 감쇠 메모리
    4. Classifier: H → num_classes
    """
    def __init__(
        self,
        feat_in,
        hidden=128,
        heads=4,
        num_classes=5,
        cond_dim=8,
        tcn_layers=3,
        dropout=0.1
    ):
        super().__init__()
        
        # 입력 투영
        self.proj = nn.Linear(feat_in, hidden)
        self.proj_norm = nn.LayerNorm(hidden)

        # TCN 백본 (경량화된 시간 인코더)
        tcn_blocks = []
        for i in range(tcn_layers):
            dil = 2 ** i  # 1, 2, 4
            tcn_blocks.append(
                TCNBlock(hidden, hidden, ks=3, dil=dil, drop=dropout)
            )
        self.tcn = nn.Sequential(*tcn_blocks)

        # Adaptive Decay Attention (핵심 모듈)
        self.decay_attn = AdaptiveDecayAttention(
            hidden=hidden,
            cond_dim=cond_dim,
            heads=heads,
            dropout=dropout,
            lambda_floor=0.0
        )

        # Classifier
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout * 2),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, X, cond_feat, delta_t, lengths=None):
        """
        Args:
            X: (B, T, F_in) 결합 특징
            cond_feat: (B, T, C) 조건 특징
            delta_t: (B, T, T) 시간 차이
            lengths: (B,) 유효 길이
            
        Returns:
            logits: (B, num_classes) 분류 로짓
            extras: dict with 'attn', 'seq_out', 'lambda'
        """
        mask = lengths_to_mask(lengths, X.size(1)) if lengths is not None else None

        # 입력 투영
        h = self.proj(X)  # (B, T, H)
        h = self.proj_norm(h)

        # TCN 시간 인코딩
        h = self.tcn(h.transpose(1, 2)).transpose(1, 2)  # (B, T, H)

        # Adaptive Decay Attention
        seq_out, pooled, attn = self.decay_attn(h, cond_feat, delta_t, mask)  # (B, T, H), (B, H), (B, h, T, T)

        # 분류
        logits = self.head(pooled)  # (B, num_classes)

        extras = {
            'attn': attn,
            'seq_out': seq_out,
        }

        return logits, extras


# ---------- 모델 생성 헬퍼 ----------
def create_ema_adaptive_decay_model(
    num_sensors: int,
    emb_dim: int = 32,
    hidden: int = 128,
    heads: int = 4,
    num_classes: int = 5,
    cond_dim: int = 8,
    tcn_layers: int = 3,
    dropout: float = 0.1
) -> EMAAdaptiveDecayModel:
    """
    Rich features 기반 모델 생성
    
    Args:
        num_sensors: 센서 개수
        emb_dim: 센서 임베딩 차원 (skip-gram)
        hidden: Hidden dimension
        heads: Attention heads
        num_classes: 활동 클래스 개수
        cond_dim: 조건 특징 차원
        tcn_layers: TCN 레이어 수
        dropout: Dropout rate
    
    Returns:
        EMAAdaptiveDecayModel
    """
    # 입력 특징 차원 계산
    # X_frame: (T, N_sensor)
    # X_ema: (T, N_sensor)
    # X_vel: (T, 6)
    # X_emb: (T, emb_dim)
    feat_in = num_sensors * 2 + 6 + emb_dim
    
    model = EMAAdaptiveDecayModel(
        feat_in=feat_in,
        hidden=hidden,
        heads=heads,
        num_classes=num_classes,
        cond_dim=cond_dim,
        tcn_layers=tcn_layers,
        dropout=dropout
    )
    
    return model

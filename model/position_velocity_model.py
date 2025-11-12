#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Position-Velocity-MMU/CMU Model for Smart Home Activity Recognition
====================================================================

핵심 구조:
1. PositionHead: 학습 가능한 센서 2D 위치
2. VelocityHead: 위치 차분으로 속도/방향/이동 특징 추출
3. MMU (Movement Memory Unit): 이동 패턴 기억
4. CMU (Context Memory Unit): 맥락/영역 기억
5. GateAndTrigger: 이동/맥락 동적 융합
6. TemporalEncoder: TCN → BiGRU → Attention
7. Classifier: 최종 활동 분류

입력:
- X_base: [B, T, F_base] (rich features: frame+ema+vel+emb)
- sensor_ids: [B, T] (각 시점의 대표 센서 ID)
- timestamps: [B, T] (선택적, 초 단위 float)

출력:
- logits: [B, n_classes]
- aux: dict (pos, vel, move_flag, gate, trigger, attn 등)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


# ==================== Utility Modules ====================

class TCNBlock(nn.Module):
    """Temporal Convolutional Network with causal padding"""
    
    def __init__(self, in_ch, out_ch, k=3, dil=1, pdrop=0.1):
        super().__init__()
        pad = (k - 1) * dil
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=dil),
            nn.ReLU(),
            nn.Dropout(pdrop),
            nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=dil),
            nn.ReLU(),
            nn.Dropout(pdrop),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [B, C, T]
        Returns:
            [B, C, T]
        """
        y = self.net(x)
        # Causal trim to match input length
        trim = y.size(-1) - x.size(-1)
        if trim > 0:
            y = y[..., :-trim]
        return y + self.skip(x)


class AdditiveAttention(nn.Module):
    """Additive (Bahdanau-style) attention mechanism"""
    
    def __init__(self, hid):
        super().__init__()
        self.scorer = nn.Linear(hid, 1, bias=False)

    def forward(self, H):
        """
        Args:
            H: [B, T, H]
        Returns:
            ctx: [B, H] - context vector
            w: [B, T] - attention weights
        """
        score = self.scorer(H)               # [B, T, 1]
        w = F.softmax(score, dim=1)          # [B, T, 1]
        ctx = (H * w).sum(dim=1)             # [B, H]
        return ctx, w.squeeze(-1)


# ==================== Core Components ====================

class PositionHead(nn.Module):
    """
    학습 가능한 센서 2D 위치
    
    선행 연구: 대부분 고정 위치 사용 → 우리는 end-to-end 학습 🆕
    """
    
    def __init__(self, vocab_size: int, init_scale: float = 0.1):
        """
        Args:
            vocab_size: 센서 개수
            init_scale: 초기 위치 분산
        """
        super().__init__()
        # 학습 가능한 2D 좌표 (N_sensor, 2) - uniform init for stability
        positions = torch.empty(vocab_size, 2)
        nn.init.uniform_(positions, -init_scale, init_scale)
        self.positions = nn.Parameter(positions)

    def forward(self, sensor_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sensor_ids: [B, T] - 각 시점의 센서 ID
        Returns:
            [B, T, 2] - 2D 위치 시퀀스
        """
        return self.positions[sensor_ids]  # [B, T, 2]


class VelocityHead(nn.Module):
    """
    속도/방향 특징 추출 + EMA 평활화 + 방향 임베딩
    
    출력:
    - ΔP (위치 차분)
    - Δt (시간 차분)
    - speed (EMA 평활)
    - direction (EMA 평활)
    - move_flag (이동/정지 이진 플래그)
    - direction embedding (8방위 임베딩)
    """
    
    def __init__(
        self,
        d_model: int = 32,
        dir_emb_dim: int = 8,
        ema_alpha: float = 0.3,
        move_thresh: float = 0.1,
        num_dir_bins: int = 8
    ):
        """
        Args:
            d_model: 출력 임베딩 차원
            dir_emb_dim: 방향 임베딩 차원
            ema_alpha: EMA 감쇠율
            move_thresh: 이동 판정 임계값
            num_dir_bins: 방향 구간 개수 (8 = 8방위)
        """
        super().__init__()
        self.ema_alpha = ema_alpha
        self.move_thresh = move_thresh
        self.num_dir_bins = num_dir_bins

        # 6D 입력 인코더: [Δx, Δy, Δt, speed, direction, move_flag]
        self.encoder = nn.Sequential(
            nn.Linear(6, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 방향 임베딩 (0=stationary, 1-8=8방위)
        self.dir_emb = nn.Embedding(num_dir_bins + 1, dir_emb_dim)
        
        # 최종 projection
        self.out_proj = nn.Linear(d_model + dir_emb_dim, d_model)
        
        # 보조 과제: 이동 분류 (moving vs stationary)
        self.mov_cls = nn.Linear(d_model, 2)

    @staticmethod
    def _ema(x, alpha):
        """
        Exponential Moving Average (gradient-friendly cumsum version)
        
        Args:
            x: [B, T, C]
            alpha: 감쇠율
        Returns:
            [B, T, C] - smoothed
        """
        # Use convolution for EMA (fully differentiable)
        B, T, C = x.shape
        
        # Simple exponential weighting
        # y_t = alpha * x_t + (1-alpha) * y_{t-1}
        # This is equivalent to convolution with exponential kernel
        y = torch.zeros_like(x)
        y[:, 0] = x[:, 0]
        
        for t in range(1, T):
            y[:, t] = alpha * x[:, t] + (1 - alpha) * y[:, t - 1]
        
        return y

    def forward(
        self,
        sensor_pos: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            sensor_pos: [B, T, 2] - 센서 2D 위치 시퀀스
            timestamps: [B, T] - 타임스탬프 (옵션, 없으면 Δt=1)
        
        Returns:
            vel: [B, T, d_model] - 속도 임베딩
            move_flag: [B, T] - 이동 여부 (0=정지, 1=이동)
            aux: dict - 중간 계산 결과들
        """
        B, T, _ = sensor_pos.shape
        device = sensor_pos.device

        # 1) 위치 차분 ΔP
        dP = torch.zeros_like(sensor_pos)
        dP[:, 1:] = sensor_pos[:, 1:] - sensor_pos[:, :-1]  # [B, T, 2]

        # 2) 시간 차분 Δt
        if timestamps is not None:
            dt = torch.ones(B, T, 1, device=device)  # initialize with 1s
            dt_raw = timestamps[:, 1:] - timestamps[:, :-1]
            # CRITICAL: Replace dt=0 with 1.0 to prevent division by zero
            dt_raw = torch.where(dt_raw > 0.01, dt_raw, torch.ones_like(dt_raw))
            dt[:, 1:, 0] = torch.clamp(dt_raw, 0.1, 1000.0)  # min 0.1초, max 1000초
        else:
            dt = torch.ones(B, T, 1, device=device)

        # 3) 속도 (거리 / 시간) - clamp to prevent explosion
        distance = torch.norm(dP, dim=-1, keepdim=True) + 1e-8  # [B, T, 1]
        # Normalize dt to reasonable scale before division
        dt_safe = torch.clamp(dt, 0.1, 1000.0) / 10.0  # [0.01, 100]
        speed = distance / (dt_safe + 1e-6)  # [B, T, 1]
        speed = torch.clamp(speed, 0, 10.0)  # clamp speed to reasonable range

        # 4) 방향 (atan2)
        # CRITICAL: Add epsilon to prevent gradient explosion when dP is near zero
        dP_safe = dP + 1e-8
        direction = torch.atan2(dP_safe[..., 1], dP_safe[..., 0]).unsqueeze(-1)  # [B, T, 1], [-π, π]
        direction_norm = torch.clamp(direction / math.pi, -1.0, 1.0)  # normalize to [-1, 1]

        # 5) EMA 평활화
        speed_s = self._ema(speed, self.ema_alpha)
        dir_s = self._ema(direction_norm, self.ema_alpha)

        # 6) 이동 플래그
        move_flag = (speed_s.squeeze(-1) > self.move_thresh).float()  # [B, T]

        # 7) 6D 특징 구성: [Δx, Δy, Δt, speed, direction, move_flag]
        feats = torch.cat([
            dP,                        # [B, T, 2]
            dt,                        # [B, T, 1]
            speed_s,                   # [B, T, 1]
            dir_s,                     # [B, T, 1]
            move_flag.unsqueeze(-1)    # [B, T, 1]
        ], dim=-1)  # [B, T, 6]

        # 8) 기본 임베딩
        base = self.encoder(feats)  # [B, T, d_model]

        # 9) 방향 임베딩 (8방위)
        ang = (dir_s.squeeze(-1) + 1) * math.pi  # [0, 2π]
        bins = (ang / (2 * math.pi) * self.num_dir_bins).long().clamp(
            0, self.num_dir_bins - 1
        )
        bins = bins * move_flag.long()  # stationary면 bin=0
        dir_emb = self.dir_emb(bins)    # [B, T, dir_emb_dim]

        # 10) 최종 속도 임베딩
        vel = self.out_proj(torch.cat([base, dir_emb], dim=-1))  # [B, T, d_model]

        # 11) 보조 과제: 이동 분류 logits
        mov_logits = self.mov_cls(vel)  # [B, T, 2]

        # 12) Auxiliary outputs
        aux = dict(
            dP=dP,
            dt=dt,
            speed=speed_s,
            direction=dir_s,
            move_flag=move_flag,
            mov_logits=mov_logits
        )
        
        return vel, move_flag, aux


class MMU(nn.Module):
    """
    Movement Memory Unit (이동 메모리)
    
    이동 패턴을 누적 학습:
    - 속도 벡터 시퀀스
    - 누적 이동/정지 카운터
    - GRU로 temporal dependencies 모델링
    
    노벨티: Movement-specific memory (선행 연구 거의 없음) 🆕
    """
    
    def __init__(self, in_dim: int, hid: int = 128):
        """
        Args:
            in_dim: 속도 임베딩 차원
            hid: GRU hidden 차원
        """
        super().__init__()
        # 입력: vel + move_cnt + stay_cnt
        self.gru = nn.GRU(
            input_size=in_dim + 2,
            hidden_size=hid,
            batch_first=True
        )
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, vel: torch.Tensor, move_flag: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vel: [B, T, D] - 속도 임베딩
            move_flag: [B, T] - 이동 플래그
        
        Returns:
            [B, T, hid] - 이동 메모리 hidden states
        """
        B, T, D = vel.shape
        
        # 누적 카운터
        move_cnt = torch.cumsum(move_flag, dim=1).unsqueeze(-1)       # [B, T, 1]
        stay_cnt = torch.cumsum(1 - move_flag, dim=1).unsqueeze(-1)   # [B, T, 1]
        
        # 입력 구성
        x = torch.cat([vel, move_cnt, stay_cnt], dim=-1)  # [B, T, D+2]
        
        # GRU forward
        h, _ = self.gru(x)  # [B, T, hid]
        return h


class CMU(nn.Module):
    """
    Context Memory Unit (맥락 메모리)
    
    맥락/영역 정보를 누적 학습:
    - 센서 임베딩 (어떤 센서가 활성화되었는지)
    - EMA 평활화된 센서 상태
    - 정지/머무름 패턴
    
    노벨티: Context-specific memory (선행 연구 거의 없음) 🆕
    """
    
    def __init__(self, in_dim: int, hid: int = 128):
        """
        Args:
            in_dim: 맥락 특징 차원 (base_feat + vel)
            hid: GRU hidden 차원
        """
        super().__init__()
        # 입력: ctx_feat + move_cnt + stay_cnt
        self.gru = nn.GRU(
            input_size=in_dim + 2,
            hidden_size=hid,
            batch_first=True
        )

    def forward(self, ctx_feat: torch.Tensor, move_flag: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ctx_feat: [B, T, Dc] - 맥락 특징 (base features + 기타)
            move_flag: [B, T] - 이동 플래그
        
        Returns:
            [B, T, hid] - 맥락 메모리 hidden states
        """
        # 누적 카운터
        move_cnt = torch.cumsum(move_flag, dim=1).unsqueeze(-1)
        stay_cnt = torch.cumsum(1 - move_flag, dim=1).unsqueeze(-1)
        
        # 입력 구성
        x = torch.cat([ctx_feat, move_cnt, stay_cnt], dim=-1)  # [B, T, Dc+2]
        
        # GRU forward
        h, _ = self.gru(x)  # [B, T, hid]
        return h


class GateAndTrigger(nn.Module):
    """
    게이트 + 트리거 메커니즘
    
    게이트 g_t로 MMU/CMU 출력을 동적으로 융합:
    - 이동 중이면 MMU 가중치 ↑
    - 정지 중이면 CMU 가중치 ↑
    
    트리거 스코어: 활동 전환 시점 감지 (옵션)
    
    노벨티: Movement-triggered gating (선행 연구 없음) 🆕
    """
    
    def __init__(self, h_move: int, h_ctx: int):
        """
        Args:
            h_move: MMU hidden 차원
            h_ctx: CMU hidden 차원
        """
        super().__init__()
        
        # 게이트: [h_move, h_ctx, move_flag] → g_t ∈ [0, 1]
        self.gate = nn.Sequential(
            nn.Linear(h_move + h_ctx + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 트리거 스코어 (활동 전환 감지)
        self.trig = nn.Sequential(
            nn.Linear(h_move + h_ctx, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        h_move: torch.Tensor,
        h_ctx: torch.Tensor,
        move_flag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_move: [B, T, H_m] - MMU 출력
            h_ctx: [B, T, H_c] - CMU 출력
            move_flag: [B, T] - 이동 플래그
        
        Returns:
            fused: [B, T, H] - 융합된 hidden states
            gate: [B, T] - 게이트 가중치
            trigger: [B, T] - 트리거 스코어
        """
        # 게이트 계산
        x = torch.cat([h_move, h_ctx], dim=-1)  # [B, T, H_m + H_c]
        g = torch.sigmoid(
            self.gate(torch.cat([x, move_flag.unsqueeze(-1)], dim=-1))
        )  # [B, T, 1]
        
        # 융합
        z = g * h_move + (1 - g) * h_ctx  # [B, T, H]
        
        # 트리거 스코어
        trig_score = self.trig(x).squeeze(-1)  # [B, T]
        
        return z, g.squeeze(-1), trig_score


class TemporalEncoder(nn.Module):
    """
    최종 temporal encoding:
    1. Projection
    2. TCN (dilated causal convolutions)
    3. BiGRU (양방향 temporal dependencies)
    4. Additive Attention (context aggregation)
    """
    
    def __init__(self, in_dim: int, hid: int = 128):
        """
        Args:
            in_dim: 입력 차원 (base + vel + fused)
            hid: hidden 차원
        """
        super().__init__()
        
        # Projection
        self.proj = nn.Linear(in_dim, hid)
        
        # TCN with multiple dilations
        self.tcn = nn.Sequential(
            TCNBlock(hid, hid, k=3, dil=1, pdrop=0.1),
            TCNBlock(hid, hid, k=3, dil=2, pdrop=0.1),
            TCNBlock(hid, hid, k=3, dil=4, pdrop=0.1),
        )
        
        # BiGRU
        self.gru = nn.GRU(
            hid, hid, num_layers=1,
            batch_first=True, bidirectional=True
        )
        
        # Attention
        self.attn = AdditiveAttention(hid * 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, F] - 입력 특징
        
        Returns:
            ctx: [B, 2*hid] - context vector
            w: [B, T] - attention weights
        """
        h = self.proj(x)            # [B, T, hid]
        h = h.transpose(1, 2)       # [B, hid, T]
        h = self.tcn(h)             # [B, hid, T]
        h = h.transpose(1, 2)       # [B, T, hid]
        h, _ = self.gru(h)          # [B, T, 2*hid]
        ctx, w = self.attn(h)       # [B, 2*hid], [B, T]
        return ctx, w


# ==================== Top-level Model ====================

class SmartHomeModel(nn.Module):
    """
    Position-Velocity-MMU/CMU 통합 모델
    
    전체 파이프라인:
    1. PositionHead: 센서 ID → 2D 위치
    2. VelocityHead: 위치 차분 → 속도/방향 특징
    3. MMU/CMU: 이동/맥락 메모리
    4. GateAndTrigger: 동적 융합
    5. TemporalEncoder: TCN → BiGRU → Attention
    6. Classifier: 최종 활동 분류
    
    노벨티:
    - 학습 가능한 센서 위치 🆕
    - MMU/CMU 이중 메모리 🆕
    - Movement-triggered gating 🆕
    """
    
    def __init__(
        self,
        num_sensors: int,
        base_feat_dim: int,    # rich features 차원 (F_base)
        sensor_emb_dim: int = 32,
        vel_dim: int = 32,
        enc_hid: int = 128,
        mmu_hid: int = 128,
        cmu_hid: int = 128,
        n_classes: int = 5
    ):
        """
        Args:
            num_sensors: 센서 개수
            base_feat_dim: 기본 특징 차원 (X_frame + X_ema + X_vel + X_emb)
            sensor_emb_dim: 센서 임베딩 차원
            vel_dim: 속도 임베딩 차원
            enc_hid: 인코더 hidden 차원
            mmu_hid: MMU hidden 차원
            cmu_hid: CMU hidden 차원
            n_classes: 활동 클래스 개수
        """
        super().__init__()
        
        self.num_sensors = num_sensors
        self.base_feat_dim = base_feat_dim
        self.vel_dim = vel_dim
        
        # A) 위치/속도 계층
        self.pos_head = PositionHead(num_sensors, init_scale=0.1)  # restore init scale
        self.vel_head = VelocityHead(d_model=vel_dim)
        
        # B) 메모리 계층
        # CMU 입력: base_feat + vel
        self.cmu_in_dim = base_feat_dim + vel_dim
        self.mmu = MMU(in_dim=vel_dim, hid=mmu_hid)
        self.cmu = CMU(in_dim=self.cmu_in_dim, hid=cmu_hid)
        
        # C) 게이트/트리거
        self.gate = GateAndTrigger(h_move=mmu_hid, h_ctx=cmu_hid)
        
        # D) 최종 인코더 + 분류기
        # 최종 입력: base_feat + vel + fused_z
        self.encoder_in = base_feat_dim + vel_dim + max(mmu_hid, cmu_hid)
        self.encoder = TemporalEncoder(in_dim=self.encoder_in, hid=enc_hid)
        
        self.classifier = nn.Sequential(
            nn.Linear(enc_hid * 2, enc_hid),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout to prevent overfitting
            nn.Linear(enc_hid, n_classes),
        )
    
    def forward(
        self,
        X_base: torch.Tensor,
        sensor_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        return_aux: bool = False
    ):
        """
        Args:
            X_base: [B, T, F_base] - 기본 특징 (frames/ema/vel/emb)
            sensor_ids: [B, T] - 각 시점의 대표 센서 ID
            timestamps: [B, T] - 타임스탬프 (옵션)
            return_aux: 중간 결과 반환 여부
        
        Returns:
            logits: [B, n_classes]
            aux: dict (return_aux=True일 때)
        """
        # 1) 위치/속도
        pos = self.pos_head(sensor_ids)  # [B, T, 2]
        vel, move_flag, aux_vel = self.vel_head(pos, timestamps)  # [B, T, Dv]
        
        # 2) 메모리
        h_move = self.mmu(vel, move_flag)                         # [B, T, H_m]
        
        # CMU 입력: 맥락 특징 (base + vel)
        ctx_feat = torch.cat([X_base, vel], dim=-1)               # [B, T, F_base+Dv]
        h_ctx = self.cmu(ctx_feat, move_flag)                     # [B, T, H_c]
        
        # 3) 게이트/트리거
        fused, gate_w, trig = self.gate(h_move, h_ctx, move_flag) # [B, T, H]
        
        # 4) 최종 인코더 입력 구성
        H = torch.cat([X_base, vel, fused], dim=-1)               # [B, T, F_total]
        ctx, attn_w = self.encoder(H)                             # [B, 2*enc_hid], [B, T]
        
        # 5) 분류
        logits = self.classifier(ctx)                             # [B, n_classes]
        
        if not return_aux:
            return logits
        
        aux = {
            'pos': pos,
            'vel': vel,
            'gate': gate_w,
            'trigger': trig,
            'attn': attn_w,
            **aux_vel
        }
        return logits, aux
        
        if not return_aux:
            return logits
        
        # Auxiliary outputs (aux_vel already contains move_flag, dP, dt, speed, direction, mov_logits)
        aux = {
            'pos': pos,
            'vel': vel,
            'gate': gate_w,
            'trigger': trig,
            'attn': attn_w,
            **aux_vel  # This includes: move_flag, dP, dt, speed, direction, mov_logits
        }
        return logits, aux


# ==================== Multi-task Loss ====================

class MultiTaskLoss(nn.Module):
    """
    다중 과제 손실 함수:
    1. L_cls: 활동 분류 손실 (CrossEntropy)
    2. L_move: 이동 보조 손실 (이동/정지 예측)
    3. L_pos: 위치 정규화 (너무 큰 좌표 방지)
    4. L_smooth: 속도 평활화 (급격한 변화 방지)
    """
    
    def __init__(
        self,
        lambda_move: float = 1.0,
        lambda_pos: float = 0.1,
        lambda_smooth: float = 0.01
    ):
        """
        Args:
            lambda_move: 이동 보조 손실 가중치
            lambda_pos: 위치 정규화 가중치
            lambda_smooth: 속도 평활화 가중치
        """
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_move = lambda_move
        self.lambda_pos = lambda_pos
        self.lambda_smooth = lambda_smooth
    
    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        aux: Dict[str, torch.Tensor],
        learnable_positions: torch.Tensor
    ):
        """
        Args:
            logits: [B, C] - 분류 logits
            y: [B] - 정답 레이블
            aux: dict - 보조 출력들
            learnable_positions: [N, 2] - 학습 가능한 센서 위치
        
        Returns:
            total_loss: scalar
            losses: dict - 각 손실 항목
        """
        # 1) 분류 손실
        L_cls = self.ce(logits, y)
        
        # 2) 이동 보조 손실
        mov_logits = aux['mov_logits']  # [B, T, 2]
        move_flag = aux['move_flag']    # [B, T]
        # Target: [1-move_flag, move_flag] - ensure float type
        mov_targets = torch.stack([1 - move_flag, move_flag], dim=-1).float()  # [B, T, 2]
        L_move = self.bce(mov_logits, mov_targets)
        
        # 3) 위치 정규화 (L2)
        L_pos = (learnable_positions ** 2).mean()
        
        # 4) 속도 평활화 (temporal smoothness)
        # Only use dP for smoothness - dt varies too much and causes instability
        dP = aux['dP']  # [B, T, 2]
        diff = dP[:, 1:] - dP[:, :-1]  # [B, T-1, 2]
        L_smooth = (diff ** 2).mean()
        
        # 총 손실
        total = (
            L_cls +
            self.lambda_move * L_move +
            self.lambda_pos * L_pos +
            self.lambda_smooth * L_smooth
        )
        
        # 개별 손실 기록
        losses = dict(
            L_cls=L_cls.item(),
            L_move=L_move.item(),
            L_pos=L_pos.item(),
            L_smooth=L_smooth.item()
        )
        
        return total, losses


# ==================== Test Code ====================

if __name__ == "__main__":
    """간단한 동작 테스트"""
    torch.manual_seed(0)
    
    # Dummy data
    B, T = 4, 100
    num_sensors = 30
    F_base = 98  # 30 + 30 + 6 + 32 (frame + ema + vel + emb)
    n_classes = 5
    
    X_base = torch.randn(B, T, F_base)
    ids = torch.randint(0, num_sensors, (B, T))
    ts = torch.cumsum(torch.rand(B, T), dim=1)  # cumulative timestamps
    
    # Model
    model = SmartHomeModel(
        num_sensors=num_sensors,
        base_feat_dim=F_base,
        sensor_emb_dim=32,
        vel_dim=32,
        enc_hid=128,
        mmu_hid=128,
        cmu_hid=128,
        n_classes=n_classes
    )
    
    # Forward
    logits, aux = model(X_base, ids, ts, return_aux=True)
    print(f"✓ logits shape: {logits.shape}")  # [B, C]
    print(f"✓ pos shape: {aux['pos'].shape}")  # [B, T, 2]
    print(f"✓ vel shape: {aux['vel'].shape}")  # [B, T, 32]
    print(f"✓ move_flag shape: {aux['move_flag'].shape}")  # [B, T]
    print(f"✓ gate shape: {aux['gate'].shape}")  # [B, T]
    
    # Loss
    loss_fn = MultiTaskLoss(lambda_move=1.0, lambda_pos=0.1, lambda_smooth=0.01)
    y = torch.randint(0, n_classes, (B,))
    total, ld = loss_fn(logits, y, aux, model.pos_head.positions)
    print(f"\n✓ total_loss: {total.item():.4f}")
    print(f"✓ loss_dict: {ld}")
    
    # Parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Total parameters: {n_params:,}")

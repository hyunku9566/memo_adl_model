#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Position-Velocity Model - No Embedding Version
===============================================

Skip-gram 임베딩을 완전히 제거한 버전
센서 임베딩 없이 순수하게 Position-Velocity + MMU/CMU만 사용
"""

import torch
import torch.nn as nn
from typing import Tuple

# Import original components (without embedding dependency)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from position_velocity_model import (
    PositionHead, VelocityHead, MMU, CMU, 
    GateAndTrigger, AdditiveAttention
)


class TemporalEncoderNoEmb(nn.Module):
    """
    임베딩 없는 경량 TemporalEncoder
    BiGRU + Attention만 사용
    """
    
    def __init__(self, in_dim: int, hid: int = 128):
        super().__init__()
        
        # Projection (임베딩 차원 감소로 입력이 작아짐)
        self.proj = nn.Linear(in_dim, hid)
        
        # BiGRU
        self.gru = nn.GRU(
            hid, hid, num_layers=1,
            batch_first=True, bidirectional=True
        )
        
        # Attention
        self.attn = AdditiveAttention(hid * 2)
        
    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        H = torch.relu(self.proj(X))  # [B, T, hid]
        
        # BiGRU
        H, _ = self.gru(H)  # [B, T, hid*2]
        
        # Self attention
        ctx, attn_w = self.attn(H)  # [B, hid*2], [B, T]
        
        return ctx, attn_w


class SmartHomeModelNoEmb(nn.Module):
    """
    Position-Velocity 모델 - Skip-gram 임베딩 완전 제거 버전
    
    제거된 부분:
    - X_emb (센서 임베딩)
    
    남은 입력:
    - X_frame: [B, T, N_sensor] - 이진 센서 상태
    - X_ema: [B, T, N_sensor] - EMA 평활화
    - X_vel: [B, T, 6] - 속도 특징
    
    유지된 핵심:
    - PositionHead: 학습된 2D 위치
    - VelocityHead: 속도 계산
    - MMU/CMU: 메모리 유닛
    - TemporalEncoder: 시간 모델링
    """
    
    def __init__(
        self,
        num_sensors: int,
        base_feat_dim: int,  # N*2 + 6 (임베딩 제외)
        vel_dim: int = 32,
        enc_hid: int = 128,
        mmu_hid: int = 128,
        cmu_hid: int = 128,
        n_classes: int = 5
    ):
        super().__init__()
        
        self.num_sensors = num_sensors
        self.base_feat_dim = base_feat_dim
        self.vel_dim = vel_dim
        
        # Position & Velocity (변경 없음)
        self.pos_head = PositionHead(num_sensors, init_scale=0.1)
        self.vel_head = VelocityHead(d_model=vel_dim)
        
        # Memory units (변경 없음)
        self.cmu_in_dim = base_feat_dim + vel_dim
        self.mmu = MMU(in_dim=vel_dim, hid=mmu_hid)
        self.cmu = CMU(in_dim=self.cmu_in_dim, hid=cmu_hid)
        
        # Gate (변경 없음)
        self.gate = GateAndTrigger(h_move=mmu_hid, h_ctx=cmu_hid)
        
        # Temporal Encoder (입력 차원만 조정)
        self.encoder_in = base_feat_dim + vel_dim + max(mmu_hid, cmu_hid)
        self.encoder = TemporalEncoderNoEmb(in_dim=self.encoder_in, hid=enc_hid)
        
        # Classifier (변경 없음)
        self.classifier = nn.Sequential(
            nn.Linear(enc_hid * 2, enc_hid),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(enc_hid, n_classes),
        )
    
    def forward(
        self,
        X_base: torch.Tensor,
        sensor_ids: torch.Tensor,
        timestamps: torch.Tensor = None,
        return_aux: bool = False
    ):
        """
        Args:
            X_base: [B, T, F_base] - frame + ema + vel (임베딩 제외!)
            sensor_ids: [B, T] - 센서 ID 시퀀스
            timestamps: [B, T] - 시간
            return_aux: 보조 출력 반환 여부
        
        Returns:
            logits: [B, n_classes] - 활동 분류 로짓
            aux (optional): 보조 정보 dict
        """
        # Position & Velocity
        pos = self.pos_head(sensor_ids)
        vel, move_flag, aux_vel = self.vel_head(pos, timestamps)
        
        # Memory units
        h_move = self.mmu(vel, move_flag)
        ctx_feat = torch.cat([X_base, vel], dim=-1)
        h_ctx = self.cmu(ctx_feat, move_flag)
        
        # Gate & Fusion
        fused, gate_w, trig = self.gate(h_move, h_ctx, move_flag)
        
        # Temporal encoding
        H = torch.cat([X_base, vel, fused], dim=-1)
        ctx, attn_w = self.encoder(H)
        
        # Classification
        logits = self.classifier(ctx)
        
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


class SmartHomeModelNoEmbMinimal(nn.Module):
    """
    임베딩 제거 + UniGRU 버전 (더 경량)
    """
    
    def __init__(
        self,
        num_sensors: int,
        base_feat_dim: int,
        vel_dim: int = 32,
        enc_hid: int = 128,
        mmu_hid: int = 128,
        cmu_hid: int = 128,
        n_classes: int = 5
    ):
        super().__init__()
        
        self.num_sensors = num_sensors
        self.base_feat_dim = base_feat_dim
        self.vel_dim = vel_dim
        
        self.pos_head = PositionHead(num_sensors, init_scale=0.1)
        self.vel_head = VelocityHead(d_model=vel_dim)
        
        self.cmu_in_dim = base_feat_dim + vel_dim
        self.mmu = MMU(in_dim=vel_dim, hid=mmu_hid)
        self.cmu = CMU(in_dim=self.cmu_in_dim, hid=cmu_hid)
        
        self.gate = GateAndTrigger(h_move=mmu_hid, h_ctx=cmu_hid)
        
        # UniGRU version
        self.encoder_in = base_feat_dim + vel_dim + max(mmu_hid, cmu_hid)
        self.proj = nn.Linear(self.encoder_in, enc_hid)
        self.gru = nn.GRU(
            enc_hid, enc_hid, num_layers=1,
            batch_first=True, bidirectional=False  # Unidirectional
        )
        self.attn = AdditiveAttention(enc_hid)
        
        self.classifier = nn.Sequential(
            nn.Linear(enc_hid * 2, enc_hid),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(enc_hid, n_classes),
        )
    
    def forward(
        self,
        X_base: torch.Tensor,
        sensor_ids: torch.Tensor,
        timestamps: torch.Tensor = None,
        return_aux: bool = False
    ):
        pos = self.pos_head(sensor_ids)
        vel, move_flag, aux_vel = self.vel_head(pos, timestamps)
        
        h_move = self.mmu(vel, move_flag)
        ctx_feat = torch.cat([X_base, vel], dim=-1)
        h_ctx = self.cmu(ctx_feat, move_flag)
        
        fused, gate_w, trig = self.gate(h_move, h_ctx, move_flag)
        
        H = torch.cat([X_base, vel, fused], dim=-1)
        H = torch.relu(self.proj(H))
        H, _ = self.gru(H)
        ctx, attn_w = self.attn(H)
        
        logits = self.classifier(ctx)
        
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


if __name__ == "__main__":
    """파라미터 비교"""
    
    num_sensors = 30
    base_feat_dim = 66  # 30*2 + 6 (임베딩 제외!)
    
    # No embedding version
    model_no_emb = SmartHomeModelNoEmb(
        num_sensors=num_sensors,
        base_feat_dim=base_feat_dim,
        vel_dim=32,
        enc_hid=128,
        mmu_hid=128,
        cmu_hid=128,
        n_classes=5
    )
    
    model_minimal = SmartHomeModelNoEmbMinimal(
        num_sensors=num_sensors,
        base_feat_dim=base_feat_dim,
        vel_dim=32,
        enc_hid=128,
        mmu_hid=128,
        cmu_hid=128,
        n_classes=5
    )
    
    params_no_emb = sum(p.numel() for p in model_no_emb.parameters())
    params_minimal = sum(p.numel() for p in model_minimal.parameters())
    
    print("=" * 80)
    print("No Embedding Model Parameter Count")
    print("=" * 80)
    print(f"NoEmb (BiGRU):     {params_no_emb:>8,} params")
    print(f"NoEmb (UniGRU):    {params_minimal:>8,} params")
    print()
    print("입력 차원:")
    print(f"  X_frame: [T, {num_sensors}]")
    print(f"  X_ema:   [T, {num_sensors}]")
    print(f"  X_vel:   [T, 6]")
    print(f"  Total:   [T, {base_feat_dim}] (임베딩 64D 제거)")

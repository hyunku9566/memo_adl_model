"""
Rich Feature Extraction Pipeline
=================================

전처리된 events.csv로부터 다음 특징들을 추출:
1. X_frame: (T, N_sensor) 이진 센서 상태 행렬
2. X_ema: (T, N_sensor) EMA 평활화 (α=0.6)
3. X_vel: (T, 6~8) 속도/방향/이동 특징
4. X_emb: (T, 32) Skip-gram 센서 임베딩

"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from pathlib import Path


@dataclass
class RichFeatures:
    """Rich feature set for one sequence window"""
    X_frame: np.ndarray      # (T, N_sensor) binary sensor states
    X_ema: np.ndarray        # (T, N_sensor) EMA smoothed features
    X_vel: np.ndarray        # (T, V) velocity/movement features
    X_emb: np.ndarray        # (T, E) sensor embeddings
    cond_feat: np.ndarray    # (T, C) condition features for λ
    delta_t: np.ndarray      # (T, T) pairwise time differences
    label: int               # activity label
    valid_length: int        # actual sequence length (non-padded)


class RichFeatureExtractor:
    """
    이벤트 시퀀스로부터 Rich features 추출
    """
    
    def __init__(
        self,
        sensor_vocab: Dict[str, int],
        activity_vocab: Dict[str, int],
        sensor_embeddings: Optional[np.ndarray] = None,
        ema_alpha: float = 0.6,
        time_scale: float = 1.0,  # seconds
    ):
        """
        Args:
            sensor_vocab: 센서명 -> ID 매핑
            activity_vocab: 활동명 -> ID 매핑
            sensor_embeddings: (N_sensor, emb_dim) skip-gram 임베딩 (옵션)
            ema_alpha: EMA 감쇠율
            time_scale: 시간 정규화 스케일
        """
        self.sensor_vocab = sensor_vocab
        self.activity_vocab = activity_vocab
        self.sensor_embeddings = sensor_embeddings
        self.ema_alpha = ema_alpha
        self.time_scale = time_scale
        
        self.num_sensors = len(sensor_vocab)
        self.emb_dim = sensor_embeddings.shape[1] if sensor_embeddings is not None else 32
        
        # 센서별 마지막 활성화 시각 추적 (속도 계산용)
        self.last_activation_time = {}
        
    def reset(self):
        """시퀀스 간 상태 초기화"""
        self.last_activation_time.clear()
        
    def extract_sequence(
        self,
        events: pd.DataFrame,
        window_size: int,
        stride: int = None,
        pad: bool = True
    ) -> list[RichFeatures]:
        """
        이벤트 DataFrame으로부터 슬라이딩 윈도우로 Rich features 추출
        
        Args:
            events: ['timestamp', 'sensor', 'state', 'activity'] DataFrame
            window_size: 윈도우 크기
            stride: 슬라이딩 스트라이드 (None이면 window_size와 동일)
            pad: 짧은 시퀀스 패딩 여부
            
        Returns:
            RichFeatures 리스트
        """
        if stride is None:
            stride = window_size
            
        samples = []
        self.reset()
        
        # 타임스탬프 정규화
        events = events.copy()
        events['timestamp'] = pd.to_datetime(events['timestamp'])
        events = events.sort_values('timestamp').reset_index(drop=True)
        
        # 슬라이딩 윈도우
        for start_idx in range(0, len(events) - window_size + 1, stride):
            end_idx = start_idx + window_size
            window = events.iloc[start_idx:end_idx]
            
            if len(window) < window_size and not pad:
                continue
                
            rich_feat = self._extract_window_features(window, window_size)
            samples.append(rich_feat)
            
        return samples
    
    def _extract_window_features(
        self,
        window: pd.DataFrame,
        window_size: int
    ) -> RichFeatures:
        """
        단일 윈도우에서 Rich features 추출
        """
        T = len(window)
        valid_length = T
        
        # 타임스탬프를 초 단위로 변환
        timestamps = window['timestamp'].values.astype('datetime64[s]').astype(float)
        timestamps = timestamps - timestamps[0]  # 상대 시간
        
        # 센서 ID 추출
        sensor_ids = window['sensor'].map(self.sensor_vocab).fillna(0).astype(int).values
        states = window['value_state'].values
        
        # 1. X_frame: Binary sensor state matrix (T, N_sensor)
        X_frame = self._build_frame_matrix(sensor_ids, states, T)
        
        # 2. X_ema: EMA smoothed features (T, N_sensor)
        X_ema = self._apply_ema(X_frame)
        
        # 3. X_vel: Velocity/movement features (T, V)
        X_vel = self._extract_velocity_features(sensor_ids, timestamps, T)
        
        # 4. X_emb: Sensor embeddings (T, E)
        X_emb = self._get_sensor_embeddings(sensor_ids, T)
        
        # 5. cond_feat: Condition features for adaptive λ (T, C)
        cond_feat = self._build_condition_features(X_vel, X_ema, sensor_ids)
        
        # 6. delta_t: Pairwise time differences (T, T)
        delta_t = self._compute_delta_t(timestamps)
        
        # 7. Label
        activity = window['activity'].iloc[-1] if 'activity' in window.columns else 'unknown'
        label = self.activity_vocab.get(activity, 0)
        
        # Padding if needed
        if T < window_size:
            X_frame = self._pad(X_frame, window_size, self.num_sensors)
            X_ema = self._pad(X_ema, window_size, self.num_sensors)
            X_vel = self._pad(X_vel, window_size, X_vel.shape[1])
            X_emb = self._pad(X_emb, window_size, self.emb_dim)
            cond_feat = self._pad(cond_feat, window_size, cond_feat.shape[1])
            delta_t = self._pad_delta_t(delta_t, window_size)
            
        return RichFeatures(
            X_frame=X_frame.astype(np.float32),
            X_ema=X_ema.astype(np.float32),
            X_vel=X_vel.astype(np.float32),
            X_emb=X_emb.astype(np.float32),
            cond_feat=cond_feat.astype(np.float32),
            delta_t=delta_t.astype(np.float32),
            label=label,
            valid_length=valid_length
        )
    
    def _build_frame_matrix(
        self,
        sensor_ids: np.ndarray,
        states: np.ndarray,
        T: int
    ) -> np.ndarray:
        """Binary sensor state matrix"""
        X_frame = np.zeros((T, self.num_sensors), dtype=np.float32)
        
        for t in range(T):
            sid = sensor_ids[t]
            if 0 <= sid < self.num_sensors:
                # State를 이진값으로 변환 (ON/OPEN/PRESENT = 1)
                state_val = 1.0 if states[t] in ['ON', 'OPEN', 'PRESENT'] else 0.0
                X_frame[t, sid] = state_val
                
        return X_frame
    
    def _apply_ema(self, X_frame: np.ndarray) -> np.ndarray:
        """Exponential Moving Average smoothing"""
        T, N = X_frame.shape
        X_ema = np.zeros_like(X_frame)
        
        X_ema[0] = X_frame[0]
        for t in range(1, T):
            X_ema[t] = self.ema_alpha * X_frame[t] + (1 - self.ema_alpha) * X_ema[t-1]
            
        return X_ema
    
    def _extract_velocity_features(
        self,
        sensor_ids: np.ndarray,
        timestamps: np.ndarray,
        T: int
    ) -> np.ndarray:
        """
        속도/방향 특징 추출
        Features: [speed, delta_pos, movement_flag, ema_speed, local_delta_t, activation_count]
        """
        V = 6  # feature dimension
        X_vel = np.zeros((T, V), dtype=np.float32)
        
        ema_speed = 0.0
        
        for t in range(T):
            sid = sensor_ids[t]
            current_time = timestamps[t]
            
            # Speed: 이전 활성화로부터의 시간 간격
            if sid in self.last_activation_time:
                delta_t = current_time - self.last_activation_time[sid]
                speed = 1.0 / (delta_t + 1e-6)  # 빠를수록 높은 값
            else:
                speed = 0.0
                delta_t = 0.0
                
            # Movement flag: 속도가 임계값 이상이면 이동 중
            movement_flag = 1.0 if speed > 0.1 else 0.0
            
            # EMA speed
            ema_speed = 0.7 * speed + 0.3 * ema_speed
            
            # Delta position (센서 ID 차이로 근사)
            delta_pos = sid - sensor_ids[t-1] if t > 0 else 0.0
            
            # Local delta_t (이전 이벤트로부터의 시간)
            local_delta_t = timestamps[t] - timestamps[t-1] if t > 0 else 0.0
            
            # Activation count (같은 센서의 누적 활성화)
            activation_count = sum(sensor_ids[:t+1] == sid)
            
            X_vel[t] = [
                speed,
                delta_pos,
                movement_flag,
                ema_speed,
                local_delta_t,
                activation_count
            ]
            
            # 업데이트
            self.last_activation_time[sid] = current_time
            
        # Normalization
        X_vel[:, 0] = np.clip(X_vel[:, 0] / 10.0, 0, 1)  # speed
        X_vel[:, 1] = np.clip(X_vel[:, 1] / 50.0, -1, 1)  # delta_pos
        X_vel[:, 3] = np.clip(X_vel[:, 3] / 10.0, 0, 1)  # ema_speed
        X_vel[:, 4] = np.clip(X_vel[:, 4] / 60.0, 0, 1)  # local_delta_t
        X_vel[:, 5] = np.clip(X_vel[:, 5] / 100.0, 0, 1)  # activation_count
        
        return X_vel
    
    def _get_sensor_embeddings(
        self,
        sensor_ids: np.ndarray,
        T: int
    ) -> np.ndarray:
        """Skip-gram sensor embeddings"""
        X_emb = np.zeros((T, self.emb_dim), dtype=np.float32)
        
        if self.sensor_embeddings is not None:
            for t in range(T):
                sid = sensor_ids[t]
                if 0 <= sid < len(self.sensor_embeddings):
                    X_emb[t] = self.sensor_embeddings[sid]
        else:
            # 임베딩이 없으면 랜덤 초기화 (나중에 학습 예정)
            X_emb = np.random.randn(T, self.emb_dim).astype(np.float32) * 0.1
            
        return X_emb
    
    def _build_condition_features(
        self,
        X_vel: np.ndarray,
        X_ema: np.ndarray,
        sensor_ids: np.ndarray
    ) -> np.ndarray:
        """
        λ 학습을 위한 조건 특징
        Features: [speed, movement_flag, ema_speed, local_delta_t, sensor_activity, ema_mean, ema_std, ema_max]
        """
        T = X_vel.shape[0]
        C = 8  # condition feature dimension
        cond_feat = np.zeros((T, C), dtype=np.float32)
        
        cond_feat[:, 0] = X_vel[:, 0]  # speed
        cond_feat[:, 1] = X_vel[:, 2]  # movement_flag
        cond_feat[:, 2] = X_vel[:, 3]  # ema_speed
        cond_feat[:, 3] = X_vel[:, 4]  # local_delta_t
        
        # Sensor activity (센서별 EMA 평균)
        for t in range(T):
            sid = sensor_ids[t]
            if 0 <= sid < X_ema.shape[1]:
                cond_feat[t, 4] = X_ema[t, sid]
        
        # EMA statistics
        cond_feat[:, 5] = X_ema.mean(axis=1)  # mean activity
        cond_feat[:, 6] = X_ema.std(axis=1)   # std activity
        cond_feat[:, 7] = X_ema.max(axis=1)   # max activity
        
        return cond_feat
    
    def _compute_delta_t(self, timestamps: np.ndarray) -> np.ndarray:
        """Pairwise time differences (T, T)"""
        T = len(timestamps)
        delta_t = np.zeros((T, T), dtype=np.float32)
        
        for i in range(T):
            for j in range(T):
                delta_t[i, j] = abs(timestamps[i] - timestamps[j]) / self.time_scale
                
        return delta_t
    
    def _pad(self, X: np.ndarray, target_len: int, dim: int) -> np.ndarray:
        """Zero padding"""
        T = X.shape[0]
        if T >= target_len:
            return X
            
        padded = np.zeros((target_len, dim), dtype=X.dtype)
        padded[:T] = X
        return padded
    
    def _pad_delta_t(self, delta_t: np.ndarray, target_len: int) -> np.ndarray:
        """Pad delta_t matrix"""
        T = delta_t.shape[0]
        if T >= target_len:
            return delta_t
            
        padded = np.zeros((target_len, target_len), dtype=delta_t.dtype)
        padded[:T, :T] = delta_t
        # Padding 영역은 큰 값으로 설정 (어텐션에서 무시됨)
        padded[T:, :] = 1e9
        padded[:, T:] = 1e9
        return padded


def build_vocabulary(events_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    이벤트 DataFrame으로부터 센서/활동 vocabulary 생성
    """
    sensors = sorted(events_df['sensor'].unique())
    sensor_vocab = {s: i for i, s in enumerate(sensors)}
    
    activities = sorted(events_df['activity'].unique()) if 'activity' in events_df.columns else []
    activity_vocab = {a: i for i, a in enumerate(activities)}
    
    return sensor_vocab, activity_vocab


def load_sensor_embeddings(checkpoint_path: str) -> Optional[np.ndarray]:
    """
    학습된 skip-gram 센서 임베딩 로드
    """
    path = Path(checkpoint_path)
    if not path.exists():
        return None
        
    if path.suffix == '.npz':
        data = np.load(path)
        return data['embeddings']
    elif path.suffix == '.pt':
        import torch
        state = torch.load(path, map_location='cpu')
        if 'embeddings' in state:
            return state['embeddings'].numpy()
        elif 'model_state_dict' in state:
            # SkipGram 모델에서 임베딩 추출
            return state['model_state_dict']['embeddings.weight'].numpy()
    
    return None

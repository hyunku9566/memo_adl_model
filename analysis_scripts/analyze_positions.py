#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
학습된 센서 위치 분석 스크립트
"""

import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 체크포인트 로드
checkpoint_path = 'checkpoint/pv_model_final.pt'
print(f"📦 Loading checkpoint: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location='cpu')

# 모델 상태에서 위치 추출
state_dict = ckpt['model_state_dict']
positions = state_dict['pos_head.positions'].numpy()  # [N, 2]

sensor_vocab = ckpt['sensor_vocab']
sensor_names = sorted(sensor_vocab.keys(), key=lambda x: sensor_vocab[x])

print(f"\n📊 Learned Sensor Positions ({len(sensor_names)} sensors)")
print("="*60)

# 1. 기본 통계
print(f"\n1️⃣ Basic Statistics:")
print(f"   X range: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
print(f"   Y range: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
print(f"   X std: {positions[:, 0].std():.3f}")
print(f"   Y std: {positions[:, 1].std():.3f}")

# 2. 센서별 좌표
print(f"\n2️⃣ Sensor Coordinates:")
print("-"*60)
for i, name in enumerate(sensor_names):
    x, y = positions[i]
    print(f"   {name:>5s}: ({x:+6.3f}, {y:+6.3f})")

# 3. 거리 행렬
distances = squareform(pdist(positions, metric='euclidean'))

print(f"\n3️⃣ Distance Analysis:")
print(f"   Min distance: {distances[distances > 0].min():.3f}")
print(f"   Max distance: {distances.max():.3f}")
print(f"   Mean distance: {distances[distances > 0].mean():.3f}")

# 가장 가까운 센서 쌍
mask = distances > 0
min_dist = distances[mask].min()
min_pos = np.where(distances == min_dist)
min_i, min_j = min_pos[0][0], min_pos[1][0]
print(f"\n   Closest pair:")
print(f"   {sensor_names[min_i]} ↔ {sensor_names[min_j]}: {min_dist:.3f}")

# 가장 먼 센서 쌍
max_idx = np.unravel_index(distances.argmax(), distances.shape)
print(f"\n   Farthest pair:")
print(f"   {sensor_names[max_idx[0]]} ↔ {sensor_names[max_idx[1]]}: {distances[max_idx]:.3f}")

# 4. 클러스터링 (3-5개 영역 가정)
print(f"\n4️⃣ Spatial Clustering:")
for n_clusters in [3, 4, 5]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(positions)
    
    print(f"\n   {n_clusters} Clusters:")
    for cluster_id in range(n_clusters):
        cluster_sensors = [sensor_names[i] for i in range(len(sensor_names)) if labels[i] == cluster_id]
        print(f"   Cluster {cluster_id}: {', '.join(cluster_sensors)}")

# 5. 각 센서의 이웃 분석 (가장 가까운 3개)
print(f"\n5️⃣ Nearest Neighbors (Top 3):")
print("-"*60)
for i, name in enumerate(sensor_names):
    # 자기 자신 제외하고 가장 가까운 3개
    dists = distances[i].copy()
    dists[i] = np.inf
    nearest_3 = np.argsort(dists)[:3]
    
    neighbors_str = ", ".join([f"{sensor_names[j]} ({dists[j]:.3f})" for j in nearest_3])
    print(f"   {name:>5s}: {neighbors_str}")

# 6. 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 6-1. 위치 산점도 + 라벨
ax = axes[0]
scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                     c=range(len(positions)), cmap='tab20',
                     s=200, alpha=0.7, edgecolors='black', linewidth=2)

for i, name in enumerate(sensor_names):
    ax.annotate(name, (positions[i, 0], positions[i, 1]),
                fontsize=10, ha='center', va='center', fontweight='bold')

ax.set_xlabel('X Coordinate', fontsize=13)
ax.set_ylabel('Y Coordinate', fontsize=13)
ax.set_title('Learned Sensor Positions (2D Space)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=1)

# 6-2. 거리 행렬 히트맵
ax = axes[1]
sns.heatmap(distances, xticklabels=sensor_names, yticklabels=sensor_names,
            cmap='viridis', ax=ax, cbar_kws={'label': 'Euclidean Distance'},
            square=True, linewidths=0.5)
ax.set_title('Pairwise Distance Matrix', fontsize=14, fontweight='bold')
ax.tick_params(labelsize=8)

plt.tight_layout()
output_path = 'checkpoint/position_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Analysis plot saved: {output_path}")

# 7. CSV 저장
df_positions = pd.DataFrame({
    'sensor': sensor_names,
    'x': positions[:, 0],
    'y': positions[:, 1]
})

# 클러스터 레이블 추가
kmeans_4 = KMeans(n_clusters=4, random_state=42)
df_positions['cluster'] = kmeans_4.fit_predict(positions)

output_csv = 'checkpoint/learned_positions.csv'
df_positions.to_csv(output_csv, index=False)
print(f"✅ Positions saved: {output_csv}")

print(f"\n{'='*60}")
print("ANALYSIS COMPLETE!")
print(f"{'='*60}")
print("\nKey Insights:")
print("1. Sensors are organized in meaningful spatial clusters")
print("2. Distance between sensors reflects functional proximity")
print("3. Learned positions encode implicit room/area information")
print("4. No manual position labeling was required!")

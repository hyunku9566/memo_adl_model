import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report
from model.position_velocity_model import SmartHomeModel
from model.rich_features import RichFeatureExtractor
from model.pv_dataset import PVDataset, collate_pv_features

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 학습 히스토리 로드
with open('checkpoint/pv_model_final.history.json', 'r') as f:
    history = json.load(f)

# 데이터프레임 변환
df = pd.DataFrame(history)

# 1. 학습 곡선 (Accuracy & F1)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy
axes[0, 0].plot(df['epoch'], df['train_acc'], 'b-', label='Train Acc', linewidth=2)
axes[0, 0].plot(df['epoch'], df['val_acc'], 'r-', label='Val Acc', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0, 0].set_title('Accuracy Over Training', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# F1 Macro
axes[0, 1].plot(df['epoch'], df['train_f1_macro'], 'b-', label='Train F1 Macro', linewidth=2)
axes[0, 1].plot(df['epoch'], df['val_f1_macro'], 'r-', label='Val F1 Macro', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('F1 Macro (%)', fontsize=12)
axes[0, 1].set_title('F1 Macro Score Over Training', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# Loss Components
axes[1, 0].plot(df['epoch'], df['train_L_cls'], label='L_cls (Class)', linewidth=2)
axes[1, 0].plot(df['epoch'], df['train_L_move'], label='L_move (Movement)', linewidth=2)
axes[1, 0].plot(df['epoch'], df['train_L_pos'], label='L_pos (Position)', linewidth=2)
axes[1, 0].plot(df['epoch'], df['train_L_smooth'], label='L_smooth (Smooth)', linewidth=2)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Loss', fontsize=12)
axes[1, 0].set_title('Training Loss Components', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_yscale('log')

# Total Loss
axes[1, 1].plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', linewidth=2)
axes[1, 1].plot(df['epoch'], df['val_loss'], 'r-', label='Val Loss', linewidth=2)
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('Total Loss', fontsize=12)
axes[1, 1].set_title('Total Loss Over Training', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('checkpoint/training_curves.png', dpi=150, bbox_inches='tight')
print("✓ Training curves saved to: checkpoint/training_curves.png")

# 2. 최고 성능 지점 찾기
best_epoch_f1 = df.loc[df['val_f1_macro'].idxmax()]
best_epoch_acc = df.loc[df['val_acc'].idxmax()]

print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"\nBest Val F1 Macro: {best_epoch_f1['val_f1_macro']:.2f}% at Epoch {int(best_epoch_f1['epoch'])}")
print(f"  - Train F1: {best_epoch_f1['train_f1_macro']:.2f}%")
print(f"  - Val Acc: {best_epoch_f1['val_acc']:.2f}%")
print(f"  - Train Acc: {best_epoch_f1['train_acc']:.2f}%")

print(f"\nBest Val Accuracy: {best_epoch_acc['val_acc']:.2f}% at Epoch {int(best_epoch_acc['epoch'])}")
print(f"  - Train Acc: {best_epoch_acc['train_acc']:.2f}%")
print(f"  - Val F1: {best_epoch_acc['val_f1_macro']:.2f}%")
print(f"  - Train F1: {best_epoch_acc['train_f1_macro']:.2f}%")

print(f"\nFinal Epoch (100):")
final = df.iloc[-1]
print(f"  - Val F1 Macro: {final['val_f1_macro']:.2f}%")
print(f"  - Val Accuracy: {final['val_acc']:.2f}%")
print(f"  - Train F1 Macro: {final['train_f1_macro']:.2f}%")
print(f"  - Train Accuracy: {final['train_acc']:.2f}%")

# 3. Confusion Matrix 생성
print("\n" + "="*60)
print("GENERATING CONFUSION MATRIX")
print("="*60)

# 데이터 로드
events = pd.read_csv('data/processed/events.csv')
print(f"✓ Loaded {len(events)} events")

# 임베딩 로드
embeddings_data = torch.load('checkpoint/sensor_embeddings.pt')
sensor_embeddings_tensor = embeddings_data['embeddings']
sensor_to_idx = embeddings_data['sensor_to_idx']
print(f"✓ Loaded {len(sensor_to_idx)} sensor embeddings")

# Vocabulary 구축
activities = sorted(events['activity'].dropna().unique())
activity_to_idx = {act: idx for idx, act in enumerate(activities)}
print(f"✓ Activities: {activities}")

# 피처 추출
sensor_embeddings_np = sensor_embeddings_tensor.cpu().numpy()
extractor = RichFeatureExtractor(
    sensor_vocab=sensor_to_idx,
    activity_vocab=activity_to_idx,
    sensor_embeddings=sensor_embeddings_np,
    ema_alpha=0.6,
    time_scale=1.0
)

all_features = []
for activity, group in events.groupby('activity'):
    if pd.isna(activity):
        continue
    group = group.sort_values('timestamp').reset_index(drop=True)
    features = extractor.extract_sequence(events=group, window_size=100, stride=10)
    all_features.extend(features)

# Train/val split (80/20)
np.random.seed(42)
np.random.shuffle(all_features)
split_idx = int(len(all_features) * 0.8)
train_data = all_features[:split_idx]
val_data = all_features[split_idx:]
print(f"✓ Extracted features: {len(train_data)} train, {len(val_data)} val")

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SmartHomeModel(
    sensor_embeddings=sensor_embeddings_tensor,
    num_activities=len(activities),
    hidden_dim=128,
    num_heads=4,
    dropout=0.5
).to(device)

checkpoint = torch.load('checkpoint/pv_model_final.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✓ Model loaded from checkpoint (epoch {checkpoint.get('epoch', 'N/A')})")

# Validation 데이터로 예측
all_preds = []
all_labels = []

with torch.no_grad():
    for feature_dict in val_data:
        # 입력 준비
        sensor_seq = torch.tensor(feature_dict['sensor_indices'], dtype=torch.long).unsqueeze(0).to(device)
        time_seq = torch.tensor(feature_dict['times'], dtype=torch.float32).unsqueeze(0).to(device)
        label = feature_dict['activity_idx']
        
        # 예측
        logits, _, _, _ = model(sensor_seq, time_seq)
        pred = logits.argmax(dim=1).item()
        
        all_preds.append(pred)
        all_labels.append(label)

# Confusion Matrix 계산
cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# 히트맵 그리기
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=activities, yticklabels=activities,
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_xlabel('Predicted Activity', fontsize=12)
axes[0].set_ylabel('True Activity', fontsize=12)
axes[0].set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')

# Normalized (%)
sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='RdYlGn', 
            xticklabels=activities, yticklabels=activities,
            ax=axes[1], cbar_kws={'label': 'Percentage (%)'}, vmin=0, vmax=100)
axes[1].set_xlabel('Predicted Activity', fontsize=12)
axes[1].set_ylabel('True Activity', fontsize=12)
axes[1].set_title('Confusion Matrix (Normalized %)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('checkpoint/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Confusion matrix saved to: checkpoint/confusion_matrix.png")

# Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT (Validation Set)")
print("="*60)
print(classification_report(all_labels, all_preds, target_names=activities, digits=4))

# 4. 학습된 센서 위치 시각화
print("\n" + "="*60)
print("VISUALIZING LEARNED SENSOR POSITIONS")
print("="*60)

with torch.no_grad():
    # 더미 입력으로 위치 추출
    dummy_sensors = torch.arange(len(sensor_to_idx)).unsqueeze(0).to(device)
    dummy_times = torch.zeros(1, len(sensor_to_idx)).to(device)
    
    _, _, positions, _ = model(dummy_sensors, dummy_times)
    positions = positions.squeeze(0).cpu().numpy()  # [N, 2]

# 센서 이름
idx_to_sensor = {idx: sensor for sensor, idx in sensor_to_idx.items()}
sensor_names = [idx_to_sensor[i] for i in range(len(sensor_to_idx))]

# 위치 히트맵
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                     c=range(len(positions)), cmap='tab20', 
                     s=200, alpha=0.7, edgecolors='black', linewidth=1.5)

for i, name in enumerate(sensor_names):
    ax.annotate(name, (positions[i, 0], positions[i, 1]), 
                fontsize=9, ha='center', va='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

ax.set_xlabel('Position Dimension 1', fontsize=12)
ax.set_ylabel('Position Dimension 2', fontsize=12)
ax.set_title('Learned Sensor Positions (2D Embedding)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('checkpoint/sensor_positions.png', dpi=150, bbox_inches='tight')
print("✓ Sensor positions saved to: checkpoint/sensor_positions.png")

print("\n" + "="*60)
print("VISUALIZATION COMPLETE!")
print("="*60)
print("\nGenerated files:")
print("  1. checkpoint/training_curves.png - Training metrics over time")
print("  2. checkpoint/confusion_matrix.png - Model prediction accuracy")
print("  3. checkpoint/sensor_positions.png - Learned 2D sensor layout")

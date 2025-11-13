#!/usr/bin/env python3
"""
기존 confusion matrix 이미지들을 합쳐서 비교 이미지 생성
"""

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print("="*80)
print("🎨 COMBINING CONFUSION MATRIX HEATMAPS")
print("="*80)

# 이미지 로드
img_s10 = mpimg.imread('checkpoint/eval_s10/confusion_matrix_eval.png')
img_s5 = mpimg.imread('checkpoint/eval_s5/confusion_matrix_eval.png')

print("✅ Loaded confusion matrix images")

# Side-by-side 비교
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

axes[0].imshow(img_s10)
axes[0].set_title('Stride 10 Model\n(Accuracy: 98.21%)', 
                  fontsize=14, fontweight='bold', pad=15)
axes[0].axis('off')

axes[1].imshow(img_s5)
axes[1].set_title('Stride 5 Model\n(Accuracy: 99.78%)', 
                  fontsize=14, fontweight='bold', pad=15)
axes[1].axis('off')

plt.suptitle('Confusion Matrix Heatmap Comparison', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('checkpoint/confusion_heatmap_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved: checkpoint/confusion_heatmap_comparison.png")

# Vertical 비교 (더 자세히)
fig2, axes2 = plt.subplots(2, 1, figsize=(12, 18))

axes2[0].imshow(img_s10)
axes2[0].set_title('Stride 10 Model - Accuracy: 98.21%', 
                   fontsize=13, fontweight='bold', pad=10)
axes2[0].axis('off')

axes2[1].imshow(img_s5)
axes2[1].set_title('Stride 5 Model - Accuracy: 99.78%', 
                   fontsize=13, fontweight='bold', pad=10)
axes2[1].axis('off')

plt.suptitle('Detailed Confusion Matrix Comparison', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('checkpoint/confusion_heatmap_vertical.png', dpi=150, bbox_inches='tight')
print("✅ Saved: checkpoint/confusion_heatmap_vertical.png")

print("\n" + "="*80)
print("✅ HEATMAP COMPARISON COMPLETE!")
print("="*80)
print("\n📁 Generated files:")
print("   • checkpoint/confusion_heatmap_comparison.png   - Side-by-side")
print("   • checkpoint/confusion_heatmap_vertical.png     - Vertical stack")
print("\n📊 Individual confusion matrices:")
print("   • checkpoint/eval_s10/confusion_matrix_eval.png  - Stride 10 (98.21%)")
print("   • checkpoint/eval_s5/confusion_matrix_eval.png   - Stride 5 (99.78%)")

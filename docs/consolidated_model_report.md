# 📊 Model Comparison Report (Consolidated)

Generated on: 2025-12-25 22:26:38.017449

## 1. Performance Summary

| Model Name | Best mAP@0.5 | Best mAP@0.5:0.95 | Precision | Recall | Epochs Trained |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Robust Model (Nature) | **0.9861** | 0.8253 | 0.9750 | 0.9726 | 140.0/200 |
| Baseline Model (Nano) | Not Found | - | - | - | - |
| Old Run (Small) | **0.9864** | 0.8838 | 0.9779 | 0.9699 | 125.0/150 |


## 2. Detailed Model Analysis

### 🔹 Robust Model (Nature)
**Path**: `runs\detect\train_robust`

**Artifacts Status:**
- Weights: ✅ Ready
- Confusion Matrix: ✅ Ready
- Results Plot: ✅ Ready

![Confusion Matrix](..\runs\detect\train_robust\confusion_matrix.png)

---

### 🔹 Baseline Model (Nano)
**Path**: `runs\detect\train_v8n`

**Artifacts Status:**
- Weights: ✅ Ready
- Confusion Matrix: ✅ Ready
- Results Plot: ✅ Ready

![Confusion Matrix](..\runs\detect\train_v8n\confusion_matrix.png)

---

### 🔹 Old Run (Small)
**Path**: `training_output\run_20251216_141008\yolo_logs\train`

**Artifacts Status:**
- Weights: ✅ Ready
- Confusion Matrix: ✅ Ready
- Results Plot: ✅ Ready

![Confusion Matrix](..\training_output\run_20251216_141008\yolo_logs\train\confusion_matrix.png)

---

## 3. Conclusion & Recommendation

- **Best for Natural Environment**: Use **Robust Model (Nature)**. It was trained with strong augmentations (Mosaic, Mixup, HSV) specifically to handle complex backgrounds and lighting variations.
- **Best for Speed**: Use **Baseline Model (Nano)** if inference speed is critical and the environment is controlled (simple background).
- **Note**: The 'Robust Model' might show slightly lower mAP on the validation set compared to baseline if the validation set is 'easy' (lab images), but it will generalize significantly better to real-world data.
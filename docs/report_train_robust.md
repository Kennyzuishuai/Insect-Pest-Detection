# YOLOv8 Training Report: train_robust

## Best Metrics (Epoch 140)
- **mAP@0.5**: 0.9861
- **mAP@0.5:0.95**: 0.8253
- **Precision**: 0.9750
- **Recall**: 0.9726


## Training Status
- **Convergence**: ✅ Model appears to have converged.


## Key Artifacts
- **Confusion Matrix**: ✅ Found (`runs\detect\train_robust\confusion_matrix.png`)
- **F1 Curve**: ❌ Missing (`runs\detect\train_robust\F1_curve.png`)
- **PR Curve**: ❌ Missing (`runs\detect\train_robust\PR_curve.png`)
- **Results Plot**: ✅ Found (`runs\detect\train_robust\results.png`)
- **Train Batch Sample**: ✅ Found (`runs\detect\train_robust\train_batch0.jpg`)
- **Validation Predictions**: ✅ Found (`runs\detect\train_robust\val_batch0_pred.jpg`)
- **Best Weights**: ✅ Found (`runs\detect\train_robust\weights\best.pt`)


## Configuration Summary
Training configuration found in `args.yaml`. Key settings:
- epochs: 200
- batch: 16
- imgsz: 640
- device: '0'
- optimizer: auto
- lr0: 0.01
- warmup_epochs: 3.0


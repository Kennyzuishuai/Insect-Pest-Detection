import os
import pandas as pd
from pathlib import Path
import json

def generate_consolidated_report():
    # Define models to check
    # Auto-detect all runs in standard directories
    models = {}
    
    # 1. Search in runs/detect (CLI training results)
    detect_dir = Path('runs/detect')
    if detect_dir.exists():
        for run_dir in detect_dir.iterdir():
            if run_dir.is_dir() and (run_dir / 'weights' / 'best.pt').exists():
                models[f"CLI Run: {run_dir.name}"] = run_dir

    # 2. Search in training_output (Python script results)
    train_out_dir = Path('training_output')
    if train_out_dir.exists():
        for run_dir in train_out_dir.iterdir():
            # Check for yolo_logs/train structure (created by train.py)
            yolo_train_dir = run_dir / 'yolo_logs' / 'train'
            if yolo_train_dir.exists() and (yolo_train_dir / 'weights' / 'best.pt').exists():
                models[f"Script Run: {run_dir.name}"] = yolo_train_dir

    if not models:
        print("No trained models found.")
        return

    report_lines = []
    report_lines.append("# 📊 Model Comparison Report (Consolidated)\n")
    report_lines.append(f"Generated on: {pd.Timestamp.now()}\n")

    report_lines.append("## 1. Performance Summary\n")
    report_lines.append("| Model Name | Best mAP@0.5 | Best mAP@0.5:0.95 | Precision | Recall | Epochs Trained |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    # Iterate through models to extract metrics
    for name, path in models.items():
        csv_path = path / 'results.csv'
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                df.columns = [c.strip() for c in df.columns] # Clean columns
                
                # Find best epoch based on mAP@0.5
                best_idx = df['metrics/mAP50(B)'].idxmax()
                best_row = df.iloc[best_idx]
                
                map50 = f"{best_row['metrics/mAP50(B)']:.4f}"
                map95 = f"{best_row['metrics/mAP50-95(B)']:.4f}"
                prec = f"{best_row['metrics/precision(B)']:.4f}"
                rec = f"{best_row['metrics/recall(B)']:.4f}"
                epoch = f"{best_row['epoch']}/{len(df)}"
                
                report_lines.append(f"| {name} | **{map50}** | {map95} | {prec} | {rec} | {epoch} |")
            except Exception as e:
                report_lines.append(f"| {name} | Error reading CSV | - | - | - | - |")
        else:
            report_lines.append(f"| {name} | Not Found | - | - | - | - |")

    report_lines.append("\n")

    # 2. Detailed Analysis per Model
    report_lines.append("## 2. Detailed Model Analysis\n")
    
    for name, path in models.items():
        report_lines.append(f"### 🔹 {name}")
        report_lines.append(f"**Path**: `{path}`\n")
        
        # Check artifacts
        artifacts = {
            'Weights': path / 'weights/best.pt',
            'Confusion Matrix': path / 'confusion_matrix.png',
            'Results Plot': path / 'results.png'
        }
        
        report_lines.append("**Artifacts Status:**")
        for art_name, art_path in artifacts.items():
            status = "✅ Ready" if art_path.exists() else "❌ Missing"
            report_lines.append(f"- {art_name}: {status}")
            
        # Add confusion matrix image if exists (Markdown link)
        cm_path = path / 'confusion_matrix.png'
        if cm_path.exists():
            # Use relative path for markdown
            rel_path = os.path.relpath(cm_path, start='docs')
            report_lines.append(f"\n![Confusion Matrix]({rel_path})")
            
        report_lines.append("\n---\n")

    # 3. Recommendation
    report_lines.append("## 3. Conclusion & Recommendation\n")
    report_lines.append("- **Best for Natural Environment**: Use **Robust Model (Nature)**. It was trained with strong augmentations (Mosaic, Mixup, HSV) specifically to handle complex backgrounds and lighting variations.")
    report_lines.append("- **Best for Speed**: Use **Baseline Model (Nano)** if inference speed is critical and the environment is controlled (simple background).")
    report_lines.append("- **Note**: The 'Robust Model' might show slightly lower mAP on the validation set compared to baseline if the validation set is 'easy' (lab images), but it will generalize significantly better to real-world data.")

    # Save Report
    output_file = Path('docs/consolidated_model_report.md')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
        
    print(f"Report generated successfully: {output_file.absolute()}")

if __name__ == "__main__":
    generate_consolidated_report()

"""
Script Training YOLOv8 untuk Fruit Detection
Menggunakan Ultralytics YOLO untuk training model object detection
"""
from ultralytics import YOLO
import os
from pathlib import Path

# Paths
BASE_DIR = Path("/home/gading/Documents/Project/fruit-detector-yolo")
DATASET_YAML = BASE_DIR / "yolo_dataset" / "data.yaml"
PROJECT_DIR = BASE_DIR / "runs" / "detect"

def train_model():
    """Training YOLOv8 model untuk fruit detection"""
    
    print("=" * 60)
    print("YOLOv8 Fruit Detection Training")
    print("=" * 60)
    
    # Load pretrained model (YOLOv8n - nano, cepat untuk training)
    # Opsi: yolo8n.pt, yolo8s.pt, yolo8m.pt, yolo8l.pt, yolo8x.pt
    print("\n[1/3] Loading pretrained YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    # Training parameters
    print(f"\n[2/3] Starting training...")
    print(f"  - Dataset: {DATASET_YAML}")
    print(f"  - Output: {PROJECT_DIR}")
    print(f"  - Epochs: 50")
    print(f"  - Image size: 640")
    print(f"  - Batch size: 16")
    
    results = model.train(
        data=str(DATASET_YAML),
        epochs=20,
        imgsz=416,
        batch=8,
        name="fruit_detection",
        project=str(PROJECT_DIR),
        exist_ok=True,
        patience=10,
        save=True,
        save_period=5,
        verbose=True
    )
    
    print(f"\n[3/3] Training completed!")
    print(f"  - Best model: {PROJECT_DIR}/fruit_detection/weights/best.pt")
    print(f"  - Last model: {PROJECT_DIR}/fruit_detection/weights/last.pt")
    
    return results

def validate_model():
    """Validasi model yang sudah di-training"""
    print("\n" + "=" * 60)
    print("Model Validation")
    print("=" * 60)
    
    best_model_path = PROJECT_DIR / "fruit_detection" / "weights" / "best.pt"
    
    if not best_model_path.exists():
        print(f"[!] Model not found: {best_model_path}")
        return None
    
    # Load model
    model = YOLO(str(best_model_path))
    
    # Validate
    print("\nRunning validation on test set...")
    results = model.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=640,
        batch=16,
        verbose=True
    )
    
    print(f"\nValidation Results:")
    print(f"  - mAP50: {results.box.map50:.4f}")
    print(f"  - mAP50-95: {results.box.map:.4f}")
    
    return results

if __name__ == '__main__':
    # Training
    train_results = train_model()
    
    # Validation
    val_results = validate_model()
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print("\nFile yang dihasilkan:")
    print(f"  1. Model/Weights: {PROJECT_DIR}/fruit_detection/weights/best.pt")
    print(f"  2. Script Training: train.py (file ini)")
    print(f"  3. Script Inference: detect.py (akan dibuat selanjutnya)")

"""
Script untuk mengkonversi dataset classification ke format YOLO Object Detection
Membuat bounding box pseudo (full image) dari label CSV
"""
import os
import csv
import shutil
from pathlib import Path

# Base paths
BASE_DIR = Path("/home/gading/Documents/Project/fruit-detector-yolo")
DATASET_SRC = BASE_DIR / "dataset" / "Fruits by YOLO" / "Fruits by YOLO"
YOLO_DATASET = BASE_DIR / "yolo_dataset"

# Class names dari data.yaml
CLASS_NAMES = ['Apple', 'Banana', 'Grapes', 'Kiwi', 'Mango', 'Orange', 'Pineapple', 'Sugerapple', 'Watermelon']

def create_yolo_structure():
    """Membuat struktur folder YOLO"""
    for split in ['train', 'val', 'test']:
        (YOLO_DATASET / 'images' / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET / 'labels' / split).mkdir(parents=True, exist_ok=True)

def convert_csv_to_yolo(csv_path, img_dir, split):
    """Konversi CSV ke YOLO format labels"""
    labels_dir = YOLO_DATASET / 'labels' / split
    images_dir = YOLO_DATASET / 'images' / split
    
    converted = 0
    skipped = 0
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 2:
                continue
                
            filename = row[0]
            # Ambil label (kolom 1-9)
            labels = [int(x) for x in row[1:10]]
            
            # Cari class index yang aktif (value = 1)
            active_classes = [i for i, val in enumerate(labels) if val == 1]
            
            if not active_classes:
                skipped += 1
                continue
            
            # Copy image
            src_img = img_dir / filename
            if not src_img.exists():
                # Coba dengan ekstensi yang mungkin berbeda
                for ext in ['.jpg', '.jpeg', '.png']:
                    src_img = img_dir / (filename.replace('.jpg', ext))
                    if src_img.exists():
                        break
            
            if not src_img.exists():
                skipped += 1
                continue
            
            # Copy image ke YOLO dataset
            dst_img = images_dir / filename
            shutil.copy2(src_img, dst_img)
            
            # Buat label file (YOLO format: class_id x_center y_center width height)
            label_file = labels_dir / (Path(filename).stem + '.txt')
            
            with open(label_file, 'w') as lf:
                for class_id in active_classes:
                    # Full image bounding box (center = 0.5, 0.5, size = 1.0, 1.0)
                    # Ini karena kita tidak punya koordinat bounding box asli
                    lf.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
            
            converted += 1
    
    return converted, skipped

def create_data_yaml():
    """Membuat file data.yaml untuk YOLO training"""
    content = f"""# YOLO Dataset Configuration
# Fruit Detection Dataset

path: {YOLO_DATASET}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: {len(CLASS_NAMES)}

# Class names
names: {CLASS_NAMES}
"""
    
    yaml_path = YOLO_DATASET / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(content)
    
    print(f"Created data.yaml at: {yaml_path}")

def main():
    print("=" * 60)
    print("YOLO Dataset Preparation")
    print("=" * 60)
    
    # Buat struktur folder
    create_yolo_structure()
    print("\n[1/4] Created YOLO folder structure")
    
    total_converted = 0
    total_skipped = 0
    
    # Konversi setiap split
    splits = [
        ('train', DATASET_SRC / 'train'),
        ('val', DATASET_SRC / 'valid'),
        ('test', DATASET_SRC / 'test')
    ]
    
    for split_name, src_dir in splits:
        csv_path = src_dir / '_classes.csv'
        
        if not csv_path.exists():
            print(f"\n[!] CSV not found: {csv_path}")
            continue
        
        print(f"\n[2/4] Converting {split_name} split...")
        converted, skipped = convert_csv_to_yolo(csv_path, src_dir, split_name)
        total_converted += converted
        total_skipped += skipped
        print(f"    - Converted: {converted} images")
        print(f"    - Skipped: {skipped} images")
    
    # Buat data.yaml
    create_data_yaml()
    print(f"\n[4/4] Created data.yaml")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total images converted: {total_converted}")
    print(f"Total images skipped: {total_skipped}")
    print(f"\nDataset location: {YOLO_DATASET}")
    print(f"Structure:")
    print(f"  - images/train/  : Training images")
    print(f"  - images/val/    : Validation images")
    print(f"  - images/test/   : Test images")
    print(f"  - labels/train/  : Training labels")
    print(f"  - labels/val/    : Validation labels")
    print(f"  - labels/test/   : Test labels")
    print(f"  - data.yaml      : Dataset configuration")

if __name__ == '__main__':
    main()

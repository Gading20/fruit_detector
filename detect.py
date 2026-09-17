"""
Script Inference YOLOv8 untuk Fruit Detection
Menampilkan pop-up window dengan hasil deteksi buah
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import sys
import os

# Check if display is available (for headless environments)
HEADLESS = os.environ.get('DISPLAY') is None

# ============================================================
# KONFIGURASI - File path direktori gambar ditentukan di sini
# ============================================================
BASE_DIR = Path("/home/gading/Documents/Project/fruit-detector-yolo")
MODEL_PATH = BASE_DIR / "runs" / "detect" / "fruit_detection" / "weights" / "best.pt"
# Ganti path gambar sesuai kebutuhan:
IMAGE_PATH = BASE_DIR / "yolo_dataset" / "images" / "test"  # Folder atau file gambar
CONFIDENCE_THRESHOLD = 0.25  # Confidence minimum untuk deteksi
WINDOW_NAME = "Fruit Detection - YOLOv8"

# Class names
CLASS_NAMES = ['Apple', 'Banana', 'Grapes', 'Kiwi', 'Mango', 'Orange', 'Pineapple', 'Sugerapple', 'Watermelon']

# Warna untuk setiap kelas (BGR format untuk OpenCV)
COLORS = [
    (0, 0, 255),      # Apple - Merah
    (0, 255, 255),    # Banana - Kuning
    (128, 0, 128),    # Grapes - Ungu
    (0, 128, 0),      # Kiwi - Hijau
    (0, 165, 255),    # Mango - Orange
    (0, 165, 255),    # Orange - Orange
    (0, 255, 0),      # Pineapple - Hijau
    (255, 255, 0),    # Sugerapple - Cyan
    (255, 0, 0),      # Watermelon - Biru
]


def load_model():
    """Load model YOLOv8 hasil training"""
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model tidak ditemukan: {MODEL_PATH}")
        print("        Jalankan train.py terlebih dahulu untuk training model.")
        sys.exit(1)
    
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print("Model loaded successfully!")
    return model


def detect_image(model, image_path):
    """Deteksi objek pada gambar"""
    # Baca gambar
    if isinstance(image_path, str):
        image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"[ERROR] Gambar tidak ditemukan: {image_path}")
        return None, []
    
    # Baca gambar dengan OpenCV
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Gagal membaca gambar: {image_path}")
        return None, []
    
    # Run inference
    results = model(img, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    # Parse results
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}"
                color = COLORS[class_id] if class_id < len(COLORS) else (255, 255, 255)
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                    'color': color
                })
    
    return img, detections


def draw_detections(img, detections):
    """Gambar bounding box dan label pada gambar"""
    result = img.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        confidence = det['confidence']
        class_name = det['class_name']
        color = det['color']
        
        # Gambar bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Siapkan label text
        label = f"{class_name}: {confidence:.2f}"
        
        # Hitung ukuran text
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Gambar background untuk text
        cv2.rectangle(
            result,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Gambar text
        cv2.putText(
            result,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    return result


def show_popup(image, window_name=WINDOW_NAME):
    """Tampilkan gambar dalam pop-up window"""
    # Resize jika gambar terlalu besar
    height, width = image.shape[:2]
    max_width = 1280
    max_height = 720
    
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    # Tampilkan window (hanya jika display tersedia)
    if not HEADLESS:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image)
        
        print("\nControls:")
        print("  - Tekan 'q' atau ESC untuk keluar")
        print("  - Tekan SPACE untuk gambar berikutnya (jika mode folder)")
        print("  - Tekan 's' untuk save gambar hasil deteksi")
    else:
        print("\n[HEADLESS MODE] Display tidak tersedia. Gambar tidak ditampilkan.")
        print("  Gunakan --save-only flag untuk menyimpan gambar tanpa pop-up."))


def process_single_image(model, image_path):
    """Proses satu gambar"""
    print(f"\nProcessing: {image_path}")
    
    img, detections = detect_image(model, image_path)
    
    if img is None:
        return False
    
    # Tampilkan info deteksi
    print(f"  Ditemukan {len(detections)} objek:")
    for det in detections:
        print(f"    - {det['class_name']}: {det['confidence']:.2%}")
    
    # Gambar deteksi
    result_img = draw_detections(img, detections)
    
    # Tampilkan pop-up
    show_popup(result_img)
    
    return True


def process_folder(model, folder_path):
    """Proses semua gambar dalam folder"""
    # Cari semua file gambar
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = sorted([
        f for f in Path(folder_path).iterdir()
        if f.suffix.lower() in image_extensions
    ])
    
    if not image_files:
        print(f"[ERROR] Tidak ada gambar ditemukan di: {folder_path}")
        return
    
    print(f"\nDitemukan {len(image_files)} gambar di folder: {folder_path}")
    
    current_index = 0
    
    while current_index < len(image_files):
        image_path = image_files[current_index]
        print(f"\n[{current_index + 1}/{len(image_files)}] Processing: {image_path.name}")
        
        img, detections = detect_image(model, image_path)
        
        if img is None:
            current_index += 1
            continue
        
        # Tampilkan info deteksi
        print(f"  Ditemukan {len(detections)} objek:")
        for det in detections:
            print(f"    - {det['class_name']}: {det['confidence']:.2%}")
        
        # Gambar deteksi
        result_img = draw_detections(img, detections)
        
        # Tambah info pada gambar
        info_text = f"Image {current_index + 1}/{len(image_files)}: {image_path.name}"
        cv2.putText(
            result_img,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Tampilkan pop-up
        show_popup(result_img)
        
        # Tunggu input keyboard
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q') or key == 27:  # q atau ESC untuk keluar
            print("\nKeluar dari program...")
            break
        elif key == ord('s'):  # Save gambar
            save_path = BASE_DIR / "results" / f"detected_{image_path.name}"
            save_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(save_path), result_img)
            print(f"  Gambar tersimpan: {save_path}")
        elif key == 32:  # Space untuk gambar berikutnya
            current_index += 1
        elif key == ord('p'):  # Previous
            if current_index > 0:
                current_index -= 1
        else:
            # Default: lanjut ke gambar berikutnya
            current_index += 1
    
    cv2.destroyAllWindows()


def main():
    """Main function"""
    print("=" * 60)
    print("Fruit Detection - YOLOv8 Inference")
    print("=" * 60)
    
    # Load model
    model = load_model()
    
    # Cek apakah IMAGE_PATH adalah file atau folder
    image_path = Path(IMAGE_PATH)
    
    if image_path.is_file():
        # Proses satu gambar
        print(f"\nMode: Single Image")
        process_single_image(model, image_path)
        
        # Tunggu input keyboard
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            result_img = draw_results(model, image_path)
            save_path = BASE_DIR / "results" / f"detected_{image_path.name}"
            save_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(save_path), result_img)
            print(f"Gambar tersimpan: {save_path}")
    
    elif image_path.is_dir():
        # Proses folder gambar
        print(f"\nMode: Folder Processing")
        process_folder(model, image_path)
    
    else:
        print(f"[ERROR] Path tidak valid: {image_path}")
        print("        Pastikan path gambar atau folder benar.")
        sys.exit(1)
    
    cv2.destroyAllWindows()
    print("\nProgram selesai!")


if __name__ == '__main__':
    main()

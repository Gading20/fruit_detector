# Fruit Detection AI - YOLOv8

Sistem Object Detection berbasis Artificial Intelligence untuk mendeteksi dan mengklasifikasikan gambar buah menggunakan YOLOv8.

## Deskripsi

Projeck ini merupakan solusi AI untuk tugas pertama yaitu membangun sistem Object Detection yang mampu mendeteksi 9 jenis buah:
- Apple (Apel)
- Banana (Pisang)
- Grapes (Anggur)
- Kiwi
- Mango (Mangga)
- Orange (Jeruk)
- Pineapple (Nanas)
- Sugerapple (Srikaya)
- Watermelon (Semangka)

## Performa Model

| Metric | Nilai |
|--------|-------|
| mAP50 | 98.47% |
| mAP50-95 | 97.32% |
| Training Epochs | 8/20 |
| Model | YOLOv8n (Nano) |

## Struktur Folder

```
fruit-detector-yolo/
├── detect.py              # Script inference dengan pop-up window
├── train.py               # Script training model
├── prepare_dataset.py     # Script konversi dataset ke format YOLO
├── yolov8n.pt             # Pretrained model YOLOv8 nano
├── runs/                  # Hasil training (weights, logs)
├── dataset/               # Dataset asli dari Kaggle
├── yolo_dataset/          # Dataset dalam format YOLO
├── results/               # Hasil deteksi gambar
└── .gitignore
```

## File yang Dikirimkan

### 1. Model/Weights
- **Lokasi**: `runs/detect/fruit_detection/weights/best.pt`
- **Ukuran**: ~23MB
- **Format**: PyTorch (.pt)

### 2. Script Training
- **File**: `train.py`
- **Library**: Ultralytics YOLO
- **Parameter**:
  - Epochs: 20 (berhenti di epoch 8 karena sudah konvergen)
  - Image Size: 416px
  - Batch Size: 8
  - Optimizer: AdamW

### 3. Script Inference
- **File**: `detect.py`
- **Fitur**:
  - Pop-up window untuk preview hasil deteksi
  - Bounding box dengan label kelas
  - Confidence score
  - Support single image dan folder mode
  - Keyboard controls (q/ESC untuk keluar, SPACE untuk next, s untuk save)

## Cara Penggunaan

### Prerequisites
```bash
pip install ultralytics opencv-python
```

### Training Model
```bash
python train.py
```

### Menjalankan Inference
```bash
python detect.py
```

### Konfigurasi
Edit bagian `KONFIGURASI` di `detect.py`:
```python
IMAGE_PATH = BASE_DIR / "yolo_dataset" / "images" / "test"  # Path gambar
CONFIDENCE_THRESHOLD = 0.25  # Confidence minimum
```

## Dataset

Dataset diambil dari Kaggle:
- **Nama**: Fruits by YOLO - Fruits Detection
- **Link**: https://www.kaggle.com/datasets/kapturovalexander/fruits-by-yolo-fruits-detection
- **Jumlah**: ~2,966 gambar (train: 2691, val: 186, test: 89)

## Teknologi yang Digunakan

- **Python 3.12**
- **Ultralytics YOLOv8** - Library Object Detection
- **OpenCV** - Image processing dan display
- **PyTorch** - Deep Learning framework

## Hasil Deteksi

Contoh hasil deteksi pada gambar test:
- Pineapple: 90.61%
- Watermelon: 79.85%
- Kiwi: 96.90%
- Banana: 98.08%
- Mango: 95.14%

## License

Project ini dibuat untuk tugas test kerja.

## Author

Dibuat dengan menggunakan YOLOv8 (Ultralytics)

# Bangladeshi Currency Detection with YOLOv8 🇧🇩

This project implements a computer vision system to detect and classify Bangladeshi banknotes in real-time using the YOLOv8 (Nano) algorithm. It is optimized for Apple Silicon (M-series) GPUs using MPS (Metal Performance Shaders).

## 🚀 Features

- **Real-time Detection:** High-speed inference using YOLOv8n.
- **9 Currency Classes:** Supports 2, 5, 10, 20, 50, 100, 200, 500, and 1000 Taka notes.
- **Apple Silicon Support:** Accelerated training and inference on Mac M-Series chips.

## 🛠 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/bd-currency-detect.git
   cd bd-currency-detect
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install ultralytics opencv-python
   ```

## 📂 Project Structure

```
bd-currency-detect/
├── datasets/            # Dataset configuration and images
│   ├── data.yaml        # YOLOv8 dataset config
│   ├── train/           # Training images/labels
│   └── valid/           # Validation images/labels
├── runs/                # Training results (weights, graphs)
├── check_gpu.py         # Utility to check MPS availability
├── test_live.py         # Real-time detection script using Webcam
├── train_model.py       # Model training script
└── yolov8n.pt           # Pre-trained YOLOv8 Base Model
```

## 🏃 Usage

### 1. Check GPU Availability
Verify if your Mac's GPU is detected for acceleration.
```bash
python check_gpu.py
```

### 2. Train the Model
Train the YOLOv8 model on the dataset.
```bash
python train_model.py
```
- **Epochs:** 50
- **Image Size:** 640
- **Batch Size:** 16
- **Output:** Trained weights will be saved in `runs/detect/bd_currency_result/weights/best.pt`.

### 3. Live Webcam Detection
Run real-time inference using your webcam.
```bash
python test_live.py
```
- Press **'q'** to quit the webcam feed.

## 📊 Dataset Classes

The model is trained on the following classes:
- 1000 Taka
- 500 Taka
- 200 Taka
- 100 Taka
- 50 Taka
- 20 Taka
- 10 Taka
- 5 Taka
- 2 Taka

## ⚠️ Configuration Note
The dataset configuration file is located at `datasets/data.yaml`. If you move the project, you might need to update the `path` variable inside this file to match your new absolute path.

---
**Author:** Zikrul Bari Tomal, Jannatul Ferdaus, Ashikur Rahman.

import os
# Library conflict fix for Mac
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from ultralytics import YOLO

def train_currency_detector():
    # 1. Load Nano model
    model = YOLO('yolov8n.pt') 

    print("🚀 Starting training for 9 classes on M4 Pro...")
    
    # 2. Train
    results = model.train(
        data='datasets/data.yaml',  # <--- Change: Path ta ekhon datasets folder er vitore
        epochs=50,                  # 50 bar shob chobi dekhbe
        imgsz=640,
        device='mps',               # Mac GPU acceleration
        batch=16,
        name='bd_currency_result',  # Output folder name
        exist_ok=True
    )

    print("✅ Training Complete!")

if __name__ == '__main__':
    train_currency_detector()
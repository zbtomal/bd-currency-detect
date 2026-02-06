from ultralytics import YOLO
import cv2
import math

def start_webcam():
    # 1. Load the trained model
    # Tomar training log onujayi model path thik kore dilam
    model_path = 'runs/detect/bd_currency_result/weights/best.pt'
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error: Model file pacche na! Path ta check koro: {model_path}")
        return

    # 2. Start Webcam
    cap = cv2.VideoCapture(0) # 0 = Built-in Camera
    
    # Resolution ektu komalam jate fast chole
    cap.set(3, 640) # Width
    cap.set(4, 480) # Height

    print("🎥 Webcam started... Taka dhore dekho! (Press 'q' to quit)")

    while True:
        success, img = cap.read()
        if not success:
            print("❌ Camera read failed.")
            break

        # 3. Predict using the model
        # conf=0.6 dilam jate 60% sure na hole box na aake (bhul kom hobe)
        results = model(img, stream=True, conf=0.6) 

        # 4. Draw boxes
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Class & Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = model.names[cls]

                # Visuals
                if conf > 0.6: # Double check
                    # Green Box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Label Background (Black)
                    cv2.rectangle(img, (x1, y1 - 25), (x1 + 150, y1), (0, 0, 0), -1)
                    
                    # Text (White)
                    cv2.putText(img, f'{class_name} {conf}', (x1 + 5, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 5. Show the video
        cv2.imshow('BD Currency Detector - M4 Pro', img)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_webcam()
from ultralytics import YOLO
import cv2
import numpy as np
import time

# Load YOLO model
model = YOLO("yolov8n.pt")
# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera.")
    exit()
prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break
    # Resize frame
    frame = cv2.resize(frame, (800, 600))
    # --------------------------------
    # YOLO Detection
    # --------------------------------
    results = model(frame, verbose=False)
    annotated = frame.copy()
    warning_triggered = False
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        # Bounding box area
        area = (x2 - x1) * (y2 - y1)
        # Default green box
        color = (0, 255, 0)
        # Trigger warning if close
        if area > 50000:
            color = (0, 0, 255)
            warning_triggered = True
        # Draw box
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            3
        )
        # Label
        cv2.putText(
            annotated,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
    # --------------------------------
    # Parking Guidelines
    # --------------------------------
    # Left guideline
    cv2.line(
        annotated,
        (250, 600),
        (350, 350),
        (0, 255, 255),
        4
    )
    # Right guideline
    cv2.line(
        annotated,
        (550, 600),
        (450, 350),
        (0, 255, 255),
        4
    )
    # Center line
    cv2.line(
        annotated,
        (400, 600),
        (400, 300),
        (255, 255, 255),
        2
    )
    # Distance zones
    cv2.line(
        annotated,
        (300, 500),
        (500, 500),
        (0, 255, 0),
        3
    )
    cv2.line(
        annotated,
        (330, 430),
        (470, 430),
        (0, 255, 255),
        3
    )
    cv2.line(
        annotated,
        (360, 370),
        (440, 370),
        (0, 0, 255),
        3
    )
    # --------------------------------
    # FPS Counter
    # --------------------------------
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(
        annotated,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )
    # --------------------------------
    # Collision Warning Banner
    # --------------------------------
    if warning_triggered:
        cv2.rectangle(
            annotated,
            (0, 0),
            (800, 80),
            (0, 0, 255),
            -1
        )
        cv2.putText(
            annotated,
            "COLLISION WARNING",
            (180, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            4
        )
    # --------------------------------
    # Display
    # --------------------------------
    cv2.imshow(
        "AI Parking Assist System",
        annotated
    )
    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
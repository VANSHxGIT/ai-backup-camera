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
# Perspective transform points
src_points = np.float32([
    [80, 120],
    [220, 120],
    [290, 200],
    [10, 200]
])
dst_points = np.float32([
    [50, 0],
    [250, 0],
    [250, 300],
    [50, 300]
])
# Compute transform matrix ONCE
matrix = cv2.getPerspectiveTransform(
    src_points,
    dst_points
)
prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break
    # Resize smaller for performance
    frame = cv2.resize(frame, (300, 200))
    # -----------------------------
    # YOLO Inference
    # -----------------------------
    results = model(
        frame,
        verbose=False
    )
    detected = results[0].plot()
    # -----------------------------
    # Perspective Warp
    # -----------------------------
    warped = cv2.warpPerspective(
        detected,
        matrix,
        (300, 300)
    )
    # Simulated camera directions
    front = warped.copy()
    rear = cv2.rotate(warped, cv2.ROTATE_180)
    left = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    right = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    # -----------------------------
    # Create canvas
    # -----------------------------
    canvas = np.zeros((800, 800, 3), dtype=np.uint8)
    canvas[0:300, 250:550] = front
    canvas[500:800, 250:550] = rear
    left_resized = cv2.resize(left, (250, 300))
    right_resized = cv2.resize(right, (250, 300))
    canvas[250:550, 0:250] = left_resized
    canvas[250:550, 550:800] = right_resized
    # Draw car placeholder
    cv2.rectangle(
        canvas,
        (300, 300),
        (500, 500),
        (255, 255, 255),
        -1
    )
    cv2.putText(
        canvas,
        "CAR",
        (350, 420),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 0),
        3
    )
    # -----------------------------
    # FPS Calculation
    # -----------------------------
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(
        canvas,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    # -----------------------------
    # Display
    # -----------------------------
    cv2.imshow(
        "Optimized AI Surround View",
        canvas
    )
    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
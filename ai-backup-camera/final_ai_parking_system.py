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

# Compute perspective matrix
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

    # Resize input
    frame = cv2.resize(frame, (300, 200))

    # --------------------------------
    # YOLO Detection
    # --------------------------------

    results = model(frame, verbose=False)

    detected = frame.copy()

    warning_triggered = False

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        conf = float(box.conf[0])

        cls = int(box.cls[0])

        label = model.names[cls]

        # Approximate distance using area
        area = (x2 - x1) * (y2 - y1)

        color = (0, 255, 0)

        if area > 15000:
            color = (0, 0, 255)
            warning_triggered = True

        # Draw bounding box
        cv2.rectangle(
            detected,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Label
        cv2.putText(
            detected,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # --------------------------------
    # Bird-Eye Warp
    # --------------------------------

    bird_eye = cv2.warpPerspective(
        detected,
        matrix,
        (300, 300)
    )

    # --------------------------------
    # Parking Guidelines
    # --------------------------------

    # Left guide line
    cv2.line(
        bird_eye,
        (90, 300),
        (130, 150),
        (0, 255, 255),
        3
    )

    # Right guide line
    cv2.line(
        bird_eye,
        (210, 300),
        (170, 150),
        (0, 255, 255),
        3
    )

    # Center line
    cv2.line(
        bird_eye,
        (150, 300),
        (150, 100),
        (255, 255, 255),
        2
    )

    # Distance zones
    cv2.line(
        bird_eye,
        (100, 250),
        (200, 250),
        (0, 255, 0),
        3
    )

    cv2.line(
        bird_eye,
        (110, 200),
        (190, 200),
        (0, 255, 255),
        3
    )

    cv2.line(
        bird_eye,
        (120, 150),
        (180, 150),
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
        bird_eye,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # --------------------------------
    # Collision Warning
    # --------------------------------

    if warning_triggered:

        cv2.rectangle(
            bird_eye,
            (0, 0),
            (300, 50),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            bird_eye,
            "COLLISION WARNING",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # --------------------------------
    # Draw source points on original
    # --------------------------------

    for point in src_points:

        cv2.circle(
            frame,
            tuple(point.astype(int)),
            5,
            (0, 0, 255),
            -1
        )

    # --------------------------------
    # Display Windows
    # --------------------------------

    cv2.imshow(
        "Original Camera",
        frame
    )

    cv2.imshow(
        "YOLO Detection",
        detected
    )

    cv2.imshow(
        "Bird-Eye Parking Assist",
        bird_eye
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
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

# Perspective points
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

matrix = cv2.getPerspectiveTransform(
    src_points,
    dst_points
)

# Steering angle
steering = 0

prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    frame = cv2.resize(frame, (400, 300))

    # -----------------------------------
    # YOLO Detection
    # -----------------------------------

    results = model(frame, verbose=False)

    detected = results[0].plot()

    # -----------------------------------
    # Bird-Eye Transform
    # -----------------------------------

    bird_eye = cv2.warpPerspective(
        detected,
        matrix,
        (400, 400)
    )

    # -----------------------------------
    # Dynamic Parking Guidelines
    # -----------------------------------

    center_x = 200

    curve_strength = steering * 2

    # Left curve
    left_points = []

    for y in range(400, 150, -20):

        offset = int(
            curve_strength * ((400 - y) / 250) ** 2
        )

        x = 120 + offset

        left_points.append((x, y))

    # Right curve
    right_points = []

    for y in range(400, 150, -20):

        offset = int(
            curve_strength * ((400 - y) / 250) ** 2
        )

        x = 280 + offset

        right_points.append((x, y))

    # Draw curves
    for i in range(len(left_points) - 1):

        cv2.line(
            bird_eye,
            left_points[i],
            left_points[i + 1],
            (0, 255, 255),
            3
        )

        cv2.line(
            bird_eye,
            right_points[i],
            right_points[i + 1],
            (0, 255, 255),
            3
        )

    # Center line
    cv2.line(
        bird_eye,
        (200, 400),
        (200 + curve_strength, 150),
        (255, 255, 255),
        2
    )

    # -----------------------------------
    # Steering Display
    # -----------------------------------

    cv2.putText(
        bird_eye,
        f"Steering: {steering}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # -----------------------------------
    # FPS
    # -----------------------------------

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(
        bird_eye,
        f"FPS: {int(fps)}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # -----------------------------------
    # Display
    # -----------------------------------

    cv2.imshow(
        "Dynamic AI Parking Assist",
        bird_eye
    )

    # -----------------------------------
    # Keyboard Controls
    # -----------------------------------

    key = cv2.waitKey(1) & 0xFF

    # Left steering
    if key == ord('a'):
        steering -= 5

    # Right steering
    elif key == ord('d'):
        steering += 5

    # Reset steering
    elif key == ord('s'):
        steering = 0

    # Quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
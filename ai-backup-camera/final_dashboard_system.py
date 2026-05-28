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

# Dashboard output size
dashboard_width = 1200
dashboard_height = 700

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    "final_output.mp4",
    fourcc,
    20.0,
    (dashboard_width, dashboard_height)
)

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

    detected = frame.copy()

    warning_triggered = False

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        conf = float(box.conf[0])

        cls = int(box.cls[0])

        label = model.names[cls]

        area = (x2 - x1) * (y2 - y1)

        color = (0, 255, 0)

        if area > 25000:
            color = (0, 0, 255)
            warning_triggered = True

        cv2.rectangle(
            detected,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            detected,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # -----------------------------------
    # Bird-Eye Transformation
    # -----------------------------------

    bird_eye = cv2.warpPerspective(
        detected,
        matrix,
        (400, 400)
    )

    # -----------------------------------
    # Parking Guidelines
    # -----------------------------------

    cv2.line(
        bird_eye,
        (120, 400),
        (170, 180),
        (0, 255, 255),
        3
    )

    cv2.line(
        bird_eye,
        (280, 400),
        (230, 180),
        (0, 255, 255),
        3
    )

    cv2.line(
        bird_eye,
        (200, 400),
        (200, 150),
        (255, 255, 255),
        2
    )

    # Distance zones
    cv2.line(
        bird_eye,
        (120, 330),
        (280, 330),
        (0, 255, 0),
        3
    )

    cv2.line(
        bird_eye,
        (140, 260),
        (260, 260),
        (0, 255, 255),
        3
    )

    cv2.line(
        bird_eye,
        (160, 200),
        (240, 200),
        (0, 0, 255),
        3
    )

    # -----------------------------------
    # FPS
    # -----------------------------------

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    # -----------------------------------
    # Create Dashboard
    # -----------------------------------

    dashboard = np.zeros(
        (dashboard_height, dashboard_width, 3),
        dtype=np.uint8
    )

    # Titles
    cv2.putText(
        dashboard,
        "ORIGINAL FEED",
        (80, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        dashboard,
        "YOLO DETECTION",
        (470, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        dashboard,
        "BIRD-EYE PARKING ASSIST",
        (780, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Place views
    dashboard[80:380, 50:450] = frame

    dashboard[80:380, 430:830] = detected

    dashboard[80:480, 780:1180] = bird_eye

    # -----------------------------------
    # System Info Panel
    # -----------------------------------

    cv2.putText(
        dashboard,
        f"FPS: {int(fps)}",
        (50, 600),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        dashboard,
        "SYSTEM STATUS: ACTIVE",
        (300, 600),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    # Collision warning
    if warning_triggered:

        cv2.rectangle(
            dashboard,
            (0, 640),
            (1200, 700),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            dashboard,
            "COLLISION WARNING DETECTED",
            (300, 680),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3
        )

    # -----------------------------------
    # Recording
    # -----------------------------------

    out.write(dashboard)

    # -----------------------------------
    # Display
    # -----------------------------------

    cv2.imshow(
        "FINAL AI PARKING DASHBOARD",
        dashboard
    )

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

out.release()

cv2.destroyAllWindows()
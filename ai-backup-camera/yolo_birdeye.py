from ultralytics import YOLO
import cv2
import numpy as np
# Load YOLO model
model = YOLO("yolov8n.pt")
# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

# -------------------------------
# Perspective Transform Setup
# -------------------------------
src_points = np.float32([
    [300, 400],
    [500, 400],
    [700, 550],
    [100, 550]
])
dst_points = np.float32([
    [200, 0],
    [600, 0],
    [600, 600],
    [200, 600]
])

# Compute transformation matrix
matrix = cv2.getPerspectiveTransform(
    src_points,
    dst_points
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break
    # Resize frame
    frame = cv2.resize(frame, (800, 600))
    # -------------------------------
    # YOLO Object Detection
    # -------------------------------
    results = model(frame)
    # Draw detection boxes
    detected_frame = results[0].plot()

    # Apply bird-eye warp
    bird_eye = cv2.warpPerspective(
        detected_frame,
        matrix,
        (800, 600)
    )
    # Draw source points
    for point in src_points:
        cv2.circle(
            detected_frame,
            tuple(point.astype(int)),
            8,
            (0, 0, 255),
            -1
        )
    # Display windows
    cv2.imshow("YOLO Detection", detected_frame)
    cv2.imshow("Bird-Eye AI View", bird_eye)
    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Cleanup
cap.release()
cv2.destroyAllWindows()
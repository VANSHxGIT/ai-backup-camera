from ultralytics import YOLO
import cv2
import numpy as np
from tracker import ObjectTracker
tracker = ObjectTracker()
# Load YOLO model
model = YOLO("yolov8n.pt")
# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera.")
    exit()
# Perspective source points
src_points = np.float32([
    [80, 120],
    [220, 120],
    [290, 200],
    [10, 200]
])
# Destination points
dst_points = np.float32([
    [50, 0],
    [250, 0],
    [250, 300],
    [50, 300]
])
# Compute homography matrix
matrix = cv2.getPerspectiveTransform(
    src_points,
    dst_points
)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break
    # Resize input frame
    frame = cv2.resize(frame, (300, 200))
    # ---------------------------------
    # YOLO Detection
    # ---------------------------------
    results = model(frame)
    detected = results[0].plot()
    # ---------------------------------
    # Perspective Warp
    # ---------------------------------
    warped = cv2.warpPerspective(
        detected,
        matrix,
        (300, 300)
    )
    # Simulated camera directions
    front = warped.copy()
    rear = cv2.rotate(
        warped,
        cv2.ROTATE_180
    )
    left = cv2.rotate(
        warped,
        cv2.ROTATE_90_COUNTERCLOCKWISE
    )
    right = cv2.rotate(
        warped,
        cv2.ROTATE_90_CLOCKWISE
    )
    # ---------------------------------
    # Create surround-view canvas
    # ---------------------------------
    canvas = np.zeros((800, 800, 3), dtype=np.uint8)
    # Front
    canvas[0:300, 250:550] = front
    # Rear
    canvas[500:800, 250:550] = rear
    # Left
    left_resized = cv2.resize(left, (250, 300))
    canvas[250:550, 0:250] = left_resized
    # Right
    right_resized = cv2.resize(right, (250, 300))
    canvas[250:550, 550:800] = right_resized
    # ---------------------------------
    # Draw center vehicle
    # ---------------------------------
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
    # ---------------------------------
    # Draw source points
    # ---------------------------------
    for point in src_points:
        cv2.circle(
            detected,
            tuple(point.astype(int)),
            5,
            (0, 0, 255),
            -1
        )
    # ---------------------------------
    # Display windows
    # ---------------------------------
    cv2.imshow("Detected Feed", detected)
    cv2.imshow("Warped Feed", warped)
    cv2.imshow("True Bird-Eye Surround View", canvas)
    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
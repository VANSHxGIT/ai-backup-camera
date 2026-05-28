import cv2
import numpy as np
# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break
    # Resize frame
    frame = cv2.resize(frame, (300, 200))
    # Simulate multiple camera feeds
    front = frame.copy()
    rear = cv2.rotate(
        frame,
        cv2.ROTATE_180
    )
    left = cv2.rotate(
        frame,
        cv2.ROTATE_90_COUNTERCLOCKWISE
    )
    right = cv2.rotate(
        frame,
        cv2.ROTATE_90_CLOCKWISE
    )
    # Create empty canvas
    canvas = np.zeros((700, 700, 3), dtype=np.uint8)
    # Place feeds on canvas
    # Front
    canvas[0:200, 200:500] = front
    # Rear
    canvas[500:700, 200:500] = rear
    # Left
    left_resized = cv2.resize(left, (200, 300))
    canvas[200:500, 0:200] = left_resized
    # Right
    right_resized = cv2.resize(right, (200, 300))
    canvas[200:500, 500:700] = right_resized
    # Draw center car placeholder
    cv2.rectangle(
        canvas,
        (250, 250),
        (450, 450),
        (255, 255, 255),
        -1
    )
    cv2.putText(
        canvas,
        "CAR",
        (310, 360),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 0),
        3
    )
    # Show output
    cv2.imshow(
        "Surround View System",
        canvas
    )
    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera.")
    exit()

# Define source points
src_points = np.float32([
    [300, 400],
    [500, 400],
    [700, 550],
    [100, 550]
])
# Define destination points
dst_points = np.float32([
    [200, 0],
    [600, 0],
    [600, 600],
    [200, 600]
])
# Compute transformation matrix
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    # Resize frame
    frame = cv2.resize(frame, (800, 600))
    # Apply perspective warp
    bird_eye = cv2.warpPerspective(frame, matrix, (800, 600))
    # Draw red points on original frame
    for point in src_points:
        cv2.circle(
            frame,
            tuple(point.astype(int)),
            8,
            (0, 0, 255),
            -1
        )
    # Display windows
    cv2.imshow("Original Camera Feed", frame)
    cv2.imshow("Bird-Eye View", bird_eye)
    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Cleanup
cap.release()
cv2.destroyAllWindows()
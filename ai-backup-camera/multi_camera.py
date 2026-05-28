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
    frame = cv2.resize(frame, (640, 480))
    # Duplicate frame
    frame1 = frame.copy()
    frame2 = frame.copy()
    # Combine horizontally
    combined = np.hstack((frame1, frame2))
    # Display
    cv2.imshow("Multi Camera View", combined)
    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
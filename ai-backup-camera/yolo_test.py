from ultralytics import YOLO
import cv2
# Load YOLO model
model = YOLO("yolov8n.pt")
# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break
    # Run YOLO detection
    results = model(frame)
    # Draw detections
    annotated_frame = results[0].plot()
    # Show output
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
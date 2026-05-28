from ultralytics import YOLO
import cv2
import time

# Load OpenVINO model
model = YOLO("yolov8n_openvino_model")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera.")
    exit()

prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    # Resize frame
    frame = cv2.resize(frame, (640, 480))

    # -----------------------------------
    # OpenVINO Inference
    # -----------------------------------

    results = model(
        frame,
        verbose=False
    )

    annotated = results[0].plot()

    # -----------------------------------
    # FPS Counter
    # -----------------------------------

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(
        annotated,
        f"OpenVINO FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # -----------------------------------
    # Display
    # -----------------------------------

    cv2.imshow(
        "OpenVINO Optimized Runtime",
        annotated
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
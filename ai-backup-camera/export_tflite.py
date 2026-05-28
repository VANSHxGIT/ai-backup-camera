from ultralytics import YOLO
# Load YOLO model
model = YOLO("yolov8n.pt")
# Export to TensorFlow Lite
model.export(format="tflite")
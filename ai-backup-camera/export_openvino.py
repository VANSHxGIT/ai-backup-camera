from ultralytics import YOLO
# Load YOLO model
model = YOLO("yolov8n.pt")
# Export to OpenVINO
model.export(format="openvino")
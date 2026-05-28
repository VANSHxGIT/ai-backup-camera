# AI-Powered Bird-Eye Parking Assistance System

An advanced AI-based parking assistance and collision warning system using YOLOv8, Bird-Eye View Transformation, OpenCV, and OpenVINO optimization.



# Features:

- Real-time object detection using YOLOv8
- Bird-eye perspective transformation
- Collision warning system
- Dynamic parking guidelines
- Parking assist overlays
- FPS monitoring
- OpenVINO optimized inference
- Video recording dashboard
- Real-time webcam processing


# Technologies Used

- Python
- OpenCV
- YOLOv8
- Ultralytics
- NumPy
- OpenVINO
- ONNX Runtime



# System Pipeline

Camera Feed
↓
YOLOv8 Object Detection
↓
Bird-Eye Perspective Transformation
↓
Collision Detection
↓
Parking Assist Visualization
↓
Dynamic Steering Guidelines
↓
Dashboard Output


# Project Structure

```bash
ai-backup-camera/
│
├── final_dashboard_system.py
├── final_ai_parking_system.py
├── dynamic_parking_guidelines.py
├── collision_warning.py
├── birdseye.py
├── yolo_detection.py
├── surround_view.py
├── export_openvino.py
├── openvino_runtime.py
│
├── assets/
├── outputs/
├── README.md
├── requirements.txt
└── .gitignore
```

# Installation

## Clone Repository

```bash
git clone https://github.com/VANSHxGIT/ai-backup-camera.git
```

## Move Into Project

```bash
cd ai-backup-camera
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running The Project

## Main Dashboard System

```bash
python final_dashboard_system.py
```

## Dynamic Parking Assist

```bash
python dynamic_parking_guidelines.py
```

## OpenVINO Runtime

```bash
python openvino_runtime.py
```



# Future Improvements

- Real multi-camera calibration
- Embedded deployment on Jetson/Raspberry Pi
- CAN bus integration
- Real steering wheel data
- Sensor fusion
- Lane detection
- Depth estimation
- Autonomous parking assistance


# Author

Vansh Rawat

---

# License

MIT License
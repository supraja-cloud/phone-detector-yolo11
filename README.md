# 📱 Phone Detector YOLOv11 (`phone-detector-yolo11`)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-orange.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)

An ultra-fast, real-time phone and mobile device detection system powered by **YOLOv11** and **OpenCV** with multithreaded asynchronous inference.

Designed for exam proctoring, distraction-free study monitoring, security surveillance, and driver distraction prevention.

---

## ✨ Features

- **⚡ Asynchronous Multithreading**: Decouples webcam frame capture from YOLO neural network inference so your camera feed stays smooth at full FPS without stutter.
- **🧠 YOLOv11 Powered**: Leverages state-of-the-art YOLOv11 architecture for superior detection accuracy and speed.
- **🚀 Automatic GPU Acceleration**: Automatically leverages NVIDIA CUDA with FP16 half-precision when available, with seamless CPU fallback.
- **🎯 False Positive Suppression**: Includes a consecutive detection counter (`hits >= 2`) before triggering high-priority visual alerts, eliminating single-frame false alarms.
- **📊 Real-Time Metrics**: Displays live inference FPS, bounding boxes, detection confidence percentages, and alert overlays.

---

## 🛠️ Project Structure

```text
phone-detector-yolo11/
├── phone_test.py       # Main real-time detection script
├── requirements.txt    # Python dependencies
├── .gitignore          # Git exclusion rules
├── LICENSE             # MIT License
└── README.md           # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository

```bash
git clone https://github.com/supraja-cloud/phone-detector-yolo11.git
cd phone-detector-yolo11
```

### 2. Create and Activate a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note for NVIDIA GPU users**: For CUDA acceleration, make sure you have PyTorch installed with CUDA support. Visit [pytorch.org](https://pytorch.org/get-started/locally/) to install the appropriate CUDA build for your system.

### 4. Run the Application

```bash
python phone_test.py
```

Press **`q`** on your keyboard at any time to exit the camera window.

---

## ⚙️ Configuration & Customization

You can easily adjust the settings directly inside [phone_test.py](phone_test.py):

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `MODEL` | `"yolo11s.pt"` | Choose between `yolo11n.pt` (fastest), `yolo11s.pt` (balanced), or `yolo11m.pt` (highest accuracy). |
| `CONF` | `0.20` | Detection confidence threshold (0.0 to 1.0). |
| `IMGSZ` | `640` | Input image resolution. Smaller (e.g., 480) increases speed; larger (800+) improves small phone detection. |
| `CONSECUTIVE` | `2` | Number of consecutive positive detections required before triggering the `"PHONE DETECTED!"` alert banner. |

---

## 👨‍💻 Author

Created and maintained by **[@supraja-cloud](https://github.com/supraja-cloud)**.

Feel free to connect or contribute!

---

## 📄 License

This project is open-source and licensed under the [MIT License](LICENSE).

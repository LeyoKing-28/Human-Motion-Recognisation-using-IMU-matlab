# ⚡ Human Activity Recognition & Bio-Dynamic Cardiac Monitor (1D-CNN)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-Deep%20Learning%20Toolbox-orange.svg?logo=mathworks)](https://www.mathworks.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

An end-to-end Deep Learning system that recognizes human activities from wearable Inertial Measurement Unit (IMU) sensors and estimates real-time cardiovascular strain using a 1D Convolutional Neural Network (1D-CNN).

Includes a live streaming IoT server connecting smartphone sensors (Accelerometer + Gyroscope via local Wi-Fi) to an interactive, real-time web dashboard.

---

## 🌟 Key Features

* **1D-CNN Architecture**: Optimized for multi-channel temporal sensor streams with 1D convolutions, batch normalization, and dropout.
* **Dual Implementation**: Fully implemented in both **PyTorch** and **MATLAB (Deep Learning Toolbox)**.
* **Benchmark Training**: Trained on the **HARTH dataset** across 12 distinct physical activities (Walking, Running, Standing, Sitting, Lying, Cycling, Stairs).
* **Real-Time Edge Streaming**: Connects an Android phone (using **Sensor Logger**) over Wi-Fi to a multi-threaded Python server.
* **Bio-Dynamic Heart Rate Engine**: Estimates real-time heart rate (BPM), cardiac zones, and calorie burn rates based on Tanaka, Karvonen, and Keytel clinical equations.
* **Interactive Web Dashboard**: Glassmorphic dark-mode interface featuring dynamic activity badges, class probabilities, and real-time dual oscilloscopes.

---

## 🏗️ Project Architecture

```
[Android Smartphone (IMU)]
  └── Accelerometer + Gyroscope (50 Hz)
            │  HTTP Push / Wi-Fi
            ▼
[Python Real-Time Server (realtime_server.py)]
  ├── Ingestion & Orientation Remapping
  ├── Rolling FIFO Buffer (128 samples / 2.56s)
  ├── 1D-CNN Inference (PyTorch)
  └── Bio-Dynamic Heart Rate & Calorie Engine
            │
            ├── Serves dashboard.html
            └── Telemetry API (/api/telemetry)
            │
            ▼
[Web Browser Dashboard (localhost:8000)]
  ├── Live Activity Recognition & Confidence
  ├── Animated Beating Heart & BPM Gauge
  ├── Physiological Telemetry (METs, Calories, BMI)
  └── Dual Real-Time Oscilloscopes (Accel & Gyro)
```

---

## 📁 Repository Structure

```
.
├── model_pytorch.py        # PyTorch 1D-CNN model architecture
├── train_pytorch.py        # Complete training pipeline & data windowing (PyTorch)
├── model_matlab.m          # MATLAB 1D-CNN layer definition
├── train_matlab.m          # MATLAB training script (trainNetwork)
├── realtime_server.py      # Real-time IoT server + Heart rate engine
├── dashboard.html          # Interactive dark-mode web dashboard
├── test_model.py           # Unit testing / offline inference script
├── best_model.pth          # Pre-trained 1D-CNN weights
├── PROJECT_EXPLANATION.md  # Detailed technical documentation & mentor defense guide
└── problem statement.txt   # Original project specification
```

---

## 🚀 Quick Start

### 1. Prerequisites
* Python 3.10+
* PyTorch, NumPy, Pandas, Scikit-learn

```bash
# Clone the repository
git clone https://github.com/LeyoKing-28/IMU-Sensor-HAR.git
cd IMU-Sensor-HAR

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch pandas numpy scikit-learn
```

### 2. Run the Real-Time System
```bash
python3 realtime_server.py
```
* Open your browser at: `http://localhost:8000/`
* In your phone's **Sensor Logger** app, enable **Accelerometer** and **Gyroscope** at 50 Hz, and set the Push URL to `http://<YOUR_PC_IP>:8000/`.

---

## 📊 Evaluation & Metrics

| Activity | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Lying** | **1.00** | **0.90** | **0.95** |
| **Running** | **0.70** | **0.93** | **0.80** |
| **Walking** | **0.65** | **0.53** | **0.58** |
| **Standing** | **0.19** | **0.89** | **0.32** |

*For complete architectural analysis and mentor viva questions, refer to [PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md).*

---

## 📜 License
This project is licensed under the MIT License.

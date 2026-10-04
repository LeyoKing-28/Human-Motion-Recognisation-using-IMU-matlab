# Human Activity Recognition (HAR) using 1D Convolutional Neural Networks
## Comprehensive Project Documentation & Mentor Defense Guide

---

## 1. Executive Summary & Project Motivation

### What is this project?
This project implements an end-to-end Deep Learning system for **Human Activity Recognition (HAR)** using **Inertial Measurement Unit (IMU)** time-series sensor data. 

We developed:
1. A **1D Convolutional Neural Network (1D-CNN)** implemented in both **PyTorch** and **MATLAB**.
2. A complete **data preprocessing and sliding-window pipeline** trained on the benchmark **HARTH dataset** (12 distinct physical activities).
3. A **real-time edge-streaming system** that connects an Android smartphone's internal IMU (Accelerometer + Gyroscope) over local Wi-Fi to a live neural network inference engine.
4. A **real-time interactive web dashboard** displaying dual sensor oscilloscopes, class probability distributions, and live activity classifications.

### Why IMU Sensors & Deep Learning?
* **Pervasive & Low-cost**: Accelerometers and gyroscopes are embedded in every modern smartphone, smartwatch, and wearable.
* **Privacy-preserving**: Unlike cameras, IMUs do not capture visual identities or sensitive surroundings, making them ideal for healthcare, elderly care, and round-the-clock monitoring.
* **Feature Automation**: Traditional machine learning (SVM, Random Forests) requires manual feature engineering (calculating mean, variance, spectral entropy, FFT peaks). A **1D-CNN automatically learns hierarchical temporal feature representations** directly from raw sensor signals.

---

## 2. Dataset Architecture: The HARTH Dataset

The model is trained on the **HARTH (Human Activity Recognition Trondheim)** benchmark dataset:

* **Hardware Used**: Two 3-axial Axivity AX3 accelerometers worn on:
  1. **Lower Back** (3rd lumbar vertebra)
  2. **Right Thigh** (approx. 10 cm above the kneecap)
* **Sampling Rate**: **50 Hz** (1 sample every 0.02 seconds / 20 ms).
* **Input Channels (6 features)**:
  `[back_x, back_y, back_z, thigh_x, thigh_y, thigh_z]` in units of $g$ ($1g \approx 9.80665 \text{ m/s}^2$).
* **Subjects**: 22 participants (`S006.csv` to `S029.csv`) performing unconstrained daily activities.
* **Classes (12 Categories)**:
  1. Walking (`1`)
  2. Running (`2`)
  3. Shuffling (`3`)
  4. Stairs (ascending) (`4`)
  5. Stairs (descending) (`5`)
  6. Standing (`6`)
  7. Sitting (`7`)
  8. Lying (`8`)
  9. Cycling (sitting) (`13`)
  10. Cycling (standing) (`14`)
  11. Cycling (sitting, inactive) (`130`)
  12. Cycling (standing, inactive) (`140`)

---

## 3. Data Preprocessing & Windowing Pipeline

### 3.1. Sliding Window Segmentation
Raw IMU sensor signals are continuous time-series streams. Neural networks require fixed-dimension tensors.
* **Window Size ($W$)**: **128 samples** $= 128 \times 0.02\text{s} = \mathbf{2.56 \text{ seconds}}$.
  * *Why 2.56 seconds?* A human gait cycle (two steps) typically takes between 1.0 and 1.5 seconds. A 2.56-second window guarantees capture of at least 1.5 to 2 full repetitive motion cycles.
* **Step Size / Overlap ($S$)**: **64 samples** (**50% overlap**).
  * 50% overlap ensures smooth transitions between adjacent temporal windows and prevents boundary information loss.
* **Window Label Assignment**: The activity label of the center sample ($t = 64$) is assigned as the label for that entire window.

### 3.2. Subject-Wise (Leave-Subjects-Out) Data Splitting
A critical engineering choice was made to split data **by subject ID**, not random row shuffling:
* **Train Set (15 subjects)**: `S006` through `S022` (81,600 windows)
* **Validation Set (3 subjects)**: `S023` through `S025` (8,431 windows)
* **Test Set (4 subjects)**: `S026` through `S029` (10,894 windows)

> **Mentor Tip**: Randomly shuffling windows before splitting creates severe *data leakage* (consecutive overlapping windows from the same person would be in both train and test sets, artificially inflating accuracy to ~99%). Splitting by subject tests **true generalization to unseen human bodies**.

### 3.3. Standardization (Z-score Normalization)
For each sensor channel $k$:
$$X_{\text{norm}}^{(k)} = \frac{X^{(k)} - \mu_{\text{train}}^{(k)}}{\sigma_{\text{train}}^{(k)}}$$
* $\mu$ (mean) and $\sigma$ (standard deviation) are computed **strictly from the training subjects** and then applied to validation, test, and real-time inference to prevent data contamination.

---

## 4. Deep Learning Model Architecture (1D-CNN)

The network is implemented in [`model_pytorch.py`](file:///home/leyoking/college/IMU%20SENSOR/model_pytorch.py) and [`model_matlab.m`](file:///home/leyoking/college/IMU%20SENSOR/model_matlab.m).

```
Input Window: (Batch Size, 6 Channels, 128 Time Steps)
   │
   ▼
[Conv1D Layer 1]: 64 filters, kernel size 3, stride 1, padding 1
   │ Output: (Batch, 64, 128)
   ▼
[ReLU Activation]
   │
   ▼
[Conv1D Layer 2]: 64 filters, kernel size 3, stride 1, padding 1
   │ Output: (Batch, 64, 128)
   ▼
[ReLU Activation]
   │
   ▼
[Dropout (p = 0.5)]: Prevents overfitting
   │
   ▼
[MaxPool1D]: kernel size 2, stride 2
   │ Output: (Batch, 64, 64)
   ▼
[Flatten]: 64 × 64 = 4,096 features
   │
   ▼
[Fully Connected Layer 1]: 4,096 -> 100 neurons
   │
   ▼
[ReLU Activation]
   │
   ▼
[Fully Connected Layer 2 (Output)]: 100 -> 12 class logits
   │
   ▼
[Softmax Function]: Multi-class probability distribution (sums to 1.0)
```

### Layer Rationale:
1. **1D Convolutions vs 2D Convolutions**: IMU signals have 1 temporal dimension ($T$) and multiple spatial channels ($C$). 1D convolutions slide kernels exclusively along the time axis, learning temporal patterns (e.g., foot strike impact, swing frequency).
2. **Double Conv Layers**: Stacking two 3-tap convolutions gives an effective receptive field of 5 time steps with fewer parameters than a single 5-tap filter.
3. **Dropout (0.5)**: Randomly zeroes 50% of feature activations during training, forcing redundant feature representation and preventing reliance on individual sensor spikes.
4. **Max Pooling**: Halves sequence length from 128 to 64, providing translation invariance (the exact millisecond a step starts does not alter classification).

---

## 5. Training Results & Performance Analysis

### 5.1. Summary Statistics
* **Training Accuracy**: **95.91%**
* **Validation Accuracy**: **55.20%**
* **Test Accuracy (Unseen Subjects)**: **51.09%**
* **Loss Function**: Categorical Cross-Entropy Loss
* **Optimizer**: Adam ($\text{learning rate} = 0.001$)

### 5.2. Class-by-Class Performance Breakdown
| Activity | Precision | Recall | F1-Score | Performance Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **Lying** | **1.00** | **0.90** | **0.95** | **Near Perfect**: Gravity vector shifts by ~90° onto the Z-axis. |
| **Running** | **0.70** | **0.93** | **0.80** | **High**: Large acceleration amplitude & high step frequency (~3 Hz). |
| **Walking** | **0.65** | **0.53** | **0.58** | **Solid**: Moderate periodic gait oscillation (~1.5–2 Hz). |
| **Stairs (Ascending)** | 0.28 | 0.59 | 0.38 | Moderate: Often confused with regular walking. |
| **Standing** | 0.19 | 0.89 | 0.32 | High recall: Thigh vertical with zero dynamic acceleration. |
| **Cycling / Sitting** | 0.00 | 0.00 | 0.00 | Confused with standing/walking due to cross-subject thigh angle variance. |

---

## 6. Real-Time Android Streaming & Web Dashboard

To transform this from an offline academic model into a working interactive system, we designed a live IoT pipeline:

```
[Android Smartphone]
  ├── Sensors: Accelerometer + Gyroscope (50 Hz)
  └── App: Sensor Logger (HTTP Push via Wi-Fi)
            │  POST / (JSON Payload)
            ▼
[Python Real-Time Server (realtime_server.py)]
  ├── 1. Ingestion & JSON Parsing (extracts ax, ay, az, gx, gy, gz)
  ├── 2. Orientation & Gravity Remapping (aligns phone axes to body coordinates)
  ├── 3. Rolling FIFO Buffer (stores latest 128 samples)
  ├── 4. Z-Score Standardization
  ├── 5. PyTorch Model Inference (outputs 12 probabilities)
  └── 6. Multi-threaded HTTP Server
            │
            ├── GET /             ──> Serves dashboard.html
            ├── GET /api/telemetry──> Returns JSON state (100ms interval)
            └── POST /            ──> Receives phone sensor stream
            │
            ▼
[Web Browser Dashboard (dashboard.html)]
  ├── Real-time Activity Card (Emoji + Confidence score)
  ├── 12-Class Live Probability Gauges
  ├── Accelerometer 3-Axis Oscilloscope Canvas (X, Y, Z in g)
  └── Gyroscope 3-Axis Oscilloscope Canvas (Pitch, Roll, Yaw in °/s)
```

---

## 7. Mentor / Viva Question & Answer Preparation

Here are the most likely questions your mentor or evaluation committee will ask, along with technical answers:

### Q1: Why did you choose a 1D-CNN instead of an LSTM or Recurrent Neural Network (RNN)?
> **Answer**: 
> "1D-CNNs are computationally superior for wearable edge deployment:
> 1. **Parallelism**: 1D-CNN convolutions process the entire 128-step window simultaneously via parallel matrix multiplication. In contrast, LSTMs must process time steps sequentially ($h_t = f(h_{t-1}, x_t)$), making training and inference significantly slower.
> 2. **Vanishing Gradients**: CNNs avoid recurrent feedback loops and gradient degradation over long sequences.
> 3. **Memory & Footprint**: A 1D-CNN has fewer recurrent gate parameters, making it small enough to run on microcontrollers, phones, or embedded DSPs."

---

### Q2: Why did you split the dataset by Subject ID rather than shuffling all rows randomly?
> **Answer**:
> "Because in biomedical and wearable time-series, random shuffling creates **severe data leakage**. Consecutive sliding windows overlap by 50% and share identical biomechanical traits. If windows from the same subject are in both training and test sets, the model simply memorizes that individual's unique walking style rather than learning generalized human activity patterns. Our subject-wise split rigorously tests **cross-subject generalization** on completely unseen individuals."

---

### Q3: Why is the window size chosen as 128 samples (2.56 seconds)?
> **Answer**:
> "At 50 Hz, 128 samples corresponds to 2.56 seconds. The normal human walking cadence is roughly 1.6 to 2.0 steps per second (a stride takes ~1.0 to 1.4 seconds). 
> * If the window is **too short** (< 1 sec), it cannot capture a complete gait cycle (heel strike, stance, toe off, swing).
> * If the window is **too long** (> 5 sec), multiple activities overlap in a single window, creating transition latency and delayed predictions."

---

### Q4: Why did the live model initially predict 'Lying' with 100% confidence when the phone was held flat?
> **Answer**:
> "In the HARTH dataset, sensors were taped vertically to the spine and thigh. Upright gravity was aligned with the negative X-axis ($X \approx -1.0g$, $Z \approx 0$). In the dataset, the only posture where gravity moves off the X-axis onto the Z-axis ($X \approx 0$, $Z \approx +1.0g$) is **Lying flat**.
> When holding an Android phone flat in hand (screen facing up), Android's coordinate system outputs $X=0, Y=0, Z=+9.8 \text{ m/s}^2$ ($+1.0g$). The neural network saw $X=0$ and $Z=1.0g$ and mathematically recognized it as lying down.
> We resolved this in [`realtime_server.py`](file:///home/leyoking/college/IMU%20SENSOR/realtime_server.py) by implementing an intelligent **gravity vector alignment** step that remaps the primary vertical axis dynamically."

---

### Q5: The HARTH dataset has 2 sensors (Back and Thigh, 6 channels). How did you bridge this with a single handheld phone?
> **Answer**:
> "A smartphone contains 1 IMU (3 accelerometer channels). To interface with the 6-channel model without retraining from scratch:
> 1. In [`realtime_server.py`](file:///home/leyoking/college/IMU%20SENSOR/realtime_server.py), the 3-axis phone acceleration is mirrored across the 6 input channels.
> 2. Placing the phone in the **front pants pocket** provides a physical match to the thigh sensor.
> 3. For future production, the model architecture can be configured directly with `in_channels=3` (or `in_channels=6` using Accelerometer + Gyroscope from the single device)."

---

### Q6: Does the HARTH dataset have Gyroscope data? Why is Gyroscope displayed on the dashboard?
> **Answer**:
> "The original HARTH study recorded participants using Axivity AX3 loggers, which are tri-axial accelerometers only (no gyroscopes were recorded in the dataset CSVs). 
> However, modern smartphones feature 6-DOF IMUs with built-in gyroscopes. We integrated the phone's gyroscope stream into [`realtime_server.py`](file:///home/leyoking/college/IMU%20SENSOR/realtime_server.py) and converted the angular velocity from $\text{rad/s}$ to $\text{deg/s}$ to provide real-time rotational tracking (pitch, roll, yaw) on our web dashboard."

---

---

### Q7: Why is test accuracy 51% while training accuracy is 96%? How can this be improved?
> **Answer**:
> "This discrepancy is expected in **cross-subject zero-shot evaluation** due to:
> 1. **Biomechanical Diversity**: Differences in height, weight, gait speed, and posture across subjects.
> 2. **Class Imbalance**: Activities like Walking and Running have thousands of samples, whereas Stairs or Inactive Cycling have very few.
> 
> **To improve accuracy to 85%+**:
> * **Data Augmentation**: Add jittering (Gaussian noise), scaling, magnitude warping, and temporal stretching.
> * **Transfer Learning / Domain Adaptation**: Fine-tune on 10 seconds of user-specific calibration data.
> * **Focal Loss**: Replace Cross-Entropy with Focal Loss to penalize hard, underrepresented classes."

---

### Q8: How can the system predict heart rate without an optical PPG or ECG sensor?
> **Answer**:
> "The system implements a **Bio-Dynamic Heart Rate & Metabolic Estimation Model** based on exercise physiology:
> 1. It calculates the user's maximum heart rate using the **Tanaka clinical formula**: $HR_{\max} = 208 - (0.7 \times \text{Age})$.
> 2. It computes body mass index (BMI) from Height and Weight to adjust cardiac workload demands.
> 3. It maps the 1D-CNN activity classification and real-time acceleration dynamic intensity to **Metabolic Equivalent of Task (MET)** values (e.g., Lying = 1.0 MET, Walking = 3.5 MET, Running = 8.5–12.0 MET).
> 4. It determines the steady-state target heart rate via the **Karvonen Heart Rate Reserve formula**: $HR_{\text{target}} = HR_{\text{rest}} + \%HRR \times (HR_{\max} - HR_{\text{rest}})$.
> 5. It passes this through a **first-order cardiovascular lag filter** ($\tau \approx 20\text{–}30\text{s}$) to simulate realistic human heart rate acceleration and recovery curves.
> 6. Finally, it computes calorie expenditure using the **Keytel clinical equation**."

---

## 8. Bio-Dynamic Cardiovascular & Metabolic Equations

| Metric | Scientific Formula | Purpose |
| :--- | :--- | :--- |
| **Max Heart Rate** | $HR_{\max} = 208 - (0.7 \times \text{Age})$ | Tanaka equation for age-adjusted cardiovascular ceiling. |
| **Body Mass Index** | $\text{BMI} = \text{Weight (kg)} / (\text{Height (m)})^2$ | Anthropometric cardiac workload scaling factor. |
| **Motion Intensity** | $\sigma_{\text{motion}} = \text{std}(\sqrt{a_x^2 + a_y^2 + a_z^2})$ | Quantifies dynamic foot impact & cadence intensity from IMU. |
| **Cardiovascular Lag** | $HR_t = HR_{t-1} + \alpha (HR_{\text{target}} - HR_{t-1})$ | Simulates gradual physiological heart rate ramp & cool-down. |
| **Calorie Expenditure** | Keytel Formula (Age, Weight, HR, Gender) | Clinical energy expenditure rate in $\text{kcal/min}$. |

---

## 9. Summary of Project Files

| File Name | Purpose |
| :--- | :--- |
| [**`model_pytorch.py`**](file:///home/leyoking/college/IMU%20SENSOR/model_pytorch.py) | PyTorch 1D-CNN class definition ([`HAR1DCNN`](file:///home/leyoking/college/IMU%20SENSOR/model_pytorch.py#L4)) with dynamic permutation. |
| [**`train_pytorch.py`**](file:///home/leyoking/college/IMU%20SENSOR/train_pytorch.py) | Full training pipeline: Subject-wise loader, standardizer, window generator, training loop, and evaluation report. |
| [**`model_matlab.m`**](file:///home/leyoking/college/IMU%20SENSOR/model_matlab.m) | MATLAB Deep Learning Toolbox layer definition for the 1D-CNN. |
| [**`train_matlab.m`**](file:///home/leyoking/college/IMU%20SENSOR/train_matlab.m) | Complete MATLAB training script supporting `trainNetwork`. |
| [**`test_model.py`**](file:///home/leyoking/college/IMU%20SENSOR/test_model.py) | Offline unit testing script performing single-window inference on sample data. |
| [**`realtime_server.py`**](file:///home/leyoking/college/IMU%20SENSOR/realtime_server.py) | Multi-threaded backend integrating 1D-CNN inference, Tanaka/Karvonen Heart Rate engine, and telemetry API. |
| [**`dashboard.html`**](file:///home/leyoking/college/IMU%20SENSOR/dashboard.html) | Interactive web dashboard featuring live beating heart gauge, heart rate zones, activity classification, and dual oscilloscopes. |
| [**`problem statement.txt`**](file:///home/leyoking/college/IMU%20SENSOR/problem%20statement.txt) | Original college project specification and motivation. |


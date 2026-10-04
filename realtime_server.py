import json
import time
import collections
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
import torch
from model_pytorch import HAR1DCNN

# Activity Label Mapping
LABEL_NAMES = {
    0: "Walking",
    1: "Running",
    2: "Shuffling",
    3: "Stairs (ascending)",
    4: "Stairs (descending)",
    5: "Standing",
    6: "Sitting",
    7: "Lying",
    8: "Cycling (sit)",
    9: "Cycling (stand)",
    10: "Cycling (sit, inactive)",
    11: "Cycling (stand, inactive)"
}

# Normalization parameters from HARTH training set
MEAN = np.array([-0.885417, 0.001669, -0.147612, -0.507975, 0.059537, 0.532519])
STD = np.array([0.295474, 0.225258, 0.338422, 0.499131, 0.308348, 0.587967])

WINDOW_SIZE = 128
data_buffer = collections.deque(maxlen=WINDOW_SIZE)
buffer_lock = threading.Lock()

# User Physiological Profile (Defaults)
user_profile = {
    "age": 21,
    "weight": 70.0,       # kg
    "height": 175.0,      # cm
    "gender": "male",
    "resting_hr": 68.0    # BPM
}

# Cardiac & Metabolic State
cardiac_state = {
    "current_bpm": 68.0,
    "target_bpm": 68.0,
    "total_kcal": 0.0,
    "last_calc_time": time.time()
}

# Live Telemetry State
latest_state = {
    "activity": "Standing",
    "confidence": 0.0,
    "bufferSize": 0,
    "isStreaming": False,
    "lastStreamTime": 0,
    "sensors": {
        "ax": 0.0, "ay": 0.0, "az": 0.0,
        "gx": 0.0, "gy": 0.0, "gz": 0.0
    },
    "probabilities": [{"name": v, "prob": 0.0} for v in LABEL_NAMES.values()],
    "cardiac": {
        "bpm": 68.0,
        "zone": "Resting",
        "percentMaxHr": 35,
        "maxHr": 193,
        "bmi": 22.9,
        "met": 1.5,
        "kcalPerMin": 1.2,
        "totalKcal": 0.0,
        "motionIntensity": 0.02
    }
}

# Load PyTorch Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HAR1DCNN(in_channels=6, num_classes=12, sequence_length=WINDOW_SIZE).to(device)
try:
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    print("[*] Successfully loaded 'best_model.pth'")
except Exception as e:
    print(f"[!] Warning: Could not load best_model.pth ({e})")
model.eval()

def read_dashboard_html():
    try:
        with open("dashboard.html", "r", encoding="utf-8") as f:
            return f.read().encode("utf-8")
    except Exception:
        return b"<h1>Dashboard HTML file not found</h1>"

def update_cardiac_engine(activity_name, window_array):
    """
    Bio-Dynamic Heart Rate and Metabolic Energy Expenditure Engine.
    Combines 1D-CNN motion class, IMU dynamic variance, and user anthropometrics.
    """
    now = time.time()
    dt = max(0.1, min(1.0, now - cardiac_state["last_calc_time"]))
    cardiac_state["last_calc_time"] = now

    age = user_profile["age"]
    weight = user_profile["weight"]
    height = user_profile["height"]
    gender = user_profile["gender"]
    resting_hr = user_profile["resting_hr"]

    # 1. Tanaka Max Heart Rate Formula
    max_hr = 208.0 - (0.7 * age)

    # 2. BMI Calculation
    height_m = height / 100.0
    bmi = weight / (height_m * height_m)

    # 3. Dynamic Motion Intensity (RMS acceleration deviation from gravity)
    # window_array has shape (128, 6), columns 0,1,2 are accelerometer X,Y,Z
    accel_mag = np.sqrt(window_array[:, 0]**2 + window_array[:, 1]**2 + window_array[:, 2]**2)
    motion_intensity = float(np.std(accel_mag)) # dynamic jerk / step force

    # 4. Metabolic Equivalent of Task (MET) Mapping
    # Standard Compendium of Physical Activities values adapted for motion intensity
    activity_lower = activity_name.lower()
    if "lying" in activity_lower:
        met = 1.0
    elif "sitting" in activity_lower:
        met = 1.3
    elif "standing" in activity_lower:
        met = 1.5
    elif "shuffling" in activity_lower:
        met = 2.5 + (1.5 * motion_intensity)
    elif "walking" in activity_lower:
        met = 3.3 + (2.5 * motion_intensity)
    elif "stairs" in activity_lower:
        met = 5.0 + (3.0 * motion_intensity)
    elif "cycling" in activity_lower:
        met = 6.0 + (2.0 * motion_intensity)
    elif "running" in activity_lower:
        met = 8.0 + (6.0 * motion_intensity)
    else:
        met = 2.0

    # BMI demand adjustment (higher body mass demands slightly higher cardiac output)
    if bmi > 25.0:
        met *= (1.0 + (bmi - 25.0) * 0.015)

    # 5. Target Steady-State Heart Rate (Karvonen Heart Rate Reserve Formula)
    # Max MET roughly corresponds to 12.0 for average population
    met_max = max(10.0, 15.0 - (0.05 * age))
    intensity_ratio = min(1.0, max(0.0, (met - 1.0) / (met_max - 1.0)))
    target_bpm = resting_hr + intensity_ratio * (max_hr - resting_hr)

    # 6. Physiological Lag Filtering (Human cardiovascular response time constant tau ~ 20-30s)
    # alpha simulates the gradual rise and recovery of heart rate
    alpha = 0.06 * (dt / 0.1)
    cardiac_state["current_bpm"] += alpha * (target_bpm - cardiac_state["current_bpm"])
    cardiac_state["current_bpm"] = float(np.clip(cardiac_state["current_bpm"], resting_hr - 5, max_hr + 2))

    current_bpm = cardiac_state["current_bpm"]

    # 7. Heart Rate Training Zones
    pct_max = (current_bpm / max_hr) * 100.0
    if pct_max < 50.0:
        zone = "Resting"
    elif pct_max < 70.0:
        zone = "Fat Burn"
    elif pct_max < 85.0:
        zone = "Cardio"
    else:
        zone = "Peak"

    # 8. Calorie Expenditure Rate (Keytel Clinical Equation)
    if gender == "female":
        kcal_per_min = (-20.4022 + (0.4472 * current_bpm) - (0.1263 * weight) + (0.074 * age)) / 4.184
    else:
        kcal_per_min = (-55.0969 + (0.6309 * current_bpm) + (0.1988 * weight) + (0.2017 * age)) / 4.184
    kcal_per_min = max(0.8, kcal_per_min)

    # Accumulate total kcal
    cardiac_state["total_kcal"] += (kcal_per_min * (dt / 60.0))

    # Update state
    latest_state["cardiac"] = {
        "bpm": round(current_bpm, 1),
        "zone": zone,
        "percentMaxHr": int(round(pct_max)),
        "maxHr": int(round(max_hr)),
        "bmi": round(bmi, 1),
        "met": round(met, 1),
        "kcalPerMin": round(kcal_per_min, 1),
        "totalKcal": round(cardiac_state["total_kcal"], 2),
        "motionIntensity": round(motion_intensity, 2)
    }

def predict_current_window():
    with buffer_lock:
        if len(data_buffer) < WINDOW_SIZE:
            return
        window_array = np.array(data_buffer) # shape: (128, 6)

    # Normalize for 1D-CNN
    norm_window = (window_array - MEAN) / STD
    tensor = torch.tensor(norm_window, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx]) * 100.0
    detected_activity = LABEL_NAMES.get(top_idx, "Unknown")

    latest_state["activity"] = detected_activity
    latest_state["confidence"] = top_conf
    latest_state["probabilities"] = [
        {"name": LABEL_NAMES[i], "prob": float(probs[i]) * 100.0}
        for i in range(len(LABEL_NAMES))
    ]

    # Update Heart Rate & Metabolic Model based on motion & window
    update_cardiac_engine(detected_activity, window_array)

class DashboardServerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Serve Frontend UI
        if self.path in ["/", "/index.html"]:
            html_bytes = read_dashboard_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)

        # Serve Live Telemetry API
        elif self.path.startswith("/api/telemetry"):
            now = time.time()
            latest_state["isStreaming"] = (now - latest_state["lastStreamTime"] < 2.5)
            latest_state["bufferSize"] = len(data_buffer)

            json_bytes = json.dumps(latest_state).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(json_bytes)))
            self.end_headers()
            self.wfile.write(json_bytes)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        # Update User Physiological Profile
        if self.path.startswith("/api/profile"):
            try:
                data = json.loads(post_data.decode('utf-8'))
                if "age" in data: user_profile["age"] = int(data["age"])
                if "weight" in data: user_profile["weight"] = float(data["weight"])
                if "height" in data: user_profile["height"] = float(data["height"])
                if "gender" in data: user_profile["gender"] = str(data["gender"])
                if "resting_hr" in data: user_profile["resting_hr"] = float(data["resting_hr"])
                print(f"[*] Updated User Profile: {user_profile}")

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "profile_updated"}')
                return
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                return

        # Handle Sensor Logger Push Stream
        try:
            payload = json.loads(post_data.decode('utf-8'))
            latest_state["lastStreamTime"] = time.time()

            if "payload" in payload:
                for item in payload["payload"]:
                    sensor_name = item.get("name", "").lower()
                    vals = item.get("values", {})

                    # ACCELEROMETER
                    if "accel" in sensor_name or "totalacceleration" in sensor_name:
                        raw_ax = vals.get("x", 0.0) / 9.80665
                        raw_ay = vals.get("y", 0.0) / 9.80665
                        raw_az = vals.get("z", 0.0) / 9.80665

                        latest_state["sensors"]["ax"] = raw_ax
                        latest_state["sensors"]["ay"] = raw_ay
                        latest_state["sensors"]["az"] = raw_az

                        # Orientation Remapping (Gravity to -1.0g on primary axis)
                        abs_axes = [abs(raw_ax), abs(raw_ay), abs(raw_az)]
                        max_axis = int(np.argmax(abs_axes))

                        if max_axis == 1: # Portrait / pocket: vertical along Y
                            hx = -abs(raw_ay)
                            hy = raw_ax
                            hz = raw_az
                        elif max_axis == 2: # Flat in hand: vertical along Z
                            hx = -abs(raw_az)
                            hy = raw_ax
                            hz = raw_ay
                        else:
                            hx = -abs(raw_ax)
                            hy = raw_ay
                            hz = raw_az

                        sample = [hx, hy, hz, hx, hy, hz]
                        with buffer_lock:
                            data_buffer.append(sample)

                    # GYROSCOPE
                    elif "gyro" in sensor_name:
                        gx = vals.get("x", 0.0) * (180.0 / np.pi)
                        gy = vals.get("y", 0.0) * (180.0 / np.pi)
                        gz = vals.get("z", 0.0) * (180.0 / np.pi)

                        latest_state["sensors"]["gx"] = gx
                        latest_state["sensors"]["gy"] = gy
                        latest_state["sensors"]["gz"] = gz

            # Flat JSON fallback
            elif "ax" in payload:
                raw_ax = float(payload.get("ax", 0.0)) / 9.80665
                raw_ay = float(payload.get("ay", 0.0)) / 9.80665
                raw_az = float(payload.get("az", 0.0)) / 9.80665
                sample = [-abs(raw_ay), raw_ax, raw_az, -abs(raw_ay), raw_ax, raw_az]
                with buffer_lock:
                    data_buffer.append(sample)

            # Run 1D-CNN + Cardiac Inference
            predict_current_window()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        except Exception:
            self.send_response(400)
            self.end_headers()

    def log_message(self, format, *args):
        return

def run_server(port=8000):
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, DashboardServerHandler)
    print("=" * 80)
    print(f"🚀 AI Human Activity & Cardiac Monitoring Server running on port {port}")
    print(f"🖥️  WEB DASHBOARD : http://localhost:{port}/")
    print(f"📱 SENSOR STREAM : http://172.16.253.48:{port}/")
    print("=" * 80)
    print("[*] Server active. Open your browser and connect your phone!\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")

if __name__ == "__main__":
    run_server(8000)

import json
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

# Normalization parameters from training set
MEAN = np.array([-0.885417, 0.001669, -0.147612, -0.507975, 0.059537, 0.532519])
STD = np.array([0.295474, 0.225258, 0.338422, 0.499131, 0.308348, 0.587967])

WINDOW_SIZE = 128
data_buffer = collections.deque(maxlen=WINDOW_SIZE)
buffer_lock = threading.Lock()
latest_raw_sensors = {"ax": 0.0, "ay": 0.0, "az": 0.0}

# Load PyTorch model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HAR1DCNN(in_channels=6, num_classes=12, sequence_length=WINDOW_SIZE).to(device)
try:
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    print("[*] Successfully loaded 'best_model.pth'")
except Exception as e:
    print(f"[!] Warning: Could not load best_model.pth ({e})")
model.eval()

def predict_current_window():
    """Runs model inference on current rolling buffer."""
    with buffer_lock:
        if len(data_buffer) < WINDOW_SIZE:
            return None, []
        window_array = np.array(data_buffer)  # shape: (128, 6)

    # Normalize using training mean and std
    norm_window = (window_array - MEAN) / STD
    
    # Tensor shape: (1, 6, 128)
    tensor = torch.tensor(norm_window, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        
    # Get top 3 predictions
    top_indices = np.argsort(probs)[::-1][:3]
    top_preds = [(LABEL_NAMES[i], probs[i] * 100) for i in top_indices]
    return top_preds[0], top_preds

class SensorRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            payload = json.loads(post_data.decode('utf-8'))
            
            # Extract accelerometer values from Sensor Logger JSON payload
            samples_to_add = []
            if "payload" in payload:
                for item in payload["payload"]:
                    if item.get("name") == "accelerometer":
                        vals = item.get("values", {})
                        # Android sends m/s^2. Convert to 'g' (1g ≈ 9.80665 m/s^2)
                        raw_ax = vals.get("x", 0.0) / 9.80665
                        raw_ay = vals.get("y", 0.0) / 9.80665
                        raw_az = vals.get("z", 0.0) / 9.80665
                        
                        latest_raw_sensors["ax"] = raw_ax
                        latest_raw_sensors["ay"] = raw_ay
                        latest_raw_sensors["az"] = raw_az
                        
                        # COORDINATE MAPPING:
                        # In HARTH: Sensors are worn vertically on Back and Thigh.
                        # Downwards gravity corresponds to -1.0g along the X-axis (harth_x ≈ -1.0 when upright).
                        # On Android:
                        # - If holding phone vertically upright or in pocket:
                        #   Gravity is along Y (-9.8 or +9.8 depending on orientation).
                        # - If holding phone flat in hand:
                        #   Gravity is along Z (+9.8).
                        
                        # We map the primary vertical axis to HARTH X axis so gravity is -1.0g when upright:
                        # If phone is in pocket or held portrait: Android -Y is roughly HARTH X.
                        # If user holds phone flat: Android -Z is roughly HARTH X.
                        
                        # Intelligent orientation mapping:
                        # Find which phone axis has the highest absolute acceleration (which is gravity ≈ 1g)
                        abs_axes = [abs(raw_ax), abs(raw_ay), abs(raw_az)]
                        max_axis = np.argmax(abs_axes)
                        
                        if max_axis == 1: # Phone is held portrait / in pocket (vertical along Y)
                            sign = -1.0 if raw_ay > 0 else 1.0
                            hx = sign * abs(raw_ay) # map gravity to -1.0g along X
                            hy = raw_ax
                            hz = raw_az
                        elif max_axis == 2: # Phone is held flat in hand (vertical along Z)
                            # Remap so the model understands the person is upright rather than lying down!
                            hx = -abs(raw_az) # gravity mapped to -1.0g
                            hy = raw_ax
                            hz = raw_ay
                        else:
                            hx = -abs(raw_ax)
                            hy = raw_ay
                            hz = raw_az
                            
                        # Thigh and Back slots (6 channels)
                        # When sitting, thigh is horizontal (hx ≈ 0, hz ≈ -1), back is vertical (hx ≈ -1)
                        sample = [hx, hy, hz, hx, hy, hz]
                        samples_to_add.append(sample)
                        
            elif "ax" in payload:
                raw_ax = float(payload.get("ax", 0.0)) / 9.80665
                raw_ay = float(payload.get("ay", 0.0)) / 9.80665
                raw_az = float(payload.get("az", 0.0)) / 9.80665
                sample = [-abs(raw_ay), raw_ax, raw_az, -abs(raw_ay), raw_ax, raw_az]
                samples_to_add.append(sample)

            with buffer_lock:
                for s in samples_to_add:
                    data_buffer.append(s)

            # Predict
            top1, all_top = predict_current_window()
            if top1:
                name, conf = top1
                top2_str = f" | 2nd: {all_top[1][0]} ({all_top[1][1]:.0f}%)" if len(all_top) > 1 else ""
                bar_len = int(conf / 5)
                bar = "█" * bar_len + "-" * (20 - bar_len)
                
                # Live status line
                print(f"\r[Live Activity] -> {name:<20} [{bar}] {conf:5.1f}%{top2_str:<28} | Raw: (x={latest_raw_sensors['ax']:+.1f}, y={latest_raw_sensors['ay']:+.1f}, z={latest_raw_sensors['az']:+.1f})g", end="", flush=True)
            else:
                print(f"\r[Buffering] Collecting samples: {len(data_buffer)}/{WINDOW_SIZE}...", end="", flush=True)

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
    httpd = HTTPServer(server_address, SensorRequestHandler)
    print("=" * 80)
    print(f"🚀 Real-time IMU Inference Server running on port {port}")
    print("💡 TIPS FOR ACCURATE DETECTION:")
    print("  1. Best placement: Put the phone in your FRONT PANTS POCKET (Thigh) or hold it vertically.")
    print("  2. In HARTH dataset, sensors are on thigh and lower back.")
    print("  3. Rhythmic movements (Walking, Running) will now show up dynamically.")
    print("=" * 80)
    print("[*] Waiting for incoming sensor stream from phone...\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n[*] Server stopped.")

if __name__ == "__main__":
    run_server(8000)

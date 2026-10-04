import torch
import pandas as pd
import numpy as np
from model_pytorch import HAR1DCNN

# 1. Activity names mapping
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

LABEL_MAP = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 
    7: 6, 8: 7, 13: 8, 14: 9, 130: 10, 140: 11
}

# Pre-computed normalization params from training set
MEAN = np.array([-0.885417, 0.001669, -0.147612, -0.507975, 0.059537, 0.532519])
STD = np.array([0.295474, 0.225258, 0.338422, 0.499131, 0.308348, 0.587967])

def test_single_window():
    # 2. Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HAR1DCNN(in_channels=6, num_classes=12, sequence_length=128).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    
    # 3. Load sample from test data (S029.csv)
    test_file = "dataset/harth/S029.csv"
    print(f"Reading sample from {test_file}...")
    df = pd.read_csv(test_file)
    
    # Get a slice of 128 continuous samples (window)
    # We choose rows 1000 to 1128
    start_idx = 1000
    window_data = df.iloc[start_idx : start_idx + 128]
    
    # Extract features and true label
    feature_cols = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
    raw_features = window_data[feature_cols].values
    
    # Get the ground truth label from the center of the window
    raw_label = window_data['label'].iloc[64]
    true_class_idx = LABEL_MAP.get(raw_label, -1)
    true_class_name = LABEL_NAMES.get(true_class_idx, "Unknown")
    
    # 4. Preprocess: Normalize and reshape
    norm_features = (raw_features - MEAN) / STD
    
    # Convert to PyTorch tensor and add batch dimension: (1, sequence_length, channels)
    input_tensor = torch.tensor(norm_features, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 5. Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)
        predicted_idx = predicted_idx.item()
        confidence = confidence.item() * 100
        
    predicted_class_name = LABEL_NAMES.get(predicted_idx, "Unknown")
    
    # 6. Display results
    print("\n================ Inference Result ================")
    print(f"True Activity      : {true_class_name} (Raw Label: {raw_label})")
    print(f"Predicted Activity : {predicted_class_name} ({confidence:.2f}% confidence)")
    print("==================================================")
    if predicted_idx == true_class_idx:
        print("Success! Prediction matches the true label.")
    else:
        print("Prediction does not match.")

if __name__ == "__main__":
    test_single_window()

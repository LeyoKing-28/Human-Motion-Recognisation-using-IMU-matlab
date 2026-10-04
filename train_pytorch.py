import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model_pytorch import HAR1DCNN

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Activity Label Mapping for HARTH Dataset
LABEL_MAP = {
    1: 0,    # Walking
    2: 1,    # Running
    3: 2,    # Shuffling (standing with leg movement)
    4: 3,    # Stairs (ascending)
    5: 4,    # Stairs (descending)
    6: 5,    # Standing
    7: 6,    # Sitting
    8: 7,    # Lying
    13: 8,   # Cycling (sit)
    14: 9,   # Cycling (stand)
    130: 10, # Cycling (sit, inactive)
    140: 11  # Cycling (stand, inactive)
}

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

class HARDataset(Dataset):
    def __init__(self, X, y):
        # PyTorch Conv1D expects shape: (batch_size, num_channels, sequence_length)
        # So we transpose X from (num_samples, sequence_length, num_channels) to (num_samples, num_channels, sequence_length)
        X_transposed = np.transpose(X, (0, 2, 1))
        self.X = torch.tensor(X_transposed, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_subject_data(file_paths):
    """
    Loads CSV data from multiple subject files, cleans the data, and returns feature and label arrays.
    """
    all_features = []
    all_labels = []
    
    feature_cols = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
    
    for path in file_paths:
        print(f"Loading {os.path.basename(path)}...")
        df = pd.read_csv(path)
        
        # Clean any missing values
        df = df.dropna(subset=feature_cols + ['label'])
        
        # Map labels to 0-11 range. Ignore any labels not in LABEL_MAP.
        valid_mask = df['label'].isin(LABEL_MAP.keys())
        df = df[valid_mask]
        
        if df.empty:
            continue
            
        features = df[feature_cols].values
        labels = df['label'].map(LABEL_MAP).values
        
        all_features.append(features)
        all_labels.append(labels)
        
    return all_features, all_labels

def segment_into_windows(features_list, labels_list, window_size=128, step_size=64):
    """
    Segments the continuous signals into overlapping sliding windows.
    The label for each window is defined as the label of the center element.
    """
    windows = []
    window_labels = []
    
    for features, labels in zip(features_list, labels_list):
        n_samples = len(features)
        for i in range(0, n_samples - window_size + 1, step_size):
            window = features[i : i + window_size]
            # Use label of the center sample of the window
            label = labels[i + window_size // 2]
            
            windows.append(window)
            window_labels.append(label)
            
    return np.array(windows), np.array(window_labels)

def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001, save_path="best_model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")
              
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model to {save_path}")

def evaluate_model(model, test_loader, device, save_path="best_model.pth"):
    # Load the best trained model weights
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print(f"\nLoaded best model from {save_path} for testing.")
        
    model.eval()
    test_correct = 0
    test_total = 0
    
    # Store predictions and ground truths for confusion matrix calculation
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    test_acc = (test_correct / test_total) * 100
    print(f"\n=================== Test Results ===================")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Print classification report / summary per class
    from sklearn.metrics import classification_report
    target_names = [LABEL_NAMES[i] for i in sorted(list(set(all_targets)))]
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=target_names, zero_division=0))

def main():
    # Paths
    dataset_dir = "dataset/harth"
    file_pattern = os.path.join(dataset_dir, "*.csv")
    file_paths = sorted(glob.glob(file_pattern))
    
    if not file_paths:
        print(f"Error: No CSV files found in {dataset_dir}. Please check your dataset folder.")
        return
        
    print(f"Found {len(file_paths)} subject files in {dataset_dir}.")
    
    # Split subjects into Train, Val, and Test to prevent subject-wise leakage
    # We use:
    # Train: S006 to S022 (first 15 subjects)
    # Val: S023 to S025 (next 3 subjects)
    # Test: S026 to S029 (remaining 4 subjects)
    train_files = file_paths[:15]
    val_files = file_paths[15:18]
    test_files = file_paths[18:]
    
    print("\nSubject Splitting:")
    print(f"Train subjects ({len(train_files)}): {[os.path.basename(f) for f in train_files]}")
    print(f"Val subjects ({len(val_files)}): {[os.path.basename(f) for f in val_files]}")
    print(f"Test subjects ({len(test_files)}): {[os.path.basename(f) for f in test_files]}")
    
    # Load raw data
    print("\n--- Loading Data ---")
    train_features, train_labels = load_subject_data(train_files)
    val_features, val_labels = load_subject_data(val_files)
    test_features, test_labels = load_subject_data(test_files)
    
    # Standardize features using training data parameters
    print("\nStandardizing features...")
    # Concatenate train features to calculate global mean and std
    flat_train = np.vstack(train_features)
    mean = np.mean(flat_train, axis=0)
    std = np.std(flat_train, axis=0)
    
    # Normalize datasets
    train_features = [(f - mean) / std for f in train_features]
    val_features = [(f - mean) / std for f in val_features]
    test_features = [(f - mean) / std for f in test_features]
    
    # Segment data into windows
    print("\n--- Segmenting signals into windows (window=128, overlap=50%) ---")
    X_train, y_train = segment_into_windows(train_features, train_labels, window_size=128, step_size=64)
    X_val, y_val = segment_into_windows(val_features, val_labels, window_size=128, step_size=64)
    X_test, y_test = segment_into_windows(test_features, test_labels, window_size=128, step_size=64)
    
    print(f"Train samples shape: {X_train.shape}, labels: {y_train.shape}")
    print(f"Val samples shape: {X_val.shape}, labels: {y_val.shape}")
    print(f"Test samples shape: {X_test.shape}, labels: {y_test.shape}")
    
    # Create PyTorch datasets and loaders
    train_dataset = HARDataset(X_train, y_train)
    val_dataset = HARDataset(X_val, y_val)
    test_dataset = HARDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on device: {device}")
    
    # Initialize Model (in_channels=6, num_classes=12, sequence_length=128)
    model = HAR1DCNN(in_channels=6, num_classes=12, sequence_length=128).to(device)
    
    # Start training
    print("\n--- Training 1D CNN ---")
    train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001, save_path="best_model.pth")
    
    # Evaluate model
    print("\n--- Evaluating Model ---")
    evaluate_model(model, test_loader, device, save_path="best_model.pth")

if __name__ == "__main__":
    main()

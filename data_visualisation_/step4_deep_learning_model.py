"""
AUTO SOS — Step 4: End-to-End Deep Learning Pipeline
=====================================================

This script bypasses the traditional manual feature engineering (the 17 stats) 
by feeding the RAW 300-point accelerometer arrays directly into a custom 
Deep Learning model.

Architecture: 1D CNN + LSTM
===========================
1. Convolutional Neural Network (1D-CNN): Automatically invents and extracts spatial features 
   (e.g., sharp spikes, impacts, free-fall signatures) by sliding filters over the raw signal.
2. Long Short-Term Memory (LSTM): Learns the temporal sequence and context of these extracted 
   features (e.g., recognizing that "free-fall" followed immediately by "impact" = Danger).

Pipeline:
  1. Load all raw CSV files.
  2. Segment into raw 300x3 matrix windows `(N_windows, 300, 3)`.
  3. Standardize and tensorize the 3D data.
  4. Train a custom PyTorch Neural Network.
  5. Evaluate results and save the .pth weights.

Run:  python3 step4_deep_learning_model.py
"""

import sys
import time
from pathlib import Path

# ---------- Auto-Install Dependencies if Missing ----------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
except ImportError:
    print("Installing PyTorch and other required dependencies for Deep Learning...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                           "torch", "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn"])
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR  = Path(__file__).parent
DATA_DIR    = SCRIPT_DIR / "new_datapoints"
OUTPUT_DIR  = SCRIPT_DIR / "output_images"

WINDOW_SIZE = 300
SENSOR_TYPE = "accelerometer"
DANGER_KEYWORDS = ["fall", "shaking", "snatch", "snatching", "impact"]

# Hyperparameters
BATCH_SIZE = 8  # Reduced further to increase updates per epoch even on tiny data
LEARNING_RATE = 5e-5  # Reduced from 0.0001 to prevent divergence
WEIGHT_DECAY = 1e-4  # Reduced from 1e-3 for stability
EPOCHS = 100  # Increased, but will be caught by Early Stopping
PATIENCE = 15  # Early stopping patience
TEST_SIZE = 0.2

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# ============================================================================
# 1. RAW DATA EXTRACTION (NO MANUAL FEATURES)
# ============================================================================

def label_from_filename(stem: str) -> int:
    lower = stem.lower()
    for kw in DANGER_KEYWORDS:
        if kw in lower:
            return 1 # DANGER
    return 0 # SAFE

def extract_raw_windows():
    print(f"\n[{time.strftime('%H:%M:%S')}] EXTRACTING RAW 300-POINT SENSOR ARRAYS...")
    
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in {DATA_DIR}")

    all_windows = []
    all_labels = []

    for csv_path in csv_files:
        label = label_from_filename(csv_path.stem)
        
        try:
            raw = pd.read_csv(csv_path)
            
            # Filter Accelerometer and drop NAs
            required = {"Sensor", "X", "Y", "Z"}
            if not required.issubset(raw.columns): continue
            
            accel = raw[raw["Sensor"].str.strip() == SENSOR_TYPE].copy()
            for ax in ["X", "Y", "Z"]:
                accel[ax] = pd.to_numeric(accel[ax], errors="coerce")
            accel = accel.dropna(subset=["X", "Y", "Z"]).reset_index(drop=True)
            
            n_rows = len(accel)
            n_windows = n_rows // WINDOW_SIZE
            
            for w_idx in range(n_windows):
                start = w_idx * WINDOW_SIZE
                # We pull raw X, Y, Z coordinates directly into a mathematically raw matrix
                window_matrix = accel.iloc[start : start + WINDOW_SIZE][["X", "Y", "Z"]].values
                
                all_windows.append(window_matrix)
                all_labels.append(label)
                
        except Exception as e:
            print(f"Skipping {csv_path.name}: {e}")

    # Shape: (Number of Windows, 300, 3)
    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    # Calculate class weights for PyTorch loss function to handle data imbalance
    num_danger = (y == 1).sum()
    num_safe = (y == 0).sum()
    pos_weight = num_safe / max(num_danger, 1)
    
    return X, y, torch.tensor([pos_weight], dtype=torch.float32).to(device)


# ============================================================================
# 2. THE DEEP LEARNING ARCHITECTURE (CNN + LSTM)
# ============================================================================

class AutoSOS_DeepNet(nn.Module):
    """
    Custom 1D Convolutional Neural Network followed by an LSTM.
    Instead of calculating "variance" or "max" manually, this model 
    invents its own mathematically optimal feature filters.
    """
    def __init__(self, input_channels=3, seq_length=300):
        super(AutoSOS_DeepNet, self).__init__()
        
        # --- FEATURE EXTRACTION (CNN) ---
        # Conv1D looks for brief, sharp anomalies (impacts) across the 3 sensors simultaneously
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2) # Reduces sequence length by half
        
        # Second layer looks for wider patterns (free-fall arcs)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2) # Reduces sequence length by half again
        
        # --- TEMPORAL CONTEXT (LSTM) ---
        # The LSTM looks at the sequences of extracted features to understand order
        # e.g.: Is there a period of zero-G (free-fall) exactly BEFORE a massive spike (impact)?
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        
        # --- CLASSIFICATION HEAD ---
        # Dense layer to output a final Danger Probability
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1) # Single output (Binary Logit)

    def forward(self, x):
        # PyTorch CNN expects shape: (Batch Size, Channels, Sequence Length)
        # We start with (Batch Size, Sequence Length, Channels) so we transpose
        x = x.transpose(1, 2)
        
        # CNN Pass
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        
        # PyTorch LSTM expects shape: (Batch Size, Sequence Length, Features)
        x = x.transpose(1, 2)
        
        # LSTM Pass
        out, (h_n, c_n) = self.lstm(x)
        
        # Grab the output at the very LAST time step of the LSTM
        last_timestep = out[:, -1, :]
        
        # Dense Classification Pass
        x = self.dropout(last_timestep)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x) 
        
        return x # Returns raw logits


# ============================================================================
# 3. TRAINING ROUTINE & EARLY STOPPING
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# ============================================================================
# 3. TRAINING ROUTINE
# ============================================================================

def prep_dataloaders(X, y):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=42)
    
    # Normalizing 3D tensor data [Batch, Timesteps, Channels]
    # We flatten, scale, and unflatten to treat each channel independently
    scaler = StandardScaler()
    b_tr, t_tr, c_tr = X_train.shape
    b_te, t_te, c_te = X_test.shape
    
    X_train_flat = X_train.reshape(-1, c_tr)
    X_test_flat  = X_test.reshape(-1, c_te)
    
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(b_tr, t_tr, c_tr)
    X_test_scaled  = scaler.transform(X_test_flat).reshape(b_te, t_te, c_te)
    
    # Convert to PyTorch Tensors
    train_datasets = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), 
                                   torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    test_datasets = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), 
                                  torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))
    
    train_loader = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_datasets, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, y_test

def train_model(model, train_loader, test_loader, pos_weight):
    print(f"\n[{time.strftime('%H:%M:%S')}] TRAINING DEEP NEURAL NETWORK...")
    print(f"Device: {device}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Switch to AdamW for better weight decay handling
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Use a more adaptive scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=PATIENCE)
    
    history_train_loss = []
    history_val_loss = []
    
    best_model_state = None
    min_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"   ⚠️ NaN detected at Epoch {epoch+1}. Aborting.")
                return model, history_train_loss, history_val_loss
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) # Tightened clipping
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
        avg_train_loss = epoch_train_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)
        
        # --- Validation Phase ---
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(test_loader)
        history_val_loss.append(avg_val_loss)
        
        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"   Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
        # Early Stopping check
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"   🛑 Early stopping at epoch {epoch+1}")
            break

    # Load the best weights back
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return model, history_train_loss, history_val_loss

# ============================================================================
# 4. EVALUATION & VISUALIZATION
# ============================================================================

def evaluate_and_plot(model, test_loader, y_true_test, history_train, history_val):
    print(f"\n[{time.strftime('%H:%M:%S')}] EVALUATING NETWORK PERFORMANCE...")
    model.eval()
    y_preds_proba = []
    
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            # Apply Sigmoid to convert raw logits to probabilities (0.0 to 1.0)
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_preds_proba.extend(probs)
            
    y_preds_proba = np.array(y_preds_proba).flatten()
    y_preds = (y_preds_proba >= 0.5).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_true_test, y_preds_proba)
    f1 = f1_score(y_true_test, y_preds)
    
    print("\n--- DEEP LEARNING RESULTS ---")
    print(f"ROC-AUC Score: {auc:.4f}")
    print(f"F1-Score:      {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true_test, y_preds, target_names=["SAFE", "DANGER"]))
    
    # Visualizations
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Training & Validation Loss Graph
    axes[0].plot(history_train, label='Train Loss', color='#3498db', marker='.', alpha=0.6)
    axes[0].plot(history_val, label='Val Loss', color='#e74c3c', marker='.', alpha=0.9)
    axes[0].set_title('Training vs Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('BCE Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true_test, y_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['SAFE', 'DANGER'], yticklabels=['SAFE', 'DANGER'])
    axes[1].set_title('Deep Learning Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / 'dl_evaluation_results.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Evaluation Plot saved: {plot_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("STEP 4: DEEP LEARNING ARCHITECTURE (CNN + LSTM)")
    print("="*80)
    
    # 1. Get raw windows (no feature engineering!)
    X, y, pos_weight = extract_raw_windows()
    print(f"✓ Formatted raw 3D Arrays: {X.shape} (Windows, Timesteps, Axes)")
    
    # 2. Create Loaders
    train_loader, test_loader, y_test = prep_dataloaders(X, y)
    
    # 3. Init Model
    model = AutoSOS_DeepNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Instantiated Neural Net ({total_params:,} parameters)")
    
    # 4. Train
    t0 = time.time()
    model, loss_tr, loss_val = train_model(model, train_loader, test_loader, pos_weight)
    print(f"✓ Training Completed in {time.time()-t0:.1f} seconds")
    
    # 5. Evaluate
    evaluate_and_plot(model, test_loader, y_test, loss_tr, loss_val)
    
    # 6. Save Engine
    save_path = OUTPUT_DIR / 'auto_sos_deep_model.pth'
    torch.save(model.state_dict(), save_path)
    print("\n" + "="*80)
    print(f"🚀 SUCCESS: Deep Learning Architecture saved to {save_path}")
    print("="*80)


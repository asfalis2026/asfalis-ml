"""
AUTO SOS — Step 5: ULTIMATE COMPARISON
=====================================

This script conducts a fair, side-by-side battle between:
1.  **Classic Machine Learning (Step 3)**: Using manually engineered statistical features (17 features).
2.  **Deep Learning (Step 4)**: Using raw 3D sensor arrays (CNN-LSTM architecture).

Both are trained and tested on the EXACT same data splits.

Run:  python3 step5_ultimate_comparison.py
"""

import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import xgboost as xgb

# Deep Learning Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import the architecture and helper from Step 4
from step4_deep_learning_model import AutoSOS_DeepNet, EarlyStopping

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR  = Path(__file__).parent
DATA_DIR    = SCRIPT_DIR / "new_datapoints"
OUTPUT_DIR  = SCRIPT_DIR / "output_images"
WINDOW_SIZE = 300
SENSOR_TYPE = "accelerometer"
DANGER_KEYWORDS = ["fall", "shaking", "snatch", "snatching", "impact"]
RANDOM_STATE = 42

# Optimized for small dataset stability
# Force CPU for benchmark to avoid MPS synchronization overhead on tiny tensors
device = torch.device('cpu') 

# ============================================================================
# 1. SHARED DATA EXTRACTION
# ============================================================================

def label_from_filename(stem: str) -> int:
    lower = stem.lower()
    for kw in DANGER_KEYWORDS:
        if kw in lower: return 1
    return 0

def extract_raw_windows():
    print(f"\n[{time.strftime('%H:%M:%S')}] EXTRACTING RAW DATA...")
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    all_windows = []
    all_labels = []

    for csv_path in csv_files:
        label = label_from_filename(csv_path.stem)
        try:
            raw = pd.read_csv(csv_path)
            accel = raw[raw["Sensor"].str.strip() == SENSOR_TYPE].copy()
            for ax in ["X", "Y", "Z"]:
                accel[ax] = pd.to_numeric(accel[ax], errors="coerce")
            accel = accel.dropna(subset=["X", "Y", "Z"]).reset_index(drop=True)
            n_windows = len(accel) // WINDOW_SIZE
            for w_idx in range(n_windows):
                start = w_idx * WINDOW_SIZE
                win = accel.iloc[start : start + WINDOW_SIZE][["X", "Y", "Z"]].values
                all_windows.append(win)
                all_labels.append(label)
        except: continue

    return np.array(all_windows, dtype=np.float32), np.array(all_labels, dtype=np.int64)

# ============================================================================
# 2. FEATURE ENGINEERING (for ML Models)
# ============================================================================

def engineer_features(X_raw):
    """Convert (N, 300, 3) raw windows to (N, 17) statistical features."""
    print(f"[{time.strftime('%H:%M:%S')}] ENGINEERING 17 STATISTICAL FEATURES...")
    X_feat = []
    for win in X_raw:
        # win shape is (300, 3) -> X, Y, Z
        features = []
        for i in range(3): # For each axis
            axis_data = win[:, i]
            features.append(np.mean(axis_data))
            features.append(np.std(axis_data))
            features.append(np.max(axis_data))
            features.append(np.min(axis_data))
            # Sum of squares
            features.append(np.sum(axis_data**2))
        
        # Sensor encoding (accelerometer=1, gyro=0)
        features.append(1.0)
        features.append(0.0)
        X_feat.append(features)
    
    return np.array(X_feat, dtype=np.float32)

# ============================================================================
# 3. COMPETITION RUNNER
# ============================================================================

def run_ml_benchmark(name, model, X_tr, X_te, y_tr, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    t_train = time.time() - t0
    
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    
    return {
        'Accuracy':  accuracy_score(y_te, y_pred),
        'Precision': precision_score(y_te, y_pred, zero_division=0),
        'Recall':    recall_score(y_te, y_pred, zero_division=0),
        'F1-Score':  f1_score(y_te, y_pred, zero_division=0),
        'ROC-AUC':   roc_auc_score(y_te, y_prob),
        'Time':      t_train
    }

def run_dl_benchmark(X_tr_raw, X_te_raw, y_tr, y_te):
    print(f"[{time.strftime('%H:%M:%S')}] TRAINING DEEP LEARNING (CNN-LSTM)...")
    
    # Scaling raw data: Flatten -> Scale -> Unflatten
    scaler = StandardScaler()
    b1, s1, c1 = X_tr_raw.shape
    b2, s2, c2 = X_te_raw.shape
    X_tr_scaled = scaler.fit_transform(X_tr_raw.reshape(-1, c1)).reshape(b1, s1, c1)
    X_te_scaled = scaler.transform(X_te_raw.reshape(-1, c2)).reshape(b2, s2, c2)
    
    print(f"   📊 Windows: {len(X_tr_raw) + len(X_te_raw)}")
    print(f"   📊 Train size: {len(X_tr_raw)}")
    
    train_ds = TensorDataset(torch.tensor(X_tr_scaled), torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1))
    test_ds  = TensorDataset(torch.tensor(X_te_scaled), torch.tensor(y_te, dtype=torch.float32).unsqueeze(1))
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    model = AutoSOS_DeepNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=5)
    
    t0 = time.time()
    for epoch in range(30):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation for early stopping
        model.eval()
        vloss = 0
        with torch.no_grad():
            for bx, by in test_loader:
                vloss += criterion(model(bx), by).item()
        vloss /= len(test_loader)
        
        print(f"   Epoch [{epoch+1}/30], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {vloss:.4f}", flush=True)

        early_stopping(vloss)
        if early_stopping.early_stop: 
            print(f"   🛑 Early stopping at epoch {epoch+1}", flush=True)
            break
    
    t_train = time.time() - t0
    
    # Predict
    model.eval()
    all_probs = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            probs = torch.sigmoid(model(bx)).cpu().numpy()
            all_probs.extend(probs)
    
    all_probs = np.array(all_probs).flatten()
    y_pred = (all_probs >= 0.5).astype(int)
    
    return {
        'Accuracy':  accuracy_score(y_te, y_pred),
        'Precision': precision_score(y_te, y_pred, zero_division=0),
        'Recall':    recall_score(y_te, y_pred, zero_division=0),
        'F1-Score':  f1_score(y_te, y_pred, zero_division=0),
        'ROC-AUC':   roc_auc_score(y_te, all_probs),
        'Time':      t_train
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ULTIMATE COMPARISON: CLASSIC ML vs DEEP LEARNING")
    print("="*80)

    # 1. Load shared windows
    X_raw, y = extract_raw_windows()
    X_feat = engineer_features(X_raw)

    # 2. Split (Identical for both)
    X_raw_tr, X_raw_te, X_feat_tr, X_feat_te, y_tr, y_te = train_test_split(
        X_raw, X_feat, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale Features for ML
    scaler_ml = StandardScaler()
    X_feat_tr_scaled = scaler_ml.fit_transform(X_feat_tr)
    X_feat_te_scaled = scaler_ml.transform(X_feat_te)

    results = []

    # Benchmark ML Models
    print(f"[{time.strftime('%H:%M:%S')}] TRAINING ML MODELS...")
    
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE)
    results.append({'Model': 'Random Forest', **run_ml_benchmark('RF', rf, X_feat_tr_scaled, X_feat_te_scaled, y_tr, y_te)})

    xgb_m = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE)
    results.append({'Model': 'XGBoost', **run_ml_benchmark('XGB', xgb_m, X_feat_tr_scaled, X_feat_te_scaled, y_tr, y_te)})

    # Benchmark DL Model
    results.append({'Model': 'CNN-LSTM (Deep)', **run_dl_benchmark(X_raw_tr, X_raw_te, y_tr, y_te)})

    # Summary
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("FINAL COMPARISON RESULTS")
    print("="*80)
    print(df.to_string(index=False))

    # Visualization
    plt.figure(figsize=(12, 6))
    df_melt = df.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
    sns.barplot(data=df_melt, x='variable', y='value', hue='Model')
    plt.title('Performance Comparison: Feature Engineering vs. Raw Deep Learning', fontweight='bold')
    plt.ylabel('Score (0.0 to 1.0)')
    plt.xlabel('Metric')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    comp_plot = OUTPUT_DIR / 'ultimate_comparison.png'
    plt.savefig(comp_plot, dpi=150)
    print(f"\n✓ Comparison plot saved: {comp_plot}")
    print("="*80)

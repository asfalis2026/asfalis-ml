"""
AUTO SOS — Step 3: Dedicated LightGBM Model Training
====================================================

This script focuses exclusively on training a LightGBM classifier for distress detection.
It includes:
1. Data loading and preprocessing (Scaling)
2. Imbalance-aware LightGBM training
3. Early stopping and hyperparameter optimization
4. Comprehensive evaluation and visualization
5. Model serialization for deployment

Run: python3 step3_lightgbm_training.py
"""

import numpy as np
import pandas as pd
import pickle
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, precision_score,
    recall_score, accuracy_score, average_precision_score
)

# Standard imports for LightGBM
try:
    import lightgbm as lgb
except ImportError:
    print("\n❌ LightGBM not found! Technical requirement missing.")
    print("Please run: pip install lightgbm\n")
    import sys
    sys.exit(1)

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / 'output_images'
INPUT_FEATURES = OUTPUT_DIR / 'features.npz'

TEST_SIZE = 0.2
VAL_SIZE = 0.1  # For early stopping
RANDOM_STATE = 42

# UI / Aesthetic Settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'Roboto', 'Arial'],
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#f8f9fa'
})

# ============================================================================
# UTILITIES
# ============================================================================

def print_header(text):
    print("\n" + "═" * 80)
    print(f" {text.upper()} ".center(80, "═"))
    print("═" * 80)

def print_step(phase, description):
    print(f"  ● {phase:15s} | {description}")

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def run_training():
    print_header("AUTO SOS: LightGBM Training Pipeline")
    
    # 1. Load Data
    print_step("DATA LOADING", f"Reading from {INPUT_FEATURES.name}")
    if not INPUT_FEATURES.exists():
        print(f"❌ Error: Features file not found. Run Step 1 and 2 first.")
        return

    data = np.load(INPUT_FEATURES)
    X, y = data['X'], data['y']
    
    # Feature Names (from Step 2 definition)
    feature_names = [
        'X_mean', 'X_std', 'X_max', 'X_min', 'X_sum_sq',
        'Y_mean', 'Y_std', 'Y_max', 'Y_min', 'Y_sum_sq',
        'Z_mean', 'Z_std', 'Z_max', 'Z_min', 'Z_sum_sq',
        'is_accelerometer', 'is_gyroscope'
    ]

    print_step("STATISTICS", f"Total Samples: {len(X):,} | Dimensions: {X.shape[1]}")
    print_step("CLASS BALANCE", f"Safe: {np.sum(y==0):,} | Danger: {np.sum(y==1):,}")

    # 2. Train-Test-Validation Split
    # We use a validation set for early stopping to prevent overfitting
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VAL_SIZE, 
        random_state=RANDOM_STATE, stratify=y_train_full
    )

    print_step("SPLIT", f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print_step("PREPROCESSING", "StandardScaler applied (Zero Mean, Unit Variance)")

    # 4. LightGBM Configuration
    # Optimized for the Asfalis 17-feature dataset
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'is_unbalance': True,  # Critical for SOS detection where danger is rare
        'random_state': RANDOM_STATE,
        'verbosity': -1,
        'n_jobs': -1
    }

    print_header("MODEL TRAINING")
    print_step("ALGORITHM", "LightGBM High-Performance Gradient Boosting")
    
    t0 = time.time()
    
    # Initialize model
    model = lgb.LGBMClassifier(**lgb_params)
    
    # Fit with early stopping
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50)
        ]
    )
    
    train_time = time.time() - t0
    print_step("SUCCESS", f"Training converged in {train_time:.2f}s")

    # 5. Evaluation
    print_header("EVALUATION METRICS")
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    print(f"  » Accuracy:  {acc:.4%}")
    print(f"  » Precision: {prec:.4%} (Trustworthiness of alarm)")
    print(f"  » Recall:    {rec:.4%} (Safety coverage)")
    print(f"  » F1-Score:  {f1:.4%}")
    print(f"  » ROC-AUC:   {auc:.4f}")
    print(f"  » Avg Prec:  {ap:.4f}")

    print("\n  Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['SAFE', 'DANGER']))

    # 6. Saving Artifacts
    print_header("SAVING MODEL")
    
    model_name = "asfalis_lgb_v1"
    model_path = OUTPUT_DIR / f'{model_name}.pkl'
    scaler_path = OUTPUT_DIR / 'asfalis_scaler.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    print_step("ARTIFACT", f"Model:  {model_path.name}")
    print_step("ARTIFACT", f"Scaler: {scaler_path.name}")

    # 8. Export for Android
    print_header("ANDROID COMPATIBILITY")
    try:
        from export_scaler_json import export as export_scaler
        export_scaler()
    except Exception as e:
        print(f"⚠️ Warning: Could not export scaler JSON: {e}")

    # 7. Visualizations
    generate_visualizations(model, X_test_scaled, y_test, y_proba, y_pred, feature_names)

# ============================================================================
# VISUALIZATION SUITE
# ============================================================================

def generate_visualizations(model, X_test, y_test, y_proba, y_pred, feature_names):
    print_header("GENERATING ANALYTICS")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['SAFE', 'DANGER'], yticklabels=['SAFE', 'DANGER'],
                cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
    plt.title('Confusion Matrix: LightGBM Performance', fontsize=14, pad=15)
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lgb_confusion_matrix.png', dpi=150)
    print_step("VISUAL", "Confusion Matrix saved")

    # 2. ROC & Precision-Recall Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax1.plot(fpr, tpr, color='#e74c3c', lw=3, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.3f})')
    ax1.plot([0, 1], [0, 1], color='#95a5a6', linestyle='--')
    ax1.set_title('Receiver Operating Characteristic', fontsize=12)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.2)

    # PR
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ax2.plot(recall, precision, color='#2980b9', lw=3, label=f'PR Curve (AP = {average_precision_score(y_test, y_proba):.3f})')
    ax2.set_title('Precision-Recall Curve', fontsize=12)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lgb_curves.png', dpi=150)
    print_step("VISUAL", "ROC and PR Curves saved")

    # 3. Feature Importance
    plt.figure(figsize=(10, 8))
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    plt.title('LightGBM Feature Importance (Gini)', fontsize=14)
    plt.barh(range(len(indices)), importances[indices], color='#1abc9c', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.grid(alpha=0.1, axis='x')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lgb_feature_importance.png', dpi=150)
    print_step("VISUAL", "Feature Importance Plot saved")

    # 4. Success Dashboard
    print_header("TRAINING COMPLETE")
    print(f"🚀 Model deployed to production-ready format.")
    print(f"📊 Visual insights available in {OUTPUT_DIR}/")
    print("═" * 80 + "\n")

if __name__ == "__main__":
    run_training()

"""
AUTO SOS — Step 3 ADVANCED: Model Training with Multiple Algorithms
===================================================================

This script:
1. Loads extracted features (from 300-reading windows)
2. Trains multiple advanced models:
   - Random Forest Classifier (Bagging Ensemble)
   - XGBoost (Gradient Boosting)
   - LightGBM (Light Gradient Boosting Machine)
   - AdaBoost (Adaptive Boosting)
3. Compares performance across all four algorithms
4. Evaluates on test set
5. Optimizes thresholds
6. Saves best model for deployment

Models Comparison:
  • RandomForest:  Good baseline, robust, handles noise well
  • XGBoost:       Strong gradient boosting, handles imbalance well
  • LightGBM:      Fastest, lowest memory footprint
  • AdaBoost:      Adaptive weight adjustment, great for weak learners

Run: python step3_advanced_model_training.py
"""

import numpy as np
import pandas as pd
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, precision_score,
    recall_score, accuracy_score, average_precision_score,
)
import time

warnings.filterwarnings("ignore")

# ============================================================================
# ADVANCED IMPORTS (Install if needed)
# ============================================================================

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not available. Install: pip install xgboost")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("⚠️  LightGBM not available. Install: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

# ============================================================================
# CONFIG
# ============================================================================

OUTPUT_DIR = Path(__file__).parent / 'output_images'
INPUT_FEATURES = OUTPUT_DIR / 'features.npz'

TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================================
# LOAD FEATURES
# ============================================================================

print("\n" + "="*80)
print("STEP 3 ADVANCED: MULTI-ALGORITHM MODEL TRAINING")
print("Algorithms: Random Forest | XGBoost | LightGBM | AdaBoost")
print("="*80)

print("\n→ Loading features...")
if not INPUT_FEATURES.exists():
    raise FileNotFoundError(f"Features file not found: {INPUT_FEATURES}")

features_data = np.load(INPUT_FEATURES)
X = features_data['X']
y = features_data['y']

print(f"  Features shape: {X.shape}")
print(f"  Labels shape:   {y.shape}")
print(f"  Window size:    300 readings")
print(f"  Class balance:  SAFE={int((y==0).sum())} | DANGER={int((y==1).sum())}")

# ============================================================================
# TRAIN-TEST SPLIT & SCALING
# ============================================================================

print("\n" + "="*80)
print("DATA PREPARATION")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTraining set: {X_train.shape[0]:,} samples")
print(f"Test set:     {X_test.shape[0]:,} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n✓ Features standardized (zero mean, unit variance)")

# Imbalance ratio for scale_pos_weight (XGBoost)
neg_pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

# Helper: evaluate a trained model
def evaluate_model(name, model, X_tr, X_te, y_tr, y_te, proba_col=1):
    """Fit, time, predict and return a results dict."""
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, proba_col]

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    auc  = roc_auc_score(y_te, y_proba)
    ap   = average_precision_score(y_te, y_proba)

    print(f"\n✓ {name} Training Complete  (⏱ {train_time:.1f}s)")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall:     {rec:.4f}")
    print(f"  F1-Score:   {f1:.4f}")
    print(f"  ROC-AUC:    {auc:.4f}")
    print(f"  Avg-Prec:   {ap:.4f}")

    return {
        'model':      model,
        'accuracy':   acc,
        'precision':  prec,
        'recall':     rec,
        'f1':         f1,
        'auc':        auc,
        'avg_prec':   ap,
        'train_time': train_time,
        'y_pred':     y_pred,
        'y_proba':    y_proba,
    }

results_dict = {}

# ============================================================================
# MODEL 1: RANDOM FOREST
# ============================================================================

print("\n" + "="*80)
print("MODEL 1: RANDOM FOREST CLASSIFIER")
print("="*80)
print("\n→ Training Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,          # Grow full trees
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced', # Handle imbalance
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

results_dict['RandomForest'] = evaluate_model(
    'RandomForest', rf_model,
    X_train_scaled, X_test_scaled, y_train, y_test
)

# Feature importances
fi_rf = pd.Series(rf_model.feature_importances_, name='importance').sort_values(ascending=False)
print("\n  Top 5 Feature Importances (Random Forest):")
for feat_idx, imp in fi_rf.head(5).items():
    print(f"    Feature {feat_idx}: {imp:.4f}")

# ============================================================================
# MODEL 2: XGBOOST
# ============================================================================

if XGBOOST_AVAILABLE:
    print("\n" + "="*80)
    print("MODEL 2: XGBOOST (GRADIENT BOOSTING)")
    print("="*80)
    print("\n→ Training XGBoost...")

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=neg_pos_ratio,   # Handle imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        eval_metric='logloss',
    )

    results_dict['XGBoost'] = evaluate_model(
        'XGBoost', xgb_model,
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    fi_xgb = pd.Series(xgb_model.feature_importances_, name='importance').sort_values(ascending=False)
    print("\n  Top 5 Feature Importances (XGBoost):")
    for feat_idx, imp in fi_xgb.head(5).items():
        print(f"    Feature {feat_idx}: {imp:.4f}")
else:
    print("\n⚠️  XGBoost skipped (not installed)")

# ============================================================================
# MODEL 3: LIGHTGBM
# ============================================================================

if LIGHTGBM_AVAILABLE:
    print("\n" + "="*80)
    print("MODEL 3: LIGHTGBM (LIGHT GRADIENT BOOSTING)")
    print("="*80)
    print("\n→ Training LightGBM...")

    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        is_unbalance=True,   # Handle imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    results_dict['LightGBM'] = evaluate_model(
        'LightGBM', lgb_model,
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    fi_lgb = pd.Series(lgb_model.feature_importances_, name='importance').sort_values(ascending=False)
    print("\n  Top 5 Feature Importances (LightGBM):")
    for feat_idx, imp in fi_lgb.head(5).items():
        print(f"    Feature {feat_idx}: {imp:.4f}")
else:
    print("\n⚠️  LightGBM skipped (not installed)")

# ============================================================================
# MODEL 4: ADABOOST
# ============================================================================

print("\n" + "="*80)
print("MODEL 4: ADABOOST (ADAPTIVE BOOSTING)")
print("="*80)
print("\n→ Training AdaBoost...")

# Use a slightly deeper base estimator to improve representation
base_dt = DecisionTreeClassifier(
    max_depth=3,
    class_weight='balanced',
    random_state=RANDOM_STATE,
)

ada_model = AdaBoostClassifier(
    estimator=base_dt,
    n_estimators=200,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
)

results_dict['AdaBoost'] = evaluate_model(
    'AdaBoost', ada_model,
    X_train_scaled, X_test_scaled, y_train, y_test
)

# ============================================================================
# COMPARISON TABLE
# ============================================================================

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

if len(results_dict) == 0:
    print("\n❌ ERROR: No models were trained successfully!")
    print("Please check your environment and re-run.")
    exit(1)

comparison_df = pd.DataFrame([
    {
        'Model':     name,
        'Accuracy':  r['accuracy'],
        'Precision': r['precision'],
        'Recall':    r['recall'],
        'F1-Score':  r['f1'],
        'ROC-AUC':   r['auc'],
        'Avg-Prec':  r['avg_prec'],
        'Train(s)':  round(r['train_time'], 2),
    }
    for name, r in results_dict.items()
])

comparison_df_display = comparison_df.copy()
for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Avg-Prec']:
    comparison_df_display[col] = comparison_df_display[col].map('{:.4f}'.format)

print("\n" + comparison_df_display.to_string(index=False))

# Best model by ROC-AUC
best_model_name = comparison_df.loc[comparison_df['ROC-AUC'].idxmax(), 'Model']

print(f"\n⭐ BEST MODEL (by ROC-AUC): {best_model_name}")
print(f"   ROC-AUC: {comparison_df['ROC-AUC'].max():.4f}")

# ============================================================================
# DETAILED EVALUATION — BEST MODEL
# ============================================================================

print("\n" + "="*80)
print(f"DETAILED EVALUATION: {best_model_name}")
print("="*80)

best_results = results_dict[best_model_name]
y_best_pred  = best_results['y_pred']
y_best_proba = best_results['y_proba']

cm = confusion_matrix(y_test, y_best_pred)
print("\nConfusion Matrix:")
print(f"  {'':10} Predicted-Safe  Predicted-Danger")
print(f"  Actual-Safe:     {cm[0,0]:4d}              {cm[0,1]:4d}")
print(f"  Actual-Danger:   {cm[1,0]:4d}              {cm[1,1]:4d}")

print("\nClassification Report:")
print(classification_report(y_test, y_best_pred, target_names=['SAFE', 'DANGER']))

# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

print("\n" + "="*80)
print("THRESHOLD OPTIMIZATION")
print("="*80)

sensitivity_thresholds = {
    'high':   0.35,
    'medium': 0.60,
    'low':    0.85
}

threshold_results = {}

for sens_name, threshold in sensitivity_thresholds.items():
    y_pred_thresh = (y_best_proba >= threshold).astype(int)

    acc  = accuracy_score(y_test, y_pred_thresh)
    prec = precision_score(y_test, y_pred_thresh, zero_division=0)
    rec  = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1   = f1_score(y_test, y_pred_thresh, zero_division=0)

    cm_thresh = confusion_matrix(y_test, y_pred_thresh)
    tn, fp, fn, tp = cm_thresh.ravel() if cm_thresh.size == 4 else (0, 0, 0, 0)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    threshold_results[sens_name] = {
        'threshold':       threshold,
        'accuracy':        acc,
        'precision':       prec,
        'recall':          rec,
        'f1':              f1,
        'false_alarm_rate': fpr,
        'miss_rate':       fnr,
    }

    print(f"\n{sens_name.upper()} (threshold={threshold:.2f}):")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Precision:         {prec:.4f}")
    print(f"  Recall:            {rec:.4f}")
    print(f"  F1-Score:          {f1:.4f}")
    print(f"  False alarm rate:  {fpr:.4f}")
    print(f"  Miss rate:         {fnr:.4f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams.update({'font.family': 'DejaVu Sans'})

MODEL_COLORS = {
    'RandomForest': '#3498db',
    'XGBoost':      '#2ecc71',
    'LightGBM':     '#e74c3c',
    'AdaBoost':     '#f39c12',
}

# ------------------------------------------------------------------
# Figure 1: Grouped bar — Model Comparison (5 metrics)
# ------------------------------------------------------------------
print("→ Creating model comparison plot...")
metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics_cols))
n_models = len(results_dict)
width = 0.18
offsets = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * width

fig, ax = plt.subplots(figsize=(14, 6))
for i, model_name in enumerate(results_dict.keys()):
    r = results_dict[model_name]
    values = [r['accuracy'], r['precision'], r['recall'], r['f1'], r['auc']]
    bars = ax.bar(x + offsets[i], values, width,
                  label=model_name,
                  color=MODEL_COLORS.get(model_name, '#888'),
                  alpha=0.87, edgecolor='white', linewidth=0.8)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Multi-Algorithm Comparison — RF | XGBoost | LightGBM | AdaBoost\n(300-Point Windows)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_cols, fontsize=11)
ax.legend(fontsize=10, framealpha=0.85)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.15])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '10_model_comparison.png', dpi=150, bbox_inches='tight')
print("  → Saved: 10_model_comparison.png")
plt.close()

# ------------------------------------------------------------------
# Figure 2: Radar / Spider chart — all models all metrics
# ------------------------------------------------------------------
print("→ Creating radar chart...")
from matplotlib.patches import FancyArrowPatch
radar_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
N = len(radar_metrics)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

plt.xticks(angles[:-1], radar_metrics, size=11, fontweight='bold')
ax.set_rlabel_position(0)
ax.set_yticks([0.25, 0.50, 0.75, 1.00])
ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], size=8, color='grey')
ax.set_ylim(0, 1)

for model_name, r in results_dict.items():
    vals = [r['accuracy'], r['precision'], r['recall'], r['f1'], r['auc']]
    vals += vals[:1]
    ax.plot(angles, vals, linewidth=2,
            label=model_name, color=MODEL_COLORS.get(model_name, '#888'))
    ax.fill(angles, vals, alpha=0.12, color=MODEL_COLORS.get(model_name, '#888'))

ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
ax.set_title('Algorithm Radar — All Metrics', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '10b_radar_comparison.png', dpi=150, bbox_inches='tight')
print("  → Saved: 10b_radar_comparison.png")
plt.close()

# ------------------------------------------------------------------
# Figure 3: ROC curves — all models on one canvas
# ------------------------------------------------------------------
print("→ Creating multi-model ROC curves...")
fig, ax = plt.subplots(figsize=(9, 7))

for model_name, r in results_dict.items():
    fpr_c, tpr_c, _ = roc_curve(y_test, r['y_proba'])
    ax.plot(fpr_c, tpr_c, linewidth=2.0,
            label=f"{model_name} (AUC={r['auc']:.3f})",
            color=MODEL_COLORS.get(model_name, '#888'))

ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '11_all_models_roc.png', dpi=150, bbox_inches='tight')
print("  → Saved: 11_all_models_roc.png")
plt.close()

# ------------------------------------------------------------------
# Figure 4: Best Model — Confusion Matrix
# ------------------------------------------------------------------
print("→ Creating confusion matrix...")
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['SAFE', 'DANGER'],
            yticklabels=['SAFE', 'DANGER'],
            ax=ax, cbar_kws={'label': 'Count'},
            annot_kws={'size': 16, 'weight': 'bold'})
ax.set_title(f'Confusion Matrix — {best_model_name} (Test Set)',
             fontsize=13, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '12_best_model_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("  → Saved: 12_best_model_confusion_matrix.png")
plt.close()

# ------------------------------------------------------------------
# Figure 5: Probability Distribution — Best Model
# ------------------------------------------------------------------
print("→ Creating probability distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(y_best_proba[y_test == 0], bins=30, alpha=0.6,
        label='SAFE (actual)',   color='#2ecc71', edgecolor='black')
ax.hist(y_best_proba[y_test == 1], bins=30, alpha=0.6,
        label='DANGER (actual)', color='#e74c3c', edgecolor='black')

thresh_colors = {'high': '#3498db', 'medium': '#f39c12', 'low': '#8e44ad'}
for sens, thresh in sensitivity_thresholds.items():
    ax.axvline(thresh, linestyle='--', linewidth=2,
               label=f'{sens.upper()} threshold ({thresh:.2f})',
               color=thresh_colors[sens])

ax.set_xlabel('Predicted Danger Probability', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title(f'Prediction Distribution — {best_model_name} (Test Set)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '13_best_model_distribution.png', dpi=150, bbox_inches='tight')
print("  → Saved: 13_best_model_distribution.png")
plt.close()

# ------------------------------------------------------------------
# Figure 6: Training time comparison
# ------------------------------------------------------------------
print("→ Creating training-time comparison...")
fig, ax = plt.subplots(figsize=(8, 5))
model_names   = list(results_dict.keys())
train_times   = [results_dict[m]['train_time'] for m in model_names]
colors_list   = [MODEL_COLORS.get(m, '#aaa') for m in model_names]
bars = ax.barh(model_names, train_times, color=colors_list, edgecolor='white', alpha=0.88)
for bar, t in zip(bars, train_times):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'{t:.1f}s', va='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Training Speed Comparison', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '14_training_time.png', dpi=150, bbox_inches='tight')
print("  → Saved: 14_training_time.png")
plt.close()

# ============================================================================
# SAVE BEST MODEL
# ============================================================================

print("\n" + "="*80)
print("SAVING BEST MODEL")
print("="*80)

best_model   = best_results['model']
model_path   = OUTPUT_DIR / f'auto_sos_model_{best_model_name}.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"\n✓ Saved {best_model_name} model: {model_path}")

scaler_path = OUTPUT_DIR / 'auto_sos_scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Saved scaler: {scaler_path}")

threshold_path = OUTPUT_DIR / 'threshold_config.pkl'
with open(threshold_path, 'wb') as f:
    pickle.dump(sensitivity_thresholds, f)
print(f"✓ Saved threshold config: {threshold_path}")

comparison_path = OUTPUT_DIR / 'model_comparison_results.pkl'
comp_payload = {
    'comparison_df':   comparison_df,
    'best_model':      best_model_name,
    'all_results':     {k: {kk: vv for kk, vv in v.items() if kk != 'model'}
                        for k, v in results_dict.items()},
    'threshold_results': threshold_results,
    'confusion_matrix':  cm.tolist(),
    'window_size':       300,
}
with open(comparison_path, 'wb') as f:
    pickle.dump(comp_payload, f)
print(f"✓ Saved comparison results: {comparison_path}")

# ============================================================================
# GENERATE REPORT
# ============================================================================

print("\n" + "="*80)
print("GENERATING COMPREHENSIVE REPORT")
print("="*80)

algo_analysis = """
### Random Forest
- **Type:** Bagging ensemble of decision trees
- **Pros:** Robust to noise, no feature scaling required, low variance, interpretable importances
- **Cons:** Slower inference than boosting models on large feature sets
- **Best for:** Production systems where reliability > raw accuracy

### XGBoost
- **Type:** Sequential gradient-boosted trees (regularised)
- **Pros:** Handles imbalance via `scale_pos_weight`, strong OOB regularisation, fast C++ backend
- **Cons:** More hyperparameters to tune, higher memory than LightGBM
- **Best for:** General-purpose fall detection with moderate dataset sizes

### LightGBM
- **Type:** Leaf-wise gradient boosting (histogram-based)
- **Pros:** Fastest training & inference, lowest RAM footprint, native imbalance handling
- **Cons:** Can overfit on very small datasets without careful `min_child_samples` tuning
- **Best for:** Real-time edge / mobile inference where speed is critical

### AdaBoost
- **Type:** Sequential re-weighting of weak learners (SAMME algorithm)
- **Pros:** Simple, interpretable; performs well when base learner captures key patterns
- **Cons:** Slower than gradient boosting variants, sensitive to noisy labels
- **Best for:** Lightweight deployment where a simpler model is preferred
"""

report_content = f"""
# AUTO SOS — ADVANCED MODEL TRAINING REPORT

## Configuration
- **Algorithms Tested:** Random Forest, XGBoost, LightGBM, AdaBoost
- **Window Size:** 300 readings (overlapping, 50 % overlap)
- **Feature Dimension:** {X.shape[1]} features per window
- **Train / Test Split:** {int((1-TEST_SIZE)*100)} % / {int(TEST_SIZE*100)} %

## Dataset Summary
| Metric        | Value         |
|---------------|---------------|
| Total samples | {len(X):,}  |
| Train samples | {len(X_train):,} |
| Test samples  | {len(X_test):,}  |
| Features      | {X.shape[1]}     |
| SAFE labels   | {int((y==0).sum())} |
| DANGER labels | {int((y==1).sum())} |

## Model Comparison

{comparison_df_display.to_string(index=False)}

**⭐ Best Model (by ROC-AUC): {best_model_name}**

## Detailed Metrics — {best_model_name}
| Metric    | Value  |
|-----------|--------|
| Accuracy  | {best_results['accuracy']:.4f} |
| Precision | {best_results['precision']:.4f} |
| Recall    | {best_results['recall']:.4f}    |
| F1-Score  | {best_results['f1']:.4f}        |
| ROC-AUC   | {best_results['auc']:.4f}       |
| Avg-Prec  | {best_results['avg_prec']:.4f}  |

## Confusion Matrix ({best_model_name})
```
                    Predicted-Safe  Predicted-Danger
Actual-Safe:        {cm[0,0]:4d}               {cm[0,1]:4d}
Actual-Danger:      {cm[1,0]:4d}               {cm[1,1]:4d}
```

## Sensitivity Configuration

| Level  | Threshold | Accuracy | Precision | Recall | False-Alarm | Miss-Rate |
|--------|-----------|----------|-----------|--------|-------------|-----------|
| HIGH   | 0.35      | {threshold_results['high']['accuracy']:.4f}   | {threshold_results['high']['precision']:.4f}    | {threshold_results['high']['recall']:.4f}  | {threshold_results['high']['false_alarm_rate']:.4f}       | {threshold_results['high']['miss_rate']:.4f}     |
| MEDIUM | 0.60      | {threshold_results['medium']['accuracy']:.4f}   | {threshold_results['medium']['precision']:.4f}    | {threshold_results['medium']['recall']:.4f}  | {threshold_results['medium']['false_alarm_rate']:.4f}       | {threshold_results['medium']['miss_rate']:.4f}     |
| LOW    | 0.85      | {threshold_results['low']['accuracy']:.4f}   | {threshold_results['low']['precision']:.4f}    | {threshold_results['low']['recall']:.4f}  | {threshold_results['low']['false_alarm_rate']:.4f}       | {threshold_results['low']['miss_rate']:.4f}     |

- **HIGH** — Medical / elderly care: catch every danger, tolerate false alarms
- **MEDIUM** — General population: balanced precision/recall
- **LOW** — Active adults: minimise false alarms

## Algorithm Analysis
{algo_analysis}

## Visualisations Generated
| File | Description |
|------|-------------|
| `10_model_comparison.png`          | Grouped bar chart — all metrics |
| `10b_radar_comparison.png`         | Radar / spider chart            |
| `11_all_models_roc.png`            | ROC curves — all models         |
| `12_best_model_confusion_matrix.png` | Best model confusion matrix   |
| `13_best_model_distribution.png`   | Probability distribution        |
| `14_training_time.png`             | Training speed comparison       |

## Next Steps
1. ✅ Model trained & saved → `auto_sos_model_{best_model_name}.pkl`
2. → Integrate **{best_model_name}** into `/api/protection/predict`
3. → Test on real-world sensor streams
4. → Monitor false-alarm rate in production
5. → Retrain monthly with new labelled data
"""

report_path = OUTPUT_DIR / 'ADVANCED_MODEL_REPORT.md'
with open(report_path, 'w') as f:
    f.write(report_content)
print(f"\n✓ Saved comprehensive report: {report_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ADVANCED MODEL TRAINING COMPLETE!")
print("="*80)

# Rank models for display
ranked = comparison_df.sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
print("\n📊 RANKINGS (by ROC-AUC):")
medals = ['🥇', '🥈', '🥉', '  ']
for i, row in ranked.iterrows():
    medal = medals[i] if i < len(medals) else '  '
    print(f"  {medal} {row['Model']:14s}  AUC={row['ROC-AUC']:.4f}  F1={row['F1-Score']:.4f}  "
          f"Prec={row['Precision']:.4f}  Rec={row['Recall']:.4f}  "
          f"Time={row['Train(s)']:.1f}s")

print(f"""
⭐ BEST MODEL: {best_model_name}
   • Accuracy:  {best_results['accuracy']:.4f}
   • Precision: {best_results['precision']:.4f}
   • Recall:    {best_results['recall']:.4f}
   • ROC-AUC:   {best_results['auc']:.4f}

📁 SAVED ARTIFACTS:
   • auto_sos_model_{best_model_name}.pkl
   • auto_sos_scaler.pkl
   • threshold_config.pkl
   • model_comparison_results.pkl

📈 VISUALIZATIONS:
   • 10_model_comparison.png
   • 10b_radar_comparison.png
   • 11_all_models_roc.png
   • 12_best_model_confusion_matrix.png
   • 13_best_model_distribution.png
   • 14_training_time.png

📖 REPORT:
   • ADVANCED_MODEL_REPORT.md

🔑 ALGORITHMS COMPARED:
   ✓ Random Forest  — robust bagging baseline
   ✓ XGBoost        — regularised gradient boosting
   ✓ LightGBM       — fastest leaf-wise boosting
   ✓ AdaBoost       — adaptive re-weighting

🚀 NEXT STEP:
   → Integrate {best_model_name} into /api/protection/predict endpoint
""")

print("="*80)
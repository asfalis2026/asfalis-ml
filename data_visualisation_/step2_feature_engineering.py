"""
AUTO SOS — Step 2: Feature Engineering (Convert Windows to 17-Feature Vectors)
===============================================================================

CRITICAL: Step 1 already created 300-point windows and extracted features.
This step converts those window features into the 17-feature vectors needed for
the ML model training in Step 3.

Input:  output_images/labeled_windows.csv (from Step 1)
Output: output_images/features.npz (X: windows × 17 features, y: labels)

Feature Breakdown (17 total):
  - [0:5]   : X-axis stats (mean, std, max, min, sum_sq)
  - [5:10]  : Y-axis stats (mean, std, max, min, sum_sq)
  - [10:15] : Z-axis stats (mean, std, max, min, sum_sq)
  - [15:17] : One-hot encoding (is_accelerometer, is_gyroscope)

Note: Since Step 1 currently only processes accelerometer data, 
      is_accelerometer will be 1 and is_gyroscope will be 0.

Run: python3 step2_feature_engineering.py
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

# ============================================================================
# CONFIG — paths relative to this script
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output_images"
INPUT_CSV  = OUTPUT_DIR / "labeled_windows.csv"  # From Step 1

# Feature names (must match exactly for backend/model compatibility)
FEATURE_NAMES = [
    # X-axis (0:5)
    'X_mean', 'X_std', 'X_max', 'X_min', 'X_sum_sq',
    # Y-axis (5:10)
    'Y_mean', 'Y_std', 'Y_max', 'Y_min', 'Y_sum_sq',
    # Z-axis (10:15)
    'Z_mean', 'Z_std', 'Z_max', 'Z_min', 'Z_sum_sq',
    # Sensor encoding (15:17)
    'is_accelerometer', 'is_gyroscope'
]

WINDOW_SIZE = 300

# ============================================================================
# BUILD 17-FEATURE VECTOR FROM WINDOW ROW
# ============================================================================

def build_17_feature_vector(row):
    """
    Convert a window row from labeled_windows.csv into a 17-feature vector.
    """
    features = []
    
    # helper to process axes
    for axis in ['x', 'y', 'z']:
        features.append(row[f'{axis}_mean'])
        features.append(row[f'{axis}_std'])
        features.append(row[f'{axis}_max'])
        features.append(row[f'{axis}_min'])
        
        # Calculate sum_sq accurately from RMS: sum(x^2) = rms^2 * N
        # If rms is missing, fallback to: (std^2 + mean^2) * N
        if f'{axis}_rms' in row:
            sum_sq = (row[f'{axis}_rms'] ** 2) * WINDOW_SIZE
        else:
            sum_sq = (row[f'{axis}_std']**2 + row[f'{axis}_mean']**2) * WINDOW_SIZE
        features.append(sum_sq)
        
    # Sensor type one-hot encoding (15:17)
    # Step 1 currently only does accelerometer
    features.append(1)  # is_accelerometer
    features.append(0)  # is_gyroscope
    
    return np.array(features, dtype=np.float32)


def validate_features(features):
    """Sanity check on extracted features."""
    if len(features) != 17:
        return False
    if not np.all(np.isfinite(features)):
        return False
    return True


# ============================================================================
# MAIN PIPELINE
# ============================================================================

print("\n" + "="*80)
print("STEP 2: FEATURE ENGINEERING (17-Feature Conversion)")
print("="*80)

if not INPUT_CSV.exists():
    print(f"❌ Input file not found: {INPUT_CSV}")
    print("   Make sure Step 1 completed successfully!")
    sys.exit(1)

df_windows = pd.read_csv(INPUT_CSV)
print(f"✓ Loaded {len(df_windows):,} windows from Step 1")

# ============================================================================
# BUILD 17-FEATURE VECTORS
# ============================================================================

print("\nBUILDING 17-FEATURE VECTORS...")
X_all = []
y_all = []
metadata_all = []
valid_count = 0
invalid_count = 0

for idx, row in df_windows.iterrows():
    try:
        features = build_17_feature_vector(row)
        
        if validate_features(features):
            X_all.append(features)
            y_all.append(row['danger_label'])
            metadata_all.append({
                'window_id': row['window_id'],
                'dataset': row['dataset_name'],
                'motion': row['motion_description'],
            })
            valid_count += 1
        else:
            invalid_count += 1
    except Exception as e:
        print(f"  ⚠ Error window {idx}: {e}")
        invalid_count += 1

print(f"✓ Created {valid_count} valid feature vectors")
if invalid_count > 0:
    print(f"  Skipped {invalid_count} invalid windows")

X = np.array(X_all, dtype=np.float32)
y = np.array(y_all, dtype=np.int32)

print(f"✓ Matrix shapes: X={X.shape}, y={y.shape}")

# Distribution
unique, counts = np.unique(y, return_counts=True)
print(f"\nLabel distribution:")
for label, count in zip(unique, counts):
    pct = count / len(y) * 100
    name = "SAFE" if label == 0 else "DANGER"
    print(f"  {name:6s} ({label}): {count:3d} windows ({pct:.1f}%)")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("\n" + "="*80)
print("SAVING OUTPUTS")
print("="*80)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Save as NPZ (standard for training)
features_output = OUTPUT_DIR / 'features.npz'
np.savez(features_output, X=X, y=y)
print(f"✓ Saved features.npz  → {features_output}")

# 2. Save detailed Pickle (includes metadata)
metadata_output = OUTPUT_DIR / 'metadata.pkl'
with open(metadata_output, 'wb') as f:
    pickle.dump({
        'X': X,
        'y': y,
        'metadata': metadata_all,
        'feature_names': FEATURE_NAMES,
        'window_size': WINDOW_SIZE,
        'total_windows': len(df_windows)
    }, f)
print(f"✓ Saved metadata.pkl  → {metadata_output}")

# 3. Save as CSV for easy viewing
df_features = pd.DataFrame(X, columns=FEATURE_NAMES)
df_features['label'] = y
df_features['motion'] = [m['motion'] for m in metadata_all]
features_csv = OUTPUT_DIR / 'features_final.csv'
df_features.to_csv(features_csv, index=False)
print(f"✓ Saved features_final.csv → {features_csv}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FEATURE ENGINEERING COMPLETE!")
print("="*80)
print(f"🚀 Ready for Step 3: Model Training")
print("="*80)
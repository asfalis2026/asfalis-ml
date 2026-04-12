"""
AUTO SOS — Step 1: Data Exploration (Window-Based)
===================================================

KEY INSIGHT:
  A single accelerometer reading CANNOT determine if an activity is
  dangerous or safe. Only a *pattern across time* can. This script
  groups accelerometer data into non-overlapping windows of 300 readings
  (matching the backend's /predict window size), extracts statistical
  features per window, and labels each window based on file origin.

Pipeline:
  1. Discover all CSVs  → new_datapoints/
  2. Filter accelerometer rows only
  3. Slice into 300-point non-overlapping windows (drop incomplete tail)
  4. Extract window-level features  (mean, std, min, max, mag, etc.)
  5. Label each window from filename keywords
  6. Statistical analysis on windows
  7. Visualisations  → output_images/
  8. Save the windowed feature table  → labeled_windows.csv

Run:  python3 step1_data_exploration.py
"""

import sys
from pathlib import Path

# ── dependency guard ──────────────────────────────────────────────────────────
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Installing missing dependencies…")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "pandas", "numpy", "matplotlib", "seaborn"])
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

# ============================================================================
# CONFIG
# ============================================================================

SCRIPT_DIR  = Path(__file__).parent
DATA_DIR    = SCRIPT_DIR / "new_datapoints"
OUTPUT_DIR  = SCRIPT_DIR / "output_images"

WINDOW_SIZE = 300          # must match backend /predict window size
SENSOR_TYPE = "accelerometer"

# Filename keywords → DANGER (label = 1); everything else → SAFE (label = 0)
DANGER_KEYWORDS = ["fall", "shaking", "snatch", "snatching", "impact"]


# ============================================================================
# HELPERS
# ============================================================================

def label_from_filename(stem: str) -> tuple[int, str]:
    lower = stem.lower()
    for kw in DANGER_KEYWORDS:
        if kw in lower:
            return 1, f"DANGER — {stem.replace('_', ' ').title()}"
    return 0, f"SAFE — {stem.replace('_', ' ').title()}"


def extract_window_features(window: pd.DataFrame, window_id: int,
                             dataset_name: str, danger_label: int,
                             motion_description: str) -> dict:
    """
    Summarise one 300-row accelerometer window into a flat feature dict.
    These are the same features that matter for fall/danger detection.
    """
    x, y, z = window["X"].values, window["Y"].values, window["Z"].values
    mag = np.sqrt(x**2 + y**2 + z**2)

    def stats(arr, prefix):
        return {
            f"{prefix}_mean":   arr.mean(),
            f"{prefix}_std":    arr.std(),
            f"{prefix}_min":    arr.min(),
            f"{prefix}_max":    arr.max(),
            f"{prefix}_range":  arr.max() - arr.min(),
            f"{prefix}_median": np.median(arr),
            f"{prefix}_iqr":    np.percentile(arr, 75) - np.percentile(arr, 25),
            f"{prefix}_rms":    np.sqrt(np.mean(arr**2)),
        }

    feat = {
        "window_id":          window_id,
        "dataset_name":       dataset_name,
        "danger_label":       danger_label,
        "motion_description": motion_description,
    }
    feat.update(stats(x,   "x"))
    feat.update(stats(y,   "y"))
    feat.update(stats(z,   "z"))
    feat.update(stats(mag, "mag"))

    # Cross-axis: correlation hints
    feat["xy_corr"] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0
    feat["xz_corr"] = float(np.corrcoef(x, z)[0, 1]) if x.std() > 0 and z.std() > 0 else 0.0
    feat["yz_corr"] = float(np.corrcoef(y, z)[0, 1]) if y.std() > 0 and z.std() > 0 else 0.0

    return feat


# ============================================================================
# STEP 1: DISCOVER FILES
# ============================================================================

print("\n" + "="*80)
print("STEP 1: DISCOVERING CSV FILES")
print("="*80)

if not DATA_DIR.exists():
    print(f"❌  Input folder not found: {DATA_DIR}")
    sys.exit(1)

csv_files = sorted(DATA_DIR.glob("*.csv"))
if not csv_files:
    print(f"❌  No CSV files found in {DATA_DIR}")
    sys.exit(1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n📂 Found {len(csv_files)} CSV files to process:")
for f in csv_files:
    lbl, desc = label_from_filename(f.stem)
    tag = "DANGER" if lbl else "SAFE  "
    print(f"   [{tag}]  {f.name}")

# ============================================================================
# STEP 2: LOAD, WINDOW & FEATURE-EXTRACT
# ============================================================================

print("\n" + "="*80)
print(f"STEP 2: WINDOWING  (window size = {WINDOW_SIZE} accelerometer readings)")
print("="*80)

all_windows: list[dict] = []

for csv_path in csv_files:
    stem  = csv_path.stem
    label, description = label_from_filename(stem)

    try:
        raw = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  ⚠️  Cannot read {csv_path.name}: {e} — skipping")
        continue

    # Validate columns
    required = {"Sensor", "X", "Y", "Z"}
    if not required.issubset(raw.columns):
        print(f"  ⚠️  Missing columns in {csv_path.name} — skipping")
        continue

    # Keep accelerometer rows only & cast axes to numeric
    accel = raw[raw["Sensor"].str.strip() == SENSOR_TYPE].copy()
    for ax in ["X", "Y", "Z"]:
        accel[ax] = pd.to_numeric(accel[ax], errors="coerce")
    accel = accel.dropna(subset=["X", "Y", "Z"]).reset_index(drop=True)

    n_rows    = len(accel)
    n_windows = n_rows // WINDOW_SIZE
    tail      = n_rows % WINDOW_SIZE

    print(f"\n  {csv_path.name}")
    print(f"   Accel rows : {n_rows:,}")
    print(f"   Windows    : {n_windows}  (tail of {tail} row{'s' if tail!=1 else ''} discarded)")

    if n_windows == 0:
        print(f"   ⚠️  Not enough rows for even one window — skipping")
        continue

    for w_idx in range(n_windows):
        start  = w_idx * WINDOW_SIZE
        window = accel.iloc[start : start + WINDOW_SIZE]
        feat   = extract_window_features(window, w_idx, stem, label, description)
        all_windows.append(feat)

if not all_windows:
    print("\n❌  No windows extracted. Exiting.")
    sys.exit(1)

df_win = pd.DataFrame(all_windows)
total_windows = len(df_win)

print(f"\n{'='*80}")
print(f"TOTAL WINDOWS: {total_windows:,}")
print(f"{'='*80}")

# ============================================================================
# STEP 3: BASIC STATS
# ============================================================================

print("\n" + "="*80)
print("STEP 3: WINDOW DISTRIBUTION")
print("="*80)

print("\nWindows per file → label:")
per_file = (df_win.groupby(["motion_description", "danger_label"])
            .size().rename("windows"))
for (desc, lbl), count in per_file.items():
    tag = "DANGER" if lbl else "SAFE  "
    print(f"   [{tag}]  {desc:45s}  {count:>5} windows")

safe_win   = (df_win["danger_label"] == 0).sum()
danger_win = (df_win["danger_label"] == 1).sum()
print(f"\nLabel distribution:")
print(f"  SAFE   (0): {safe_win:,} windows  ({safe_win/total_windows*100:.1f}%)")
print(f"  DANGER (1): {danger_win:,} windows  ({danger_win/total_windows*100:.1f}%)")

# ============================================================================
# STEP 4: FEATURE STATISTICS
# ============================================================================

print("\n" + "="*80)
print("STEP 4: KEY FEATURE STATISTICS BY LABEL")
print("="*80)

key_feats = ["mag_mean", "mag_std", "mag_max", "mag_range", "mag_rms"]
for feat in key_feats:
    safe_vals   = df_win[df_win["danger_label"] == 0][feat]
    danger_vals = df_win[df_win["danger_label"] == 1][feat]
    print(f"\n  {feat}:")
    print(f"    SAFE   → mean={safe_vals.mean():.3f}  std={safe_vals.std():.3f}  "
          f"min={safe_vals.min():.3f}  max={safe_vals.max():.3f}")
    print(f"    DANGER → mean={danger_vals.mean():.3f}  std={danger_vals.std():.3f}  "
          f"min={danger_vals.min():.3f}  max={danger_vals.max():.3f}")

# ============================================================================
# STEP 5: VISUALISATIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 5: GENERATING VISUALISATIONS")
print("="*80)

sns.set_style("whitegrid")
SAFE_COLOR   = "#2ecc71"
DANGER_COLOR = "#e74c3c"
palette      = {0: SAFE_COLOR, 1: DANGER_COLOR}

# ── Fig 1: Window count per file ─────────────────────────────────────────────
print("\n→ Fig 1: Window count per file…")
per_file_df = per_file.reset_index()
per_file_df["short"] = per_file_df["motion_description"].str.replace(
    r"^(SAFE|DANGER) — ", "", regex=True)
per_file_df["color"] = per_file_df["danger_label"].map(palette)

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.barh(
    per_file_df["short"], per_file_df["windows"],
    color=per_file_df["color"], edgecolor="black", linewidth=0.8, alpha=0.85
)
for bar, val in zip(bars, per_file_df["windows"]):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=8)
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=SAFE_COLOR,   label="SAFE"),
                  Patch(facecolor=DANGER_COLOR, label="DANGER")]
ax.legend(handles=legend_handles, loc="lower right")
ax.set_xlabel(f"Number of {WINDOW_SIZE}-point windows", fontsize=11)
ax.set_title(f"Windows Extracted per File  (window size = {WINDOW_SIZE})",
             fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
out1 = OUTPUT_DIR / "01_windows_per_file.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"   ✅ Saved → {out1}")
plt.close()

# ── Fig 2: Magnitude stats per window — box plots (SAFE vs DANGER) ───────────
print("→ Fig 2: Magnitude feature distributions (SAFE vs DANGER)…")
fig, axes = plt.subplots(1, len(key_feats), figsize=(5*len(key_feats), 6))
fig.suptitle("Window-Level Magnitude Features: SAFE vs DANGER",
             fontsize=14, fontweight="bold")

for ax, feat in zip(axes, key_feats):
    data = [
        df_win[df_win["danger_label"] == 0][feat].dropna().values,
        df_win[df_win["danger_label"] == 1][feat].dropna().values,
    ]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], [SAFE_COLOR, DANGER_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(["SAFE", "DANGER"], fontsize=10)
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.set_ylabel("Value (m/s²)")
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out2 = OUTPUT_DIR / "02_magnitude_features_boxplot.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"   ✅ Saved → {out2}")
plt.close()

# ── Fig 3: Per-dataset distribution of mag_mean (violin) ─────────────────────
print("→ Fig 3: mag_mean distribution per activity (violin)…")
fig, ax = plt.subplots(figsize=(16, 6))
order = df_win.groupby("motion_description")["mag_mean"].median().sort_values().index.tolist()
colors_list = [DANGER_COLOR if df_win[df_win["motion_description"]==d]["danger_label"].iloc[0]==1
               else SAFE_COLOR for d in order]

parts = ax.violinplot(
    [df_win[df_win["motion_description"] == d]["mag_mean"].values for d in order],
    showmedians=True, showextrema=True
)
for pc, col in zip(parts["bodies"], colors_list):
    pc.set_facecolor(col)
    pc.set_alpha(0.6)
parts["cmedians"].set_color("white")
parts["cmedians"].set_linewidth(2)

short_labels = [d.replace("SAFE — ", "").replace("DANGER — ", "") for d in order]
ax.set_xticks(range(1, len(order)+1))
ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Window mag_mean  (m/s²)", fontsize=11)
ax.set_title("Magnitude Mean per Window — Activity Distribution",
             fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
legend_handles2 = [Patch(facecolor=SAFE_COLOR,   label="SAFE"),
                   Patch(facecolor=DANGER_COLOR, label="DANGER")]
ax.legend(handles=legend_handles2)
plt.tight_layout()
out3 = OUTPUT_DIR / "03_mag_mean_violin.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
print(f"   ✅ Saved → {out3}")
plt.close()

# ── Fig 4: mag_std vs mag_mean scatter  (separability check) ─────────────────
print("→ Fig 4: mag_mean vs mag_std scatter ( separability )…")
fig, ax = plt.subplots(figsize=(10, 7))
for lbl, grp in df_win.groupby("danger_label"):
    tag = "DANGER" if lbl else "SAFE"
    ax.scatter(grp["mag_mean"], grp["mag_std"],
               color=palette[lbl], alpha=0.35, s=14, label=tag)
ax.set_xlabel("mag_mean  (m/s²)", fontsize=12)
ax.set_ylabel("mag_std   (m/s²)", fontsize=12)
ax.set_title("Window Separability: mag_mean vs mag_std",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out4 = OUTPUT_DIR / "04_scatter_mean_vs_std.png"
plt.savefig(out4, dpi=150, bbox_inches="tight")
print(f"   ✅ Saved → {out4}")
plt.close()

# ── Fig 5: Label balance ──────────────────────────────────────────────────────
print("→ Fig 5: Label (window) balance…")
fig, ax = plt.subplots(figsize=(7, 5))
bars2 = ax.bar(["SAFE (0)", "DANGER (1)"],
               [safe_win, danger_win],
               color=[SAFE_COLOR, DANGER_COLOR],
               alpha=0.85, edgecolor="black", linewidth=1.5)
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h,
            f"{int(h):,}\n({h/total_windows*100:.1f}%)",
            ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylabel("Number of Windows", fontsize=12, fontweight="bold")
ax.set_title("Window-Level Label Distribution", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
out5 = OUTPUT_DIR / "05_label_balance.png"
plt.savefig(out5, dpi=150, bbox_inches="tight")
print(f"   ✅ Saved → {out5}")
plt.close()

# ── Fig 6: Correlation heatmap of features ───────────────────────────────────
print("→ Fig 6: Feature correlation heatmap…")
feature_cols = [c for c in df_win.columns
                if c not in ("window_id", "dataset_name",
                             "danger_label", "motion_description")]
corr = df_win[feature_cols + ["danger_label"]].corr()

fig, ax = plt.subplots(figsize=(18, 14))
sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0,
            linewidths=0.3, annot=False, xticklabels=True, yticklabels=True)
ax.set_title("Feature Correlation Matrix  (including danger_label)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out6 = OUTPUT_DIR / "06_feature_correlation.png"
plt.savefig(out6, dpi=150, bbox_inches="tight")
print(f"   ✅ Saved → {out6}")
plt.close()

# ============================================================================
# STEP 6: SAVE WINDOWED FEATURE TABLE
# ============================================================================

print("\n" + "="*80)
print("STEP 6: SAVING LABELED WINDOWS")
print("="*80)

out_csv = OUTPUT_DIR / "labeled_windows.csv"
df_win.to_csv(out_csv, index=False)
print(f"\n✓ Saved → {out_csv}")
print(f"  Shape   : {df_win.shape[0]} windows × {df_win.shape[1]} columns")
print(f"  Columns : {', '.join(df_win.columns.tolist())}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

ratio = safe_win / danger_win if danger_win > 0 else float("inf")
print(f"""
📊 WINDOW SUMMARY:
  Window size      : {WINDOW_SIZE} accelerometer readings
  Files loaded     : {len(csv_files)}
  Total windows    : {total_windows:,}
  SAFE windows     : {safe_win:,}
  DANGER windows   : {danger_win:,}
  Imbalance ratio  : {ratio:.1f}:1  (safe:danger)
  Features/window  : {df_win.shape[1] - 4}  (excl. metadata)

📁 Outputs saved to: {OUTPUT_DIR}
   • 01_windows_per_file.png      — windows available per activity
   • 02_magnitude_features_boxplot.png — SAFE vs DANGER feature spread
   • 03_mag_mean_violin.png       — per-activity magnitude distribution
   • 04_scatter_mean_vs_std.png   — 2-D separability check
   • 05_label_balance.png         — window label balance
   • 06_feature_correlation.png   — feature correlation matrix
   • labeled_windows.csv          — feature table ready for Step 2

📈 NEXT STEPS:
  1. Review the plots, especially 04_scatter_mean_vs_std.png
     (good separation = model will work; overlap = need more features)
  2. Run Step 2 (Feature Engineering):   python step2_feature_engineering.py
  3. Run Step 3 (Model Training):         python step3_model_training.py
""")
print("="*80)
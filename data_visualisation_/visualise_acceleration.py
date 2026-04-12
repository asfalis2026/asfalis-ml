"""
visualise_acceleration.py
--------------------------
Plots a line graph of Timestamp vs Acceleration (X, Y, Z axes)
Handles both old format (MEDIUM_SAFE.csv) and new format (moderate_vigorous_shaking.csv)
Output file name matches input file name automatically

BATCH MODE: Automatically processes all CSV files in /new_datapoints folder
and generates corresponding PNG files in /output_images folder

Usage:
    python3 visualise_acceleration.py

Requirements:
    pip install matplotlib pandas
"""

import sys
from pathlib import Path

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    print("Missing dependencies. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "pandas"])
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates




def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Load data from CSV file.
    Supports both old format (Value 1, Value 2) and new format (Value)
    """
    df = pd.read_csv(csv_path)

    # Build a proper datetime timestamp from Date + Time columns
    df["Timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%Y-%m-%d %H:%M:%S.%f",
        errors="coerce"
    )

    # Keep only accelerometer rows (ignore gyroscope)
    df = df[df["Sensor"].str.strip() == "accelerometer"].copy()

    # Cast X, Y, Z to numeric
    for axis in ["X", "Y", "Z"]:
        df[axis] = pd.to_numeric(df[axis], errors="coerce")

    df = df.dropna(subset=["Timestamp", "X", "Y", "Z"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    return df


def plot(df: pd.DataFrame, output_png: Path, title: str):
    """
    Generate acceleration plot with 4 subplots (X, Y, Z, Magnitude)
    """
    # Compute resultant magnitude for a bonus overview panel
    df["Magnitude"] = (df["X"]**2 + df["Y"]**2 + df["Z"]**2) ** 0.5

    fig, axes = plt.subplots(
        nrows=4, ncols=1,
        figsize=(16, 12),
        sharex=True,
        facecolor="#0d1117"
    )
    fig.suptitle(
        f"{title} — Accelerometer Data  |  Index vs Acceleration",
        fontsize=15, fontweight="bold", color="#e6edf3",
        y=0.98
    )

    panel_cfg = [
        ("X", "#58a6ff", "X-axis (m/s²)"),
        ("Y", "#3fb950", "Y-axis (m/s²)"),
        ("Z", "#f78166", "Z-axis (m/s²)"),
        ("Magnitude", "#d2a8ff", "Magnitude (m/s²)"),
    ]

    # Create index array for x-axis (row numbers)
    x_indices = range(len(df))
    
    for ax, (col, color, ylabel) in zip(axes, panel_cfg):
        ax.set_facecolor("#161b22")
        ax.plot(x_indices, df[col], color=color, linewidth=0.7, alpha=0.9)
        ax.fill_between(x_indices, df[col], alpha=0.12, color=color)

        # Mean reference line
        mean_val = df[col].mean()
        ax.axhline(mean_val, color=color, linewidth=0.8, linestyle="--", alpha=0.5)

        ax.set_ylabel(ylabel, color="#8b949e", fontsize=9)
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.grid(axis="y", color="#21262d", linewidth=0.5)
        ax.grid(axis="x", color="#21262d", linewidth=0.3)

        # Annotate mean
        ax.text(
            0.01, 0.92, f"μ = {mean_val:.4f}",
            transform=ax.transAxes, color=color,
            fontsize=8, alpha=0.8
        )

    # X-axis: show time ticks every 300 timestamps
    num_points = len(df)
    tick_positions = list(range(0, num_points, 300))
    
    # Get corresponding timestamps for labels
    tick_labels = [df["Timestamp"].iloc[i].strftime("%H:%M:%S") for i in tick_positions]
    
    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels(tick_labels, rotation=45, ha="right", color="#8b949e", fontsize=8)
    axes[-1].set_xlabel("Timestamp (every 300 samples)", color="#8b949e", fontsize=10)
    axes[-1].set_xlim(0, num_points - 1)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # Save PNG
    plt.savefig(output_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"   ✅ Plot saved → {output_png}")
    
    # Close the figure to free memory
    plt.close()


def process_csv_file(csv_file: Path, output_dir: Path):
    """
    Process a single CSV file and generate corresponding PNG
    """
    file_name = csv_file.stem  # Get filename without extension
    output_png = output_dir / f"{file_name}.png"
    
    print(f"\n📄 Processing: {file_name}.csv")
    
    try:
        df = load_data(csv_file)
        
        if len(df) == 0:
            print(f"   ⚠️  No accelerometer data found - skipping")
            return False
        
        print(f"   Rows loaded : {len(df):,}")
        print(f"   Time range  : {df['Timestamp'].iloc[0]}  →  {df['Timestamp'].iloc[-1]}")
        print(f"   X range     : [{df['X'].min():.4f}, {df['X'].max():.4f}]")
        print(f"   Y range     : [{df['Y'].min():.4f}, {df['Y'].max():.4f}]")
        print(f"   Z range     : [{df['Z'].min():.4f}, {df['Z'].max():.4f}]")
        print(f"   📊 Rendering plot...")
        
        plot(df, output_png, file_name)
        return True
        
    except Exception as e:
        print(f"   ❌ Error processing file: {e}")
        return False


def main():
    # Input/Output directories
    INPUT_DIR = Path.cwd() / "new_datapoints"
    OUTPUT_DIR = Path.cwd() / "output_images"
    
    # Check if input folder exists
    if not INPUT_DIR.exists():
        print(f"❌ Error: Input folder '{INPUT_DIR}' does not exist!")
        print(f"   Please create the 'new_datapoints' folder and add CSV files to it.")
        return
    
    # Check if output_images folder exists, if not create it
    if not OUTPUT_DIR.exists():
        print(f"📁 Creating output_images folder...")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✅ output_images folder created at {OUTPUT_DIR}\n")
    else:
        print(f"📁 output_images folder found at {OUTPUT_DIR}\n")
    
    # Find all CSV files in the input directory
    csv_files = list(INPUT_DIR.glob("*.csv"))
    
    if len(csv_files) == 0:
        print(f"❌ No CSV files found in {INPUT_DIR}")
        return
    
    print(f"🔍 Found {len(csv_files)} CSV file(s) to process:")
    for csv_file in csv_files:
        print(f"   • {csv_file.name}")
    
    print(f"\n{'='*70}")
    print(f"🚀 Starting batch processing...")
    print(f"{'='*70}\n")
    
    # Process each CSV file
    successful = 0
    failed = 0
    
    for csv_file in sorted(csv_files):
        if process_csv_file(csv_file, OUTPUT_DIR):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"📊 Batch Processing Summary")
    print(f"{'='*70}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output folder: {OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
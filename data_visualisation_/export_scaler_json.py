import pickle
import json
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'output_images'
SCALER_PATH = OUTPUT_DIR / 'asfalis_scaler.pkl'
JSON_OUTPUT_PATH = OUTPUT_DIR / 'scaler_params.json'

def export():
    print(f"Loading scaler from {SCALER_PATH}...")
    if not SCALER_PATH.exists():
        print("❌ Error: Scaler file not found!")
        return

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # Extract mean and scale (standard deviation)
    # StandardScaler.scale_ is the standard deviation
    params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }

    with open(JSON_OUTPUT_PATH, 'w') as f:
        json.dump(params, f)

    print(f"✅ Success! Scaler parameters exported to {JSON_OUTPUT_PATH}")
    print(f"Mean (first 3): {params['mean'][:3]}")
    print(f"Scale (first 3): {params['scale'][:3]}")

if __name__ == "__main__":
    export()

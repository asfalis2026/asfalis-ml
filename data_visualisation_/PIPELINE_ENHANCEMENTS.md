# Asfalis ML Pipeline — System Enhancements (April 2026)

This document details the critical enhancements made to the Machine Learning pipeline to ensure seamless integration with the Asfalis Android application. These changes focus on automation, mobile compatibility, and mathematical accuracy.

## 1. Automated Android Export
We have introduced a new tool, **`export_scaler_json.py`**, which acts as the bridge between the Python training environment and the Android deployment.

*   **Function:** Extracts `StandardScaler` parameters (mean and standard deviation) from `asfalis_scaler.pkl`.
*   **Output:** Generates `scaler_params.json` in the `output_images/` directory.
*   **Automation:** This script is now called automatically at the end of **`step3_lightgbm_training.py`**.
*   **Benefit:** Eliminates manual data entry errors in the frontend, ensuring the Android app always uses the exact normalization factors the model was trained on.

## 2. Optimized ONNX Conversion
The **`convert_to_onnx_unified.py`** script has been enhanced to produce a production-ready model for mobile devices.

*   **ZipMap Removal:** We have explicitly disabled `ZipMap` in the ONNX conversion options.
*   **Mobile Impact:** ONNX Runtime on Android prefers simple float arrays over complex map structures. This change ensures the model output (`P(DANGER)`) can be read instantaneously by the app without additional parsing logic.

## 3. Unit Standardization (g Units)
A technical audit revealed that the training data and model metadata were mathematically centered around **g units** (where resting gravity ≈ 1.0).

*   **Update:** All visualization labels and plot axes in **`step1_data_exploration.py`** have been updated from `m/s²` to `g`.
*   **Consistency:** The Android app has been updated to scale its raw `m/s²` readings to `g` units before inference, ensuring 100% alignment with this training pipeline.

## 4. Usage Instructions
To retrain and deploy a new model for the app:
1. Run `python step1_data_exploration.py`
2. Run `python step2_feature_engineering.py`
3. Run `python step3_lightgbm_training.py` (this will automatically create the `scaler_params.json`)
4. Run `python convert_to_onnx_unified.py`

Copy the resulting `.onnx` and `.json` files from `output_images/` into the Android app's `assets/` folder.

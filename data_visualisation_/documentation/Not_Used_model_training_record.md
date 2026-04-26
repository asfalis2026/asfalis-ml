# Technical Report: Intelligent Distress Detection via Temporal Feature Engineering and Gradient Boosting

**Author:** Antigravity ML Engineering Unit  
**Date:** April 26, 2026  
**Subject:** Asfalis Auto SOS Model Training, Feature Evolution, and Evaluation  

---

## Abstract
This report provides a comprehensive analysis of the machine learning pipeline developed for the Asfalis Auto SOS system. The system identifies high-risk physical events—such as falls, snatching, and impacts—by analyzing tri-axial accelerometer data. We implement a three-phase data evolution process, transforming raw time-series logs into 17-dimensional feature vectors. Utilizing a LightGBM classifier with imbalance-aware training, the model achieves high recall and precision, ensuring robust safety coverage for mobile users.

---

## 1. Introduction
Real-time distress detection on mobile devices requires a balance between computational efficiency and predictive accuracy. Since a single sensor reading cannot distinguish between a fall and a common movement, the Asfalis system focuses on **temporal patterns**. By analyzing windows of motion, the model can identify the specific "signature" of a dangerous event. This report details the end-to-end process from raw data acquisition to model deployment.

---

## 2. Materials and Methods

### 2.1 Phase 1: Raw Data Acquisition
The foundation of the model is raw tri-axial data collected from mobile sensors. Each session is stored as a CSV file representing a specific activity (e.g., "walking", "running", "fall_test").

**Raw Dataset Header Structure:**
| Column | Type | Description |
| :--- | :--- | :--- |
| `Sensor` | String | The type of sensor (e.g., `accelerometer`, `gyroscope`) |
| `X` | Float | Acceleration/Rotation along the X-axis |
| `Y` | Float | Acceleration/Rotation along the Y-axis |
| `Z` | Float | Acceleration/Rotation along the Z-axis |
| `Timestamp` | Integer | Millisecond timestamp for sequencing |

### 2.2 Phase 2: Temporal Segmentation and Labeling
To analyze motion patterns, raw data is grouped into **Tumbling Windows**:
-   **Window Size ($N$):** 300 readings (approx. 6 seconds at 50Hz).
-   **Method:** Non-overlapping windows. Incomplete segments at the end of files are discarded.
-   **Labeling Logic:** Ground truth is derived from the source filename. A window is labeled as **DANGER (1)** if the filename contains any of the following case-insensitive keywords:
    -   `fall`, `shaking`, `snatch`, `snatching`, `impact`
-   **Labeling Logic (Safe):** All other files are labeled as **SAFE (0)**.

**Intermediate Labeled Dataset Headers (`labeled_windows.csv`):**
| Group | Columns |
| :--- | :--- |
| **Metadata** | `window_id`, `dataset_name`, `danger_label`, `motion_description` |
| **X-Axis Stats** | `x_mean`, `x_std`, `x_min`, `x_max`, `x_range`, `x_median`, `x_iqr`, `x_rms` |
| **Y-Axis Stats** | `y_mean`, `y_std`, `y_min`, `y_max`, `y_range`, `y_median`, `y_iqr`, `y_rms` |
| **Z-Axis Stats** | `z_mean`, `z_std`, `z_min`, `z_max`, `z_range`, `z_median`, `z_iqr`, `z_rms` |
| **Magnitude Stats** | `mag_mean`, `mag_std`, `mag_min`, `mag_max`, `mag_range`, `mag_median`, `mag_iqr`, `mag_rms` |
| **Correlations** | `xy_corr`, `xz_corr`, `yz_corr` |

### 2.3 Phase 3: Final Feature Engineering
To optimize for on-device inference, the 32 intermediate metrics are refined into a **17-feature vector**. This step ensures the model focuses on the most discriminative signals while maintaining a small memory footprint.

**Final Feature Dataset Headers (`features_final.csv`):**
| Feature Index | Name | Origin / Formula |
| :--- | :--- | :--- |
| `0 - 4` | `X_mean`, `X_std`, `X_max`, `X_min`, `X_sum_sq` | Summary of X-axis motion |
| `5 - 9` | `Y_mean`, `Y_std`, `Y_max`, `Y_min`, `Y_sum_sq` | Summary of Y-axis motion |
| `10 - 14` | `Z_mean`, `Z_std`, `Z_max`, `Z_min`, `Z_sum_sq` | Summary of Z-axis motion |
| `15` | `is_accelerometer` | Binary flag for sensor source |
| `16` | `is_gyroscope` | Binary flag for sensor source |
| **Target** | `label` | Danger (1) or Safe (0) |
| **Extra** | `motion` | Descriptive activity label |

> [!IMPORTANT]
> The `sum_sq` (Sum of Squares) feature is a critical indicator of total energy, calculated as:
> $$E_i = \text{RMS}_i^2 \times 300$$
> *Note: If RMS is unavailable, the fallback calculation is $(\sigma^2 + \mu^2) \times 300$.*

### 2.4 Deployment Pipeline: Unified ONNX Conversion
To minimize latency and dependency overhead in production environments (Mobile/Edge), the trained model and scaler are bundled into a single **ONNX (Opset 13)** pipeline.

-   **Pipeline Architecture:** `StandardScaler` → `LightGBMClassifier`.
-   **Interoperability:** The `.onnx` format allows for cross-platform execution (Android/iOS) using **ONNX Runtime (ort)**.
-   **Verification:** Post-conversion validation is performed using a `(1, 17)` FloatTensor input to ensure parity between the Python environment and the deployment artifact.

---

## 3. Model Architecture and Training

### 3.1 Algorithm: LightGBM
The core classifier is a **LightGBM Classifier**. It was selected over traditional Neural Networks or Random Forests due to its superior performance on tabular sensor data and its ability to handle severe class imbalance natively.

### 3.2 Training Protocol
1.  **Normalization:** A `StandardScaler` is fitted on the training set to ensure zero mean and unit variance.
2.  **Stratified Split:** Data is split into **Training (72%)**, **Validation (8%)**, and **Testing (20%)**.
3.  **Early Stopping:** Training is halted if validation AUC fails to improve for 100 rounds, mitigating overfitting.
4.  **Imbalance Handling:** The `is_unbalance: True` parameter is utilized to increase the penalty for missing a danger event.

---

## 4. Experimental Results and Artifacts

### 4.1 Performance Metrics
The model's performance is quantified using metrics that prioritize user safety:
-   **Recall (Safety Sensitivity):** The primary KPI—ensuring no distress event goes undetected.
-   **Precision (Alarm Reliability):** Minimizing false positives to maintain user trust.
-   **ROC-AUC:** Threshold-independent classification quality.

### 4.2 Project Artifacts (Directory Structure)
All generated artifacts are stored in the `/output_images/` directory:

| Artifact | Purpose |
| :--- | :--- |
| `asfalis_lgb_v1.pkl` | Serialized LightGBM model |
| `asfalis_scaler.pkl` | Serialized StandardScaler |
| `asfalis_sos_pipeline.onnx` | **Unified deployment artifact** (Scaler + Model) |
| `scaler_params.json` | Human-readable mean/scale values for manual implementation |
| `labeled_windows.csv` | Intermediate feature table (Step 1 output) |
| `features.npz` | Training-ready NumPy arrays (Step 2 output) |

---

---

---

## 6. Implementation Guide for Android: Continuous Inference

In a production environment, the application must process a continuous stream of sensor data without ground-truth labels. To handle this, the implementation should move from the "batch" logic used in training to a **Real-Time Sliding Window** approach.

### 6.1 Input Format and Shape
The model expects a **Single Input Tensor**:
- **Name:** `input`
- **Shape:** `[1, 17]` (Batch Size 1, 17 Features)
- **Type:** `FloatTensor`

### 6.2 Continuous Data Flow Strategy
To detect distress events with minimal latency, the following architecture is recommended:

1.  **Circular Buffer (Ring Buffer):** Implement a circular buffer to hold the most recent **300 accelerometer samples**.
2.  **Sliding Window Logic:** Do not wait for 300 *new* samples to run inference. Instead, slide the window.
    - **Step Size ($S$):** Suggested 50 samples.
    - **Inference Trigger:** Every time 50 new samples arrive, compute the 17 features using the *entire* current 300-sample buffer and run the model.
3.  **Feature Extraction:** Calculate the 17 floats (means, stds, max, min, sum_sq) as detailed in Section 2.3.

### 6.3 Post-Inference Logic (Thresholding & Confirmation)
Since the app cannot provide labels, it must interpret the model's output probability:

1.  **Confidence Threshold:** The model outputs a probability $P \in [0, 1]$.
    -   **Suggested Threshold:** $P > 0.85$ (Adjust based on field testing).
2.  **Temporal Confirmation:** To prevent false alarms from brief impacts (e.g., dropping the phone on a table), implement a **"2-of-3" logic**:
    -   Trigger the SOS alert only if at least 2 out of the last 3 inference windows exceed the confidence threshold.
3.  **Hysteresis:** Once a danger state is detected, the app should enter a "High Alert" state for a cooldown period (e.g., 10 seconds) to avoid redundant triggers.

### 6.4 Implementation Example (Kotlin)
```kotlin
// circularBuffer stores last 300 x,y,z points
fun onSensorChanged(event: SensorEvent) {
    circularBuffer.add(event.values)
    
    samplesSinceLastInference++
    if (samplesSinceLastInference >= 50) { // Slide every 50 samples
        val features = extractFeatures(circularBuffer.getAll())
        val probability = runInference(features)
        
        if (confirmDistress(probability)) {
            triggerSOS()
        }
        samplesSinceLastInference = 0
    }
}
```

---

## 7. Discussion and Future Work

## 7. Discussion and Future Work

### 5.1 Handling Real-World Noise
The 300-point window provides enough context to filter out brief accidental drops while remaining responsive enough to trigger an SOS alert within seconds of a sustained impact or snatch event.

### 5.2 Future Enhancements
-   **Gyroscope Integration:** Expanding Phase 3 to include rotational volatility for multi-sensor fusion.
-   **Quantization:** Reducing the ONNX model from Float32 to Int8 to further decrease mobile power consumption.
-   **Kalman Filtering:** Implementing real-time smoothing of raw accelerometer data before windowing to reduce high-frequency jitter.

---
**End of Report**

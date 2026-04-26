# 🚀 AUTO SOS — Production Model Report (LightGBM)

## 📊 Overview: Optimized Gradient Boosting

While previous iterations explored Deep Learning architectures (CNN+LSTM), the production-ready **Asfalis LightGBM** pipeline provides superior performance, faster inference, and a significantly lower memory footprint. 

This model uses **Gradient Boosted Decision Trees (GBDT)** with leaf-wise growth, optimized for the 17-feature motion vectors extracted from raw sensor streams.

---

## 🛠️ Model & Scaler Usage

In the production environment, two primary artifacts are required for inference:

1. **Model (`output_images/asfalis_lgb_v1.pkl`)**: 
   - A trained LightGBM classifier.
   - Responsible for analyzing the extracted motion features and predicting the probability of a distress event.
   - Optimized with `is_unbalance=True` to maximize sensitivity to rare accident events.

2. **Scaler (`output_images/asfalis_scaler.pkl`)**:
   - A `StandardScaler` instance.
   - **Crucial Step**: Raw features must be normalized (zero mean, unit variance) using this exact scaler before being passed to the model. Failing to use the scaler will result in incorrect predictions due to magnitude discrepancies.

---

## 📋 Input Schema (17-Feature Vector)

The model does not take raw sensor readings directly. Instead, it processes a structured 1D array of **17 features** derived from a 300-point window:

| Feature Index | Name | Description |
| :--- | :--- | :--- |
| **0 - 4** | `X Stats` | Mean, Std Dev, Max, Min, and Sum of Squares for X-axis. |
| **5 - 9** | `Y Stats` | Mean, Std Dev, Max, Min, and Sum of Squares for Y-axis. |
| **10 - 14** | `Z Stats` | Mean, Std Dev, Max, Min, and Sum of Squares for Z-axis. |
| **15** | `is_accelerometer` | Binary (1 if window is from accelerometer, 0 otherwise). |
| **16** | `is_gyroscope` | Binary (1 if window is from gyroscope, 0 otherwise). |

---

## 🔄 Raw Data Pre-processing

When receiving raw data (in the format of `.csv` files found in `/new_datapoints`), the following operations must be performed:

1. **Windowing**: Group the raw data into contiguous windows of **300 readings** (approx. 10s at 30Hz).
2. **Feature Extraction**: For each window, calculate the five statistical metrics (Mean, Std, Max, Min, Sum of Squares) for each axis (X, Y, Z).
3. **Encoding**: Identify the sensor type from the `Sensor` column and set the one-hot bits (`is_accelerometer`, `is_gyroscope`).
4. **Standardization**: Pass the resulting 17-element vector through the loaded **Scaler** (`.transform()`).

### Raw Data Format (Reference)
The model pipeline expects CSV data with columns:
`Date, Time, Pid, Tid, Tag, Package, Level, X, Y, Z, Sensor, Value`

---

## 📤 Model Output

The `predict_proba()` method returns a probability distribution:

- **Probability Scope**: `[0.0 to 1.0]` representing the likelihood of a "DANGER" event.
- **Classification**:
    - `0 (SAFE)`: Probability < 0.5 (Normal movement, walking, etc.)
    - `1 (DANGER)`: Probability ≥ 0.5 (Impact, free-fall, or distress signature)

---

## ✅ Latest Performance Metrics (LightGBM)

Based on the most recent evaluation of the specialized LightGBM model:

| Metric | Score |
| :--- | :--- |
| **ROC-AUC** | 0.9733 |
| **Accuracy** | 94.28% |
| **DANGER Precision** | 1.0000 |
| **DANGER Recall** | 0.8667 |
| **F1-Score** | 0.9286 |

> **Production Note**: The model currently maintains **100% precision**, ensuring zero false alarms for the user while capturing over 86% of actual distress events.


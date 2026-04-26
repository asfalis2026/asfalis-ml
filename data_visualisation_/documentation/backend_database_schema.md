# Asfalis Backend Schema Requirements
**Prepared for the Backend Engineering Team**

This document outlines the exact data lifecycle for the Machine Learning pipeline, detailing how sensor data is collected, processed, and structured for continuous retraining.

## 1. The Data Lifecycle Overview

1. **Local Pre-Filtering:** The IoT Bracelet / Android App continuously monitors accelerometer and gyroscope data.
2. **Local Inference (Auto SOS):** When a violent motion is detected locally, the frontend groups 300 readings into a `window` and runs the ONNX model locally.
3. **Triggering SOS:** If the ONNX model outputs DANGER (> 0.6 probability), the frontend calls `POST /api/protection/predict` or `POST /api/sos/trigger` to alert emergency contacts.
4. **Data Collection (RL Loop):** Regardless of the outcome, the 300-reading window is sent to the backend via `POST /api/protection/collect` with an initial label (0 = Safe, 1 = Danger).
5. **Relabeling (Feedback):** If the user cancels an Auto SOS, the frontend sends a feedback request to change the label of those 300 readings from 1 (Danger) to 0 (Safe).
6. **Continuous Training:** A cron job on the backend runs `train_model()`, pulling all verified data from the database, extracting the 17 features mathematically, training a new LightGBM/Random Forest model, and deploying it.

---

## 2. API Payload Schemas (Frontend to Backend)

### A. Background Data Collection (`POST /api/protection/collect`)
The frontend sends **raw sensor arrays**, not pre-computed features. The backend MUST accept the raw arrays so the server-side ML scripts can dynamically extract features.

**JSON Request Body:**
```json
{
  "sensor_type": "accelerometer",
  "label": 1, 
  "data": [
    { "x": -1.2, "y": 9.8, "z": 0.4, "timestamp": 1713993821000 },
    { "x": -1.5, "y": 15.2, "z": 0.1, "timestamp": 1713993821020 },
    // ... exactly 300 readings (~6 seconds at 50Hz)
  ]
}
```
*Note: `label` is an integer where `0` = SAFE and `1` = DANGER.*

### B. Auto-SOS Trigger / Prediction (`POST /api/protection/predict`)
**JSON Request Body:**
```json
{
  "sensor_type": "accelerometer",
  "latitude": 12.9716,
  "longitude": 77.5946,
  "location": "Home",
  "window": [
    [-1.2, 9.8, 0.4],
    [-1.5, 15.2, 0.1]
    // ... exactly 300 readings representing [x, y, z]
  ]
}
```

---

## 3. Required Database Schema (PostgreSQL/SQLAlchemy)

To support the Continuous Learning loop, the backend MUST implement the following table to store the incoming raw data.

**Table Name:** `sensor_training_data`

| Column Name | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | UUID / Integer | Primary Key | Unique row identifier |
| `user_id` | UUID / Integer | Foreign Key, Indexed | The user who generated the data |
| `sensor_type` | VARCHAR | Not Null | 'accelerometer' or 'gyroscope' |
| `timestamp` | BIGINT | Not Null, Indexed | Millisecond Unix timestamp of reading |
| `x` | FLOAT | Not Null | Raw X-axis value |
| `y` | FLOAT | Not Null | Raw Y-axis value |
| `z` | FLOAT | Not Null | Raw Z-axis value |
| `label` | INTEGER | Not Null | 0 = Safe, 1 = Danger |
| `is_verified` | BOOLEAN | Default FALSE | TRUE if the user confirmed or cancelled the SOS, FALSE if it was auto-labeled. |
| `created_at` | TIMESTAMP | Default NOW() | When the record was inserted |

### Why this specific table structure?
The `train_model()` function in the backend's `protection_service.py` runs a SQL query:
`SELECT * FROM sensor_training_data WHERE label IS NOT NULL ORDER BY timestamp`
It chunks these rows into groups of 40-300, reconstructs the `[x, y, z]` arrays, and passes them to `extract_features(window, stype)` to get the 17 statistical features needed for `RandomForestClassifier`. **If the database stores pre-computed features instead of raw X/Y/Z data, the backend ML retraining pipeline will completely break.**

---

## 4. The Feedback Loop (`POST /api/protection/feedback/{alert_id}`)

When the user dismisses an Auto-SOS countdown, the frontend hits the feedback endpoint.

**JSON Request Body:**
```json
{
  "is_false_alarm": true
}
```

**Backend Action Required:**
When `is_false_alarm=true` is received, the backend must find all records in `sensor_training_data` for that `user_id` surrounding the alert's timestamp, and execute:
```sql
UPDATE sensor_training_data 
SET label = 0, is_verified = TRUE 
WHERE user_id = ? AND is_verified = FALSE AND timestamp BETWEEN (alert_time - 5s) AND (alert_time + 5s);
```
This re-labels the violent shaking/fall from a `1` (Danger) to a `0` (Safe), so the ML model learns not to trigger on that specific motion again.

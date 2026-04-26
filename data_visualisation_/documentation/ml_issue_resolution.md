# ML Issue Resolution Documentation

This document outlines the resolutions to the issues raised by the frontend team in `ML_Issues.md`.

## 1. Model Prediction Anomaly (Rigid Fall-Detection)

**Issue summary:** The `asfalis_sos_lgb.onnx` LightGBM model confidently rejected violent cyclic shaking, predicting 0.0000 (SAFE), because the training dataset heavily focused on fall signatures rather than shaking motions.

**Resolution:** 
To unblock immediate end-to-end testing of the SOS trigger, a temporary override was added to the Android frontend.
- Modified `AutoSosManager.kt` to bypass the ML model's threshold validation.
- The `probability` is now hardcoded to `1.0f` after feature extraction completes, ensuring that any violent motion triggering the pre-filter sliding window gate will immediately be treated as a DANGER event.
- **Next steps for the ML team:** Retrain the model using the cyclic shaking datasets (e.g., `moderate_vigorous_shaking.csv`, `strong_vigorous_shaking.csv`) located in the `data_visualisation_/new_datapoints` folder. Once the new model is deployed, the frontend override must be removed.

## 2. API Schema Mismatch on Background Data Collection (422 Errors)

**Issue summary:** The frontend encountered `422 Unprocessable Entity` errors when attempting to send labeled data to the backend via `POST /api/protection/collect`. The frontend was sending 39 pre-computed statistical features, while the backend expected a raw `window` array of `[x, y, z]` sensor readings. 

**Resolution:**
The frontend was incorrectly modified to send the 39 statistical features (`mag_max`, `x_iqr`, etc.) which was causing the backend ingestion to fail. The backend's model retraining pipeline (`train_model()` function) strictly relies on raw `x, y, z` data to dynamically extract its own 17 features. Accepting 39 features would have broken the backend's ML model calibration functionality.

Instead of rewriting the backend database schema and training pipeline to accept the 39 features, the frontend codebase was reverted to match the backend:
- Reverted `SensorTrainingRequest` in `ProtectionDtos.kt` to send the raw sensor `window` array (`List<SensorReading>`) instead of the 39 fields.
- Reverted the caller in `ProtectionRepository.kt` to bypass `AdvancedFeatureExtractor.kt` entirely and pass the raw `window` directly.
- The backend `POST /api/protection/collect` endpoint now correctly accepts the payload, resolving the 422 errors and allowing the continuous learning data collection to function.

> **NOTE**
> The `AdvancedFeatureExtractor.kt` in the frontend was a misunderstanding of the ingestion pipeline. The 39 features (such as those in `labeled_windows.csv`) are meant for offline data exploration and ML model design, not for the backend API's continuous ingestion endpoints.

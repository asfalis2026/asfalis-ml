# ML Integration Flow

**File Responsible for Integration:** `AutoSosManager.kt` (and `SOSDetector.kt` as the model wrapper)

### 1. Handling the Integration
The `AutoSosManager.kt` file acts as the primary orchestrator between the raw sensor data collection, the feature extraction, and the machine learning model. It is the core pipeline that connects the `SensorEventListener` to the ONNX Runtime model.

### 2. Passing Features to the Model
Within `AutoSosManager.kt`, there is a method called `runLocalInference(...)`. This is where the 300-point window is passed to the feature extractor, and the resulting features are immediately fed into the ML model for prediction. 

Here is the exact snippet illustrating this data flow:

```kotlin
// Inside AutoSosManager.kt

private fun runLocalInference(
    snapshot: List<List<Float>>,
    rawSnapshot: List<SensorReading>
) {
    scope.launch {
        try {
            // 1. EXTRACT: Pass the 300-point snapshot to get the 17 features
            val features = FeatureExtractor.extract(snapshot, activeSensorType)
            
            // 2. PREDICT: Pass the calculated features directly into the ML model
            // The SOSDetector class wraps the ONNX Runtime model inference.
            val probability = sosDetector.predictDanger(features)
            
            // 3. EVALUATE: Check if probability exceeds the 0.60 threshold
            if (sosDetector.shouldTriggerSOS(probability)) {
                // Trigger SOS on backend...
            } else {
                // Safe...
            }
        } catch (e: Exception) {
            // Error handling
        }
    }
}
```

### File Roles Summary
* **`AutoSosManager.kt`**: The orchestrator. It manages the sliding window, decides *when* to run inference (every 50 readings), calls the feature extractor, and then feeds the features to the model.
* **`SOSDetector.kt`**: The ML Wrapper. It initializes the ONNX environment, loads the LightGBM model from the app's assets, takes the 17-feature array passed by `AutoSosManager`, and returns the danger probability (0.0 to 1.0).
* **`AdvancedFeatureExtractor.kt` / `FeatureExtractor.kt`**: The mathematical engine. It takes the raw 300-point window and computes the required statistical features (mean, std, max, min, sum_sq).

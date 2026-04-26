# 📲 Android Integration Guide: On-Device Distress Detection

This document provides a step-by-step technical blueprint for Android (Kotlin) developers to implement the **Asfalis SOS** Machine Learning model directly on the mobile frontend.

---

## 🛠️ 1. Environmental Setup

To execute the LightGBM model on Android, we use the **ONNX Runtime**, which is highly optimized for mobile hardware acceleration (CPU/NNAPI).

### Add Dependency
In your `app/build.gradle` file:
```gradle
dependencies {
    // Standard ONNX Runtime for Android
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:latest.release'
    
    // GSON for parsing scaling parameters
    implementation 'com.google.code.gson:gson:2.10.1'
}
```

---

## 📦 2. Asset Preparation

Place the following two artifacts in your project's `src/main/assets/` folder:

1.  **`asfalis_sos_lgb.onnx`**: The compiled LightGBM model.
2.  **`scaler_params.json`**: The normalization constants (Mean & Scale).

---

## 🔄 3. Implementation Workflow

### Step 1: Real-time Windowing logic
The model requires **300 snapshots** of sensor data per inference. You must maintain a sliding window buffer.

```kotlin
class SensorBuffer {
    val windowSize = 300
    private val bufferX = mutableListOf<Float>()
    private val bufferY = mutableListOf<Float>()
    private val bufferZ = mutableListOf<Float>()

    fun addReading(x: Float, y: Float, z: Float) {
        if (bufferX.size >= windowSize) {
            bufferX.removeAt(0)
            bufferY.removeAt(0)
            bufferZ.removeAt(0)
        }
        bufferX.add(x)
        bufferY.add(y)
        bufferZ.add(z)
    }

    fun isReady(): Boolean = bufferX.size == windowSize
}
```

### Step 2: Feature Extraction (The 17-Vector)
Once the buffer hits 300 readings, calculate the 17 statistical features required by the model:

| Feature Index | Calculation |
| :--- | :--- |
| 0 - 4 | X-axis: Mean, Std, Max, Min, Sum of Squares |
| 5 - 9 | Y-axis: Mean, Std, Max, Min, Sum of Squares |
| 10 - 14 | Z-axis: Mean, Std, Max, Min, Sum of Squares |
| 15 | `is_accelerometer` (Set to 1.0f) |
| 16 | `is_gyroscope` (Set to 0.0f) |

### Step 3: Pre-processing (Z-Score Normalization)
Before running inference, you **must** scale the features using the `mean` and `scale` values from `scaler_params.json`:

```kotlin
// Formula: normalized = (raw - mean) * scale
val normalizedFeatures = FloatArray(17) { i ->
    (extractedFeatures[i] - scalerParams.mean[i]) * scalerParams.scale[i]
}
```

### Step 4: ONNX Inference
Initialize the session once and reuse it for efficiency.

```kotlin
// 1. Setup Environment & Session
val env = OrtEnvironment.getEnvironment()
val modelBytes = assets.open("asfalis_sos_lgb.onnx").readBytes()
val session = env.createSession(modelBytes)

// 2. Create Input Tensor (Shape: 1 x 17)
val shape = longArrayOf(1, 17)
val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(normalizedFeatures), shape)

// 3. Run and Parse
val results = session.run(mapOf("input" to inputTensor))
val outputProbas = results[1].value as Array<FloatArray>

val dangerProbability = outputProbas[0][1] // Probability of "DANGER" class
```

---

## ⚡ 4. Best Practices for Mobile

### 🔋 Power Management
- **Sampling Frequency**: The model is trained on data sampled at approximately **30Hz**. Lowering this significantly may reduce accuracy.
- **Batch Inference**: Do not run inference on every single sensor tick. Use an **overlap strategy** (e.g., run inference once every 150 new readings).

### ⚙️ Performance Optimization
- **Background Service**: Use a **Foreground Service** with a partial wake lock to ensure SOS detection continues when the screen is off.
- **Model Lifecycle**: Close the `OrtSession` and `OrtEnvironment` when the monitoring service is stopped to prevent memory leaks.

### 🚨 Thresholding
| Probability | Action |
| :--- | :--- |
| **< 0.50** | Safe: Log and continue. |
| **0.50 - 0.75** | Caution: Potential anomalous movement. |
| **> 0.85** | **Critical Danger**: Trigger SOS sequence immediately. |

---

> **Note to Android Lead**: The model is converted with `zipmap=False`, meaning the output is a raw `FloatArray`. This avoids the high overhead of Java Map objects during inference.

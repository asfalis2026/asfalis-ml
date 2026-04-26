# Sensor Sliding Window & Feature Extraction Logic

Below is the complete Kotlin implementation demonstrating how the Android application handles continuous sensor reading, the 300-point sliding window, and the subsequent feature extraction required by the backend streaming architecture.

### Implementation Overview

1. **Continuous Reading & Sliding Window**: The `SensorEventListener` continuously receives data. Once 300 points are collected, the 301st point pushes out the 1st (oldest) point, maintaining a fixed-size `ArrayDeque`.
2. **Feature Calculation**: The `AdvancedFeatureExtractor` calculates the required 17 features (including mean, standard deviation, minimum, maximum, and sum of squares for X, Y, and Z).
3. **Execution**: Feature calculation correctly triggers on the newest window of 300 points dynamically.

```kotlin
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import kotlin.math.sqrt

/**
 * Data class representing a single [x, y, z] sensor reading.
 */
data class SensorReading(
    val x: Float,
    val y: Float,
    val z: Float,
    val timestamp: Long
)

/**
 * Data class representing the 17 computed statistical features.
 */
data class SensorTrainingRequest(
    val sensorType: String,
    val label: Int,
    val window: List<List<Float>>?,

    val xMean: Float, val xStd: Float, val xMax: Float, val xMin: Float, val xSumSq: Float,
    val yMean: Float, val yStd: Float, val yMax: Float, val yMin: Float, val ySumSq: Float,
    val zMean: Float, val zStd: Float, val zMax: Float, val zMin: Float, val zSumSq: Float,

    val isAccelerometer: Float,
    val isGyroscope: Float
)

/**
 * Manages the continuous reading of sensor data and maintains the sliding window.
 */
class SensorSlidingWindowManager : SensorEventListener {

    companion object {
        // We maintain exactly 300 readings for the feature calculation window.
        private const val WINDOW_SIZE = 300
        
        // We calculate features and run inference every INFERENCE_STRIDE readings 
        // to prevent overlapping computational overload while still being highly responsive.
        private const val INFERENCE_STRIDE = 50 
    }

    /**
     * Sliding Window Buffer:
     * We use an ArrayDeque to efficiently maintain the 300-point sliding window.
     */
    private val slidingWindowBuffer = ArrayDeque<SensorReading>(WINDOW_SIZE + 1)
    
    // Tracks how many new readings have arrived since the last feature calculation
    private var readingsSinceLastCalculation = 0

    /**
     * 1. Continuous reading of sensor data (x, y, z values).
     */
    override fun onSensorChanged(event: SensorEvent) {
        val x = event.values[0]
        val y = event.values[1]
        val z = event.values[2]

        val reading = SensorReading(x, y, z, System.currentTimeMillis())

        // 3. Sliding Window Behavior (301st Reading Case)
        // Check if the buffer is full (already has 300 points)
        if (slidingWindowBuffer.size >= WINDOW_SIZE) {
            // YES: The first (oldest) data point is explicitly discarded.
            slidingWindowBuffer.removeFirst()
        }
        
        // The newest data point is appended to the end of the window.
        slidingWindowBuffer.addLast(reading)
        readingsSinceLastCalculation++

        // 2. Feature Calculation After 300 Readings
        // Confirming the window has reached the required 300 points
        if (slidingWindowBuffer.size == WINDOW_SIZE && readingsSinceLastCalculation >= INFERENCE_STRIDE) {
            
            // Reset the stride counter
            readingsSinceLastCalculation = 0

            // YES: The feature calculations are updated again using the LATEST 300 data points.
            // A snapshot of the current 300 points is passed to the extractor.
            val currentWindowSnapshot = slidingWindowBuffer.toList()
            
            // Perform feature extraction and inference
            val features = AdvancedFeatureExtractor.extractFeatures(
                window = currentWindowSnapshot,
                sensorType = "accelerometer",
                dangerLabel = 1 // or 0 based on context
            )
            
            // -> [Send features to backend or use in local ML model]
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Not needed for this implementation
    }
}

/**
 * Extracts the required statistical features from the 300-point window.
 */
object AdvancedFeatureExtractor {

    /**
     * Calculates: mean, std, min, max, and sum of squares for X, Y, and Z.
     */
    fun extractFeatures(
        window: List<SensorReading>,
        sensorType: String,
        dangerLabel: Int
    ): SensorTrainingRequest {
        
        // Map the data into separate arrays for X, Y, and Z
        val xs = window.map { it.x }.toFloatArray()
        val ys = window.map { it.y }.toFloatArray()
        val zs = window.map { it.z }.toFloatArray()

        // Construct the 17-feature payload
        return SensorTrainingRequest(
            sensorType = sensorType,
            label = dangerLabel,
            window = window.map { listOf(it.x, it.y, it.z) },

            // For X: mean, standard deviation, minimum, maximum, sum of squares
            xMean = mean(xs), 
            xStd = std(xs), 
            xMax = xs.maxOrNull() ?: 0f, 
            xMin = xs.minOrNull() ?: 0f, 
            xSumSq = sumSq(xs),

            // For Y: mean, standard deviation, minimum, maximum, sum of squares
            yMean = mean(ys), 
            yStd = std(ys), 
            yMax = ys.maxOrNull() ?: 0f, 
            yMin = ys.minOrNull() ?: 0f, 
            ySumSq = sumSq(ys),

            // For Z: mean, standard deviation, minimum, maximum, sum of squares
            zMean = mean(zs), 
            zStd = std(zs), 
            zMax = zs.maxOrNull() ?: 0f, 
            zMin = zs.minOrNull() ?: 0f, 
            zSumSq = sumSq(zs),

            isAccelerometer = if (sensorType.equals("accelerometer", ignoreCase = true)) 1f else 0f,
            isGyroscope = if (sensorType.equals("gyroscope", ignoreCase = true)) 1f else 0f
        )
    }

    // Helper Math Functions

    private fun mean(arr: FloatArray): Float {
        if (arr.isEmpty()) return 0f
        return arr.average().toFloat()
    }

    private fun std(arr: FloatArray): Float {
        if (arr.size < 2) return 0f
        val m = mean(arr)
        val variance = arr.map { (it - m) * (it - m) }.average().toFloat()
        return sqrt(variance)
    }

    private fun sumSq(arr: FloatArray): Float {
        return arr.map { (it * it).toDouble() }.sum().toFloat()
    }
}
```

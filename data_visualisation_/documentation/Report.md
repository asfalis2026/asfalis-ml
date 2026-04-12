# AUTO SOS — Deep Learning Deep Dive Report

## 🚀 Overview: Moving Beyond Manual Features

The traditionally engineered machine learning pipeline required a 17-feature extraction process (calculating variance, max, min, etc., manually). **Step 4** introduces an end-to-end Deep Learning architecture that processes **RAW sensor data**.

Instead of a human deciding that "variance" is important, the model uses **Convolutional Neural Network (CNN)** layers to "invent" its own feature filters directly from the accelerometer's X, Y, and Z axes.

---

## 🧠 Model Architecture: CNN + LSTM

The model, `AutoSOS_DeepNet`, combines two powerful neural architectures:

### 1. Spatial Feature Extraction (1D CNN)

* **Layer 1 (Conv1D)**: 32 filters, kernel size 5. It slides across the 300-point signal to find brief, sharp anomalies like sudden impacts or spikes.
* **Layer 2 (Conv1D)**: 64 filters, kernel size 5. It looks for wider, more complex patterns (like the characteristic arc of a free-fall).
* **Batch Normalization & MaxPool**: These stabilize the signal and reduce the sequence length, distilling the raw data into high-level features.

### 2. Temporal Context (LSTM)

* **Hidden State (64 units)**: The LSTM processes the sequence of features extracted by the CNN.
* **Why it works**: It learns the *order* of events. For example, it can recognize that a period of zero-G (free-fall) immediately followed by high acceleration (impact) is a high-probability "DANGER" event.

---

## 🛠️ Optimization & Training Stability

The pipeline includes several advanced mechanisms to ensure reliable convergence, especially on small datasets:

* **AdamW Optimizer**: Uses decoupled weight decay (`1e-4`) to prevent overfitting while maintaining better weight regularization than standard Adam.
* **Adaptive Learning Rate**: Utilizing `ReduceLROnPlateau`, the model automatically cuts the learning rate in half (`factor=0.5`) if validation loss stalls for more than 5 epochs.
* **Early Stopping**: Training monitors validation loss and stops (`patience=15`) if the model starts to diverge or overfit, ensuring the saved model is always the "best" version.
* **Gradient Clipping**: Gradients are normalized to a `max_norm=0.5` to prevent numerical instability (exploding gradients) during the LSTM update steps.

---

## 📊 Data Pipeline

1. **Raw Windowing**: Segmenting each sensor record into `300` point windows (roughly 10 seconds of data if sampled at 30Hz).
2. **Standardization**: Every window is scaled using a `StandardScaler` fitted on the training data to ensure all sensor axes share a common range.
3. **Imbalance Handling**: Automatically calculates `pos_weight` (Safe counts / Danger counts) to punish the model more for missing distress signals in imbalanced datasets.

---

## ✅ Latest Performance Metrics

Based on the most recent evaluation:

| Metric                     | Score  |
| :------------------------- | :----- |
| **ROC-AUC**          | 0.8900 |
| **F1-Score**         | 0.8462 |
| **DANGER Recall**    | 0.73   |
| **DANGER Precision** | 1.00   |

> **Note**: The model currently has **100% precision** on danger detections, meaning there are zero false positives. To improve safety, future iterations may focus on increasing **Recall** (lowering the decision threshold) to ensure no distress signals are missed.


# AUTO SOS — ADVANCED MODEL TRAINING REPORT

## Configuration
- **Algorithms Tested:** Random Forest, XGBoost, LightGBM, AdaBoost
- **Window Size:** 300 readings (overlapping, 50 % overlap)
- **Feature Dimension:** 17 features per window
- **Train / Test Split:** 80 % / 20 %

## Dataset Summary
| Metric        | Value         |
|---------------|---------------|
| Total samples | 174  |
| Train samples | 139 |
| Test samples  | 35  |
| Features      | 17     |
| SAFE labels   | 101 |
| DANGER labels | 73 |

## Model Comparison

       Model Accuracy Precision Recall F1-Score ROC-AUC Avg-Prec  Train(s)
RandomForest   0.9429    1.0000 0.8667   0.9286  0.9633   0.9718      0.12
     XGBoost   0.9429    0.9333 0.9333   0.9333  0.9300   0.9575      0.14
    LightGBM   0.9714    1.0000 0.9333   0.9655  0.9867   0.9860      0.15
    AdaBoost   0.9714    1.0000 0.9333   0.9655  0.9800   0.9810      0.12

**⭐ Best Model (by ROC-AUC): LightGBM**

## Detailed Metrics — LightGBM
| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.9714 |
| Precision | 1.0000 |
| Recall    | 0.9333    |
| F1-Score  | 0.9655        |
| ROC-AUC   | 0.9867       |
| Avg-Prec  | 0.9860  |

## Confusion Matrix (LightGBM)
```
                    Predicted-Safe  Predicted-Danger
Actual-Safe:          20                  0
Actual-Danger:         1                 14
```

## Sensitivity Configuration

| Level  | Threshold | Accuracy | Precision | Recall | False-Alarm | Miss-Rate |
|--------|-----------|----------|-----------|--------|-------------|-----------|
| HIGH   | 0.35      | 0.9714   | 1.0000    | 0.9333  | 0.0000       | 0.0667     |
| MEDIUM | 0.60      | 0.9714   | 1.0000    | 0.9333  | 0.0000       | 0.0667     |
| LOW    | 0.85      | 0.9714   | 1.0000    | 0.9333  | 0.0000       | 0.0667     |

- **HIGH** — Medical / elderly care: catch every danger, tolerate false alarms
- **MEDIUM** — General population: balanced precision/recall
- **LOW** — Active adults: minimise false alarms

## Algorithm Analysis

### Random Forest
- **Type:** Bagging ensemble of decision trees
- **Pros:** Robust to noise, no feature scaling required, low variance, interpretable importances
- **Cons:** Slower inference than boosting models on large feature sets
- **Best for:** Production systems where reliability > raw accuracy

### XGBoost
- **Type:** Sequential gradient-boosted trees (regularised)
- **Pros:** Handles imbalance via `scale_pos_weight`, strong OOB regularisation, fast C++ backend
- **Cons:** More hyperparameters to tune, higher memory than LightGBM
- **Best for:** General-purpose fall detection with moderate dataset sizes

### LightGBM
- **Type:** Leaf-wise gradient boosting (histogram-based)
- **Pros:** Fastest training & inference, lowest RAM footprint, native imbalance handling
- **Cons:** Can overfit on very small datasets without careful `min_child_samples` tuning
- **Best for:** Real-time edge / mobile inference where speed is critical

### AdaBoost
- **Type:** Sequential re-weighting of weak learners (SAMME algorithm)
- **Pros:** Simple, interpretable; performs well when base learner captures key patterns
- **Cons:** Slower than gradient boosting variants, sensitive to noisy labels
- **Best for:** Lightweight deployment where a simpler model is preferred


## Visualisations Generated
| File | Description |
|------|-------------|
| `10_model_comparison.png`          | Grouped bar chart — all metrics |
| `10b_radar_comparison.png`         | Radar / spider chart            |
| `11_all_models_roc.png`            | ROC curves — all models         |
| `12_best_model_confusion_matrix.png` | Best model confusion matrix   |
| `13_best_model_distribution.png`   | Probability distribution        |
| `14_training_time.png`             | Training speed comparison       |

## Next Steps
1. ✅ Model trained & saved → `auto_sos_model_LightGBM.pkl`
2. → Integrate **LightGBM** into `/api/protection/predict`
3. → Test on real-world sensor streams
4. → Monitor false-alarm rate in production
5. → Retrain monthly with new labelled data

# Q1 – Classifier Comparison on HTRU2 (Pulsar Star Detection)

Dataset: [UCI HTRU2 (id 372)](https://archive.ics.uci.edu/dataset/372/htru2) – 17 898 pulsar candidates from the
High Time Resolution Universe survey. Binary target: pulsar (1) vs non-pulsar (0).

## Task

| Part | Requirement | Where in `q1_classifiers.py` |
|------|-------------|------------------------------|
| a | KNN with cross-validation | `GridSearchCV` over k = 1, 3, …, 31 (5-fold) |
| b | Decision Tree | `GridSearchCV` over `max_depth` = 1–20 (5-fold) |
| c | SVM with RBF kernel | `SVC(kernel="rbf", C=1.0, gamma="scale")` |
| d | Deep CNN (bonus) | `CNN` class (PyTorch, 1-D convolutions) |
| i | 80 % train / 20 % test split | stratified `train_test_split` |
| ii | Confusion matrix | `evaluate()` → `cm_<model>.png` |
| iii | Accuracy, Precision, Recall, F1-score | `evaluate()` → `q1_results.csv`, `metrics_comparison.png` |
| iv | Analyse overfitting and model complexity | train vs test sweeps → `overfitting_complexity.png` |
| v | ROC curve analysis | `roc_curves.png`, best model by AUC |

## Data

- File: `data/HTRU_2.csv` (extracted from `htru2.zip`), no header.
- **8 features:** mean, std, excess kurtosis and skewness of the *integrated pulse profile* (`mean_ip`, `std_ip`,
  `kurt_ip`, `skew_ip`) and of the *DM-SNR curve* (`mean_dmsnr`, `std_dmsnr`, `kurt_dmsnr`, `skew_dmsnr`).
- **Class imbalance:** 16 259 non-pulsars vs 1 639 pulsars (~9 %). Accuracy alone is misleading, so precision,
  recall, F1 and ROC-AUC are reported as well.
- **Split:** stratified 80/20 (`random_state=42`) → 14 318 train, 3 580 test (328 pulsars in test).
- Features standardised with `StandardScaler` fit on the training set only.

## Models

**a. KNN.** 5-fold CV over odd k from 1 to 31 → **best k = 11** (CV accuracy 0.978).

**b. Decision Tree.** 5-fold CV over `max_depth` 1–20 → **best depth = 4** (an unrestricted tree overfits).

**c. SVM (RBF).** `C = 1`, `gamma = "scale"`. ROC uses `decision_function` scores.

**d. CNN (bonus).** The 8 standardised features are treated as a 1-channel 1-D signal of length 8:

```
Conv1d(1→32, k=3) → ReLU → Conv1d(32→64, k=3) → ReLU → MaxPool(2)
→ Conv1d(64→64, k=3) → ReLU → Flatten → Linear(256→64) → ReLU → Dropout(0.3) → Linear(64→1)
```

Loss `BCEWithLogitsLoss`, Adam (lr 0.001), 30 epochs, batch size 64. Threshold 0.5 on the sigmoid output.

## Results (test set)

| Model | Train Acc | Test Acc | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|----------|-----------|--------|----|---------|
| KNN (k = 11) | 0.9795 | **0.9810** | 0.9392 | 0.8476 | **0.8910** | 0.9541 |
| Decision Tree (depth 4) | 0.9808 | 0.9779 | 0.9220 | 0.8293 | 0.8732 | 0.9698 |
| SVM RBF | 0.9795 | 0.9807 | **0.9481** | 0.8354 | 0.8882 | 0.9542 |
| CNN | 0.9821 | 0.9802 | 0.9003 | **0.8811** | 0.8906 | **0.9761** |

Confusion matrices (rows = actual, columns = predicted, order non-pulsar / pulsar):

| Model | TN | FP | FN | TP |
|-------|----|----|----|----|
| KNN | 3234 | 18 | 50 | 278 |
| Decision Tree | 3229 | 23 | 56 | 272 |
| SVM RBF | 3237 | 15 | 54 | 274 |
| CNN | 3220 | 32 | 39 | 289 |

**Comparison.** All four models reach ~98 % accuracy, so accuracy does not separate them. The differences are in
the minority class:

- SVM has the highest precision (fewest false alarms).
- CNN has the highest recall – it misses the fewest real pulsars (39 vs 50–56) – and the highest ROC-AUC.
- KNN has the best accuracy and F1 by a very small margin.
- The shallow Decision Tree is the weakest on every threshold metric.

## Overfitting and model complexity

| Model | Low complexity | Chosen | High complexity |
|-------|----------------|--------|-----------------|
| KNN (smaller k = more complex) | k = 31 | k = 11: train 0.9795 / test 0.9810 | k = 1: train **1.0000** / test 0.9729 |
| Decision Tree (deeper = more complex) | depth 2: train 0.9772 / test 0.9796 | depth 4 | depth 25: train **1.0000** / test 0.9684 |
| SVM (larger gamma = more complex) | gamma 0.001: train 0.9723 / test 0.9760 | gamma 1: train 0.9833 / test 0.9830 | gamma 100: train **0.9992** / test 0.9084 |

In every model, high complexity drives training accuracy to ~100 % while test accuracy drops – classic
overfitting. Very low complexity lowers both (underfitting). Cross-validation picks the middle of the curve.
The CNN shows a small train/test gap (0.9821 vs 0.9802), helped by dropout; its loss curve is in `cnn_loss.png`.

## ROC curve analysis

| Model | AUC |
|-------|-----|
| **CNN** | **0.9761** |
| Decision Tree | 0.9698 |
| SVM RBF | 0.9542 |
| KNN | 0.9541 |

All curves are far above the random diagonal. **Best classifier by ROC-AUC: CNN (0.9761)** – it ranks pulsars above
non-pulsars most consistently across all thresholds. ROC-AUC is threshold-independent, so it is a better summary
than accuracy for this imbalanced data.

## Output files (`outputs/`)

| File | Content |
|------|---------|
| `q1_results.csv` | all metrics per model |
| `cm_KNN.png`, `cm_Decision_Tree.png`, `cm_SVM_RBF.png`, `cm_CNN.png` | confusion matrices |
| `metrics_comparison.png` | bar chart of test metrics |
| `overfitting_complexity.png` | train vs test accuracy vs k / depth / gamma |
| `cnn_loss.png` | CNN training loss per epoch |
| `roc_curves.png` | ROC curves of all four models |

## How to run

```bash
pip install numpy pandas scikit-learn matplotlib seaborn torch
cd ml-project-1
python q1_classifiers.py
```

The script uses relative paths (`data/HTRU_2.csv`, `outputs/`), so run it from inside `ml-project-1`.
Runtime: a few minutes (most of it is the SVM gamma sweep and the CNN).

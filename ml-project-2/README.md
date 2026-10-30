# Q2 – Regression Model Comparison on the UCI Air Quality Dataset

Dataset: [UCI Air Quality (id 360)](https://archive.ics.uci.edu/dataset/360/air) – hourly readings from a
gas multisensor device deployed in an Italian city (March 2004 – February 2005).

## Task

Predict **CO(GT)** (true hourly CO concentration, mg/m³) from the sensor and weather readings, and compare:

| Part | Requirement | Where in `air_quality_regression.py` |
|------|-------------|--------------------------------------|
| a | Linear Regression with gradient descent | `LinearRegressionGD` class (from scratch, batch GD on MSE) |
| b | Polynomial Regression | `GridSearchCV` over degree 2–4 |
| c | Ridge and Lasso Regression | `Ridge` / `Lasso` on degree-4 polynomial features |
| i | Evaluate with MSE, RMSE, R² | `metrics()` + results table + `metric_comparison.png` |
| ii | Analyse underfitting vs overfitting | degree sweep 1–5 + single-feature curve-fit plot |
| iii | Use cross-validation to select regularisation parameter | 5-fold `GridSearchCV` over `alpha` |

## Data preparation

- File: `AirQualityUCI.csv` (`;` separator, `,` decimal). Empty trailing columns dropped.
- **Features (8):** `PT08.S1(CO)`, `PT08.S2(NMHC)`, `PT08.S3(NOx)`, `PT08.S4(NO2)`, `PT08.S5(O3)`, `T`, `RH`, `AH`.
- **Target:** `CO(GT)`.
- `-200` marks a missing value and is replaced with NaN.
- Rows without a target are dropped; rows where every sensor is missing are dropped.
- Remaining feature gaps are filled with the **median** (fit on the training fold only, inside a `Pipeline`).
- Features are standardised. Polynomial features are standardised again after expansion.
- **Split:** 80 % train / 20 % test (`random_state=42`) → 5875 train, 1469 test samples.

## Models

**a. Linear Regression (GD).** Bias column + weights initialised at 0, learning rate 0.1, stops when the MSE change
is below `1e-9` (max 5000 iterations). Converges in ~880 iterations (`gd_convergence.png`).

**b. Polynomial Regression.** Ordinary least squares on polynomial features. Degree chosen by 5-fold CV from {2, 3, 4}
→ **degree 3**.

**c. Ridge / Lasso.** Applied to degree-4 features – the degree where plain polynomial regression starts to overfit –
so regularisation has something to fix.

- Ridge `alpha` grid: `logspace(-2, 4, 13)` → best **alpha = 1**
- Lasso `alpha` grid: `logspace(-5, -2, 7)` → best **alpha ≈ 3.16e-5**

## Results (test set)

| Model | Train MSE | Test MSE | Test RMSE | Test R² |
|-------|-----------|----------|-----------|---------|
| Linear Regression (GD) | 0.2410 | 0.1997 | 0.4468 | 0.8977 |
| Polynomial (deg = 3) | 0.1578 | 0.1501 | 0.3875 | 0.9231 |
| **Ridge (deg = 4, α = 1)** | 0.1411 | **0.1472** | **0.3837** | **0.9246** |
| Lasso (deg = 4, α = 3.16e-5) | 0.1416 | 0.1484 | 0.3852 | 0.9240 |

**Best model: Ridge (degree 4).** Full table in `outputs/regression_results.csv`.

## Underfitting vs overfitting

Unregularised polynomial degree sweep:

| Degree | Train MSE | Test MSE | Regime |
|--------|-----------|----------|--------|
| 1 | 0.2410 | 0.1997 | underfitting (high bias) |
| 2 | 0.1782 | 0.1628 | |
| 3 | 0.1578 | 0.1501 | best balance |
| 4 | 0.1277 | 0.1597 | overfitting begins |
| 5 | 0.0912 | 0.4160 | strong overfitting (high variance) |

Train error keeps falling as complexity grows, while test error has a U-shape with the minimum at degree 3.
Ridge and Lasso take the overfitting degree-4 model and pull its test MSE (0.1597) down to ~0.147,
better than any unregularised model.

## Output files (`outputs/`)

| File | Content |
|------|---------|
| `regression_results.csv` | train/test MSE, RMSE, R² per model |
| `metric_comparison.png` | bar charts of test MSE, RMSE, R² |
| `underfitting_overfitting.png` | train vs test MSE over polynomial degree |
| `fit_curves_under_over.png` | degree 1 / 3 / 15 fits on one feature (visual under/over-fit) |
| `cv_alpha_selection.png` | train vs 5-fold validation MSE over alpha, for Ridge and Lasso |
| `gd_convergence.png` | gradient descent training MSE per iteration |

## How to run

```bash
pip install numpy pandas scikit-learn matplotlib
python air_quality_regression.py
```

Paths are resolved relative to the script, so it can be run from any directory. Runtime: under a minute.

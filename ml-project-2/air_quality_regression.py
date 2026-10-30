"""
Q2: Regression model comparison on the UCI Air Quality dataset (target: CO(GT)).

Models
  a. Linear Regression with gradient descent (implemented from scratch)
  b. Polynomial Regression (degree chosen by cross-validation)
  c. Ridge and Lasso Regression (on polynomial features, alpha chosen by
     cross-validation)

Implementation details covered
  i)   MSE, RMSE and R^2 for every model, compared in a table and a bar chart
  ii)  Underfitting / overfitting shown with a polynomial-degree sweep and
       a single-feature curve-fit picture
  iii) 5-fold cross-validation selects the regularisation parameter alpha

Run:  python air_quality_regression.py
"""
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# very small Lasso alphas converge slowly; harmless for the CV comparison
# (env var silences the same warning inside the parallel CV worker processes)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
os.environ["PYTHONWARNINGS"] = "ignore"
SEED = 42
HERE = Path(__file__).resolve().parent
CSV = HERE / "AirQualityUCI.csv"
OUT = HERE / "outputs"
FEATURES = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)",
            "PT08.S5(O3)", "T", "RH", "AH"]
TARGET = "CO(GT)"
REG_DEGREE = 4          # degree where plain polynomial starts to overfit -> regularise it


# ------------------------------------------------- a. Linear Regression (GD)
class LinearRegressionGD:
    """Batch gradient descent minimising the MSE cost."""

    def __init__(self, learning_rate=0.1, n_iter=5000, tol=1e-9):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        Xb = np.c_[np.ones(len(X)), X]                 # add bias column
        self.theta_ = np.zeros(Xb.shape[1])
        self.loss_history_ = []
        for _ in range(self.n_iter):
            err = Xb @ self.theta_ - y
            self.loss_history_.append(np.mean(err ** 2))
            grad = (2 / len(y)) * Xb.T @ err
            self.theta_ -= self.learning_rate * grad
            if len(self.loss_history_) > 1 and \
                    abs(self.loss_history_[-2] - self.loss_history_[-1]) < self.tol:
                break
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        return np.c_[np.ones(len(X)), X] @ self.theta_

    # needed so the model can sit inside an sklearn Pipeline
    def get_params(self, deep=True):
        return {"learning_rate": self.learning_rate, "n_iter": self.n_iter, "tol": self.tol}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self


# ------------------------------------------------------------------ data
def load_data():
    df = pd.read_csv(CSV, sep=";", decimal=",").dropna(axis=1, how="all")
    df = df[FEATURES + [TARGET]].replace(-200, np.nan)   # -200 marks missing
    df = df.dropna(subset=[TARGET])                       # need a label
    df = df[df[FEATURES].notna().any(axis=1)]            # drop full sensor outages
    return df[FEATURES], df[TARGET]


def make_pipe(model, degree=1):
    steps = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    if degree > 1:
        steps += [("poly", PolynomialFeatures(degree, include_bias=False)),
                  ("sc2", StandardScaler())]
    return Pipeline(steps + [("reg", model)])


def metrics(model, X, y):
    p = model.predict(X)
    mse = mean_squared_error(y, p)
    return mse, np.sqrt(mse), r2_score(y, p)


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(exist_ok=True)
    X, y = load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    print(f"Samples: {len(X)}  (train {len(Xtr)}, test {len(Xte)}), features: {len(FEATURES)}\n")

    # a) Linear regression with gradient descent
    gd = make_pipe(LinearRegressionGD()).fit(Xtr, ytr)
    print(f"Linear GD converged in {len(gd['reg'].loss_history_)} iterations")

    # b) Polynomial regression: degree picked by CV
    poly_gs = GridSearchCV(make_pipe(LinearRegression(), 2),
                           {"poly__degree": [2, 3, 4]}, cv=cv, n_jobs=-1,
                           scoring="neg_mean_squared_error").fit(Xtr, ytr)
    deg = poly_gs.best_params_["poly__degree"]
    print(f"Polynomial: best degree by CV = {deg}")

    # c) Ridge and Lasso: alpha picked by CV  (iii)
    ridge_gs = GridSearchCV(make_pipe(Ridge(), REG_DEGREE),
                            {"reg__alpha": np.logspace(-2, 4, 13)}, cv=cv, n_jobs=-1,
                            scoring="neg_mean_squared_error", return_train_score=True).fit(Xtr, ytr)
    lasso_gs = GridSearchCV(make_pipe(Lasso(max_iter=5000, tol=1e-3), REG_DEGREE),
                            {"reg__alpha": np.logspace(-5, -2, 7)}, cv=cv, n_jobs=-1,
                            scoring="neg_mean_squared_error", return_train_score=True).fit(Xtr, ytr)
    ra, la = ridge_gs.best_params_["reg__alpha"], lasso_gs.best_params_["reg__alpha"]
    print(f"Ridge: best alpha by CV = {ra:.4g}")
    print(f"Lasso: best alpha by CV = {la:.4g}")

    # i) MSE, RMSE, R^2 comparison
    models = {"Linear Regression (GD)": gd,
              f"Polynomial (deg={deg})": poly_gs.best_estimator_,
              f"Ridge (deg={REG_DEGREE}, alpha={ra:.3g})": ridge_gs.best_estimator_,
              f"Lasso (deg={REG_DEGREE}, alpha={la:.3g})": lasso_gs.best_estimator_}
    rows = []
    for name, m in models.items():
        tr, te = metrics(m, Xtr, ytr), metrics(m, Xte, yte)
        rows.append({"Model": name, "Train MSE": tr[0], "Train RMSE": tr[1], "Train R2": tr[2],
                     "Test MSE": te[0], "Test RMSE": te[1], "Test R2": te[2]})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "regression_results.csv", index=False)
    print("\n" + res.round(4).to_string(index=False))
    print(f"\nBest model on test MSE: {res.loc[res['Test MSE'].idxmin(), 'Model']}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axs, ["Test MSE", "Test RMSE", "Test R2"]):
        ax.barh(res["Model"], res[col]); ax.set_title(col); ax.invert_yaxis()
        ax.grid(alpha=.3, axis="x")
        if col != "Test MSE":
            ax.set_yticklabels([])
    fig.tight_layout(); fig.savefig(OUT / "metric_comparison.png", dpi=140); plt.close(fig)

    # ii) Underfitting / overfitting: train vs test MSE over polynomial degree
    degrees = [1, 2, 3, 4, 5]
    tr_mse, te_mse = [], []
    for d in degrees:
        m = make_pipe(LinearRegression(), d).fit(Xtr, ytr)
        tr_mse.append(mean_squared_error(ytr, m.predict(Xtr)))
        te_mse.append(mean_squared_error(yte, m.predict(Xte)))
    print("\nDegree sweep (unregularised):")
    for d, a, b in zip(degrees, tr_mse, te_mse):
        print(f"  degree {d}: train MSE={a:.4f}  test MSE={b:.4f}")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(degrees, tr_mse, "o-", label="Train MSE")
    ax.plot(degrees, te_mse, "s-", label="Test MSE")
    ax.set_yscale("log"); ax.set_xticks(degrees)
    ax.set_xlabel("Polynomial degree (model complexity)"); ax.set_ylabel("MSE (log scale)")
    ax.set_title("Underfitting vs Overfitting"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "underfitting_overfitting.png", dpi=150); plt.close(fig)

    # ii) Same idea on one feature, easy to see by eye
    sample = pd.concat([Xtr["PT08.S2(NMHC)"], ytr], axis=1).dropna().sample(40, random_state=SEED)
    xs, ys = sample.iloc[:, [0]].values, sample.iloc[:, 1].values
    grid = np.linspace(xs.min(), xs.max(), 400).reshape(-1, 1)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, d, title in zip(axs, [1, 3, 15], ["Underfitting (degree 1)",
                                              "Good fit (degree 3)", "Overfitting (degree 15)"]):
        m = make_pipe(LinearRegression(), d).fit(xs, ys)
        ax.scatter(xs, ys, s=18, alpha=.7); ax.plot(grid, m.predict(grid), "r", lw=2)
        ax.set_title(title); ax.set_xlabel("PT08.S2(NMHC)"); ax.set_ylim(ys.min() - 1, ys.max() + 1)
    axs[0].set_ylabel(TARGET)
    fig.tight_layout(); fig.savefig(OUT / "fit_curves_under_over.png", dpi=150); plt.close(fig)

    # iii) CV curves used to select alpha
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, gs, name in [(axs[0], ridge_gs, "Ridge"), (axs[1], lasso_gs, "Lasso")]:
        al = gs.cv_results_["param_reg__alpha"].astype(float)
        ax.semilogx(al, -gs.cv_results_["mean_train_score"], "o-", label="Train MSE")
        ax.semilogx(al, -gs.cv_results_["mean_test_score"], "s-", label="Validation MSE (5-fold CV)")
        ax.axvline(gs.best_params_["reg__alpha"], color="g", ls="--",
                   label=f"best alpha = {gs.best_params_['reg__alpha']:.3g}")
        ax.set_xlabel("alpha"); ax.set_ylabel("MSE"); ax.set_title(f"{name}: alpha selection by CV")
        ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "cv_alpha_selection.png", dpi=150); plt.close(fig)

    # GD convergence (shows the optimisation of model a)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gd["reg"].loss_history_); ax.set_yscale("log")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Training MSE")
    ax.set_title("Gradient descent convergence"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "gd_convergence.png", dpi=150); plt.close(fig)

    print(f"\nOutputs saved in: {OUT}")


if __name__ == "__main__":
    main()

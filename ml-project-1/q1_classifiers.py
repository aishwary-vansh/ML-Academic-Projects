# Question 1: Compare classifiers on the HTRU2 (pulsar star) dataset
# KNN with cross-validation, Decision Tree, SVM (RBF kernel), Deep CNN (bonus)

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)

import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)
os.makedirs("outputs", exist_ok=True)

# ---------------- Load dataset ----------------
cols = ["mean_ip", "std_ip", "kurt_ip", "skew_ip",
        "mean_dmsnr", "std_dmsnr", "kurt_dmsnr", "skew_dmsnr", "label"]
df = pd.read_csv("data/HTRU_2.csv", header=None, names=cols)
print("Dataset shape:", df.shape)
print("Class counts:\n", df["label"].value_counts())

X = df.drop("label", axis=1).values
y = df["label"].values

# ---------------- i) 80% train / 20% test split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Train size:", X_train.shape, " Test size:", X_test.shape)

results = {}   # name -> dict of metrics
roc_data = {}  # name -> (fpr, tpr, auc)


def evaluate(name, y_pred, y_score, train_acc):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    results[name] = {"Train Acc": train_acc, "Test Acc": acc, "Precision": prec,
                     "Recall": rec, "F1-score": f1, "ROC-AUC": auc}
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_data[name] = (fpr, tpr, auc)

    # ii) Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n===== {name} =====")
    print("Confusion matrix:\n", cm)
    print(f"Accuracy={acc:.4f} Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f} ROC-AUC={auc:.4f}")

    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-pulsar", "Pulsar"], yticklabels=["Non-pulsar", "Pulsar"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"outputs/cm_{name.replace(' ', '_')}.png")
    plt.close()


# ---------------- a) KNN with cross-validation ----------------
param_grid = {"n_neighbors": list(range(1, 32, 2))}
knn_cv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy")
knn_cv.fit(X_train, y_train)
best_k = knn_cv.best_params_["n_neighbors"]
print("\nKNN 5-fold CV best k =", best_k, " CV accuracy =", round(knn_cv.best_score_, 4))

knn = knn_cv.best_estimator_
evaluate("KNN", knn.predict(X_test), knn.predict_proba(X_test)[:, 1],
         accuracy_score(y_train, knn.predict(X_train)))

# ---------------- b) Decision Tree ----------------
# choose max_depth with CV as well (unrestricted tree overfits, see complexity plot)
dt_cv = GridSearchCV(DecisionTreeClassifier(random_state=42),
                     {"max_depth": list(range(1, 21))}, cv=5, scoring="accuracy")
dt_cv.fit(X_train, y_train)
print("\nDecision Tree best max_depth =", dt_cv.best_params_["max_depth"])
dt = dt_cv.best_estimator_
evaluate("Decision Tree", dt.predict(X_test), dt.predict_proba(X_test)[:, 1],
         accuracy_score(y_train, dt.predict(X_train)))

# ---------------- c) SVM with RBF kernel ----------------
svm = SVC(kernel="rbf", C=1.0, gamma="scale")
svm.fit(X_train, y_train)
evaluate("SVM RBF", svm.predict(X_test), svm.decision_function(X_test),
         accuracy_score(y_train, svm.predict(X_train)))

# ---------------- d) Deep CNN (bonus) ----------------
# The 8 features are treated as a 1-D signal of length 8 with 1 channel.
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


Xtr_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
ytr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
Xte_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()
epochs, batch = 30, 64
train_losses = []

for ep in range(epochs):
    cnn.train()
    perm = torch.randperm(len(Xtr_t))
    total = 0
    for i in range(0, len(Xtr_t), batch):
        idx = perm[i:i + batch]
        optimizer.zero_grad()
        loss = loss_fn(cnn(Xtr_t[idx]), ytr_t[idx])
        loss.backward()
        optimizer.step()
        total += loss.item() * len(idx)
    train_losses.append(total / len(Xtr_t))
    if (ep + 1) % 5 == 0:
        print(f"CNN epoch {ep + 1}/{epochs} loss = {train_losses[-1]:.4f}")

cnn.eval()
with torch.no_grad():
    prob_test = torch.sigmoid(cnn(Xte_t)).numpy().ravel()
    prob_train = torch.sigmoid(cnn(Xtr_t)).numpy().ravel()
evaluate("CNN", (prob_test >= 0.5).astype(int), prob_test,
         accuracy_score(y_train, (prob_train >= 0.5).astype(int)))

# ---------------- iii) Compare metrics ----------------
res_df = pd.DataFrame(results).T.round(4)
print("\n===== Comparison of classifiers =====")
print(res_df)
res_df.to_csv("outputs/q1_results.csv")

res_df[["Test Acc", "Precision", "Recall", "F1-score", "ROC-AUC"]].plot(kind="bar", figsize=(9, 5))
plt.ylim(0.7, 1.0)
plt.title("Classifier comparison (test set)")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/metrics_comparison.png")
plt.close()

# ---------------- iv) Overfitting and model complexity ----------------
# KNN: small k = complex model, large k = simple model
ks = list(range(1, 32, 2))
knn_tr, knn_te = [], []
for k in ks:
    m = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    knn_tr.append(m.score(X_train, y_train))
    knn_te.append(m.score(X_test, y_test))

# Decision tree: deeper tree = more complex model
depths = list(range(1, 26))
dt_tr, dt_te = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
    dt_tr.append(m.score(X_train, y_train))
    dt_te.append(m.score(X_test, y_test))

# SVM: larger gamma = more complex decision boundary
gammas = [0.001, 0.01, 0.1, 1, 10, 100]
svm_tr, svm_te = [], []
for g in gammas:
    m = SVC(kernel="rbf", C=1.0, gamma=g).fit(X_train, y_train)
    svm_tr.append(m.score(X_train, y_train))
    svm_te.append(m.score(X_test, y_test))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(ks, knn_tr, "o-", label="Train")
ax[0].plot(ks, knn_te, "s-", label="Test")
ax[0].set_xlabel("k (number of neighbours)")
ax[0].set_title("KNN: accuracy vs k")
ax[1].plot(depths, dt_tr, "o-", label="Train")
ax[1].plot(depths, dt_te, "s-", label="Test")
ax[1].set_xlabel("max_depth")
ax[1].set_title("Decision Tree: accuracy vs depth")
ax[2].semilogx(gammas, svm_tr, "o-", label="Train")
ax[2].semilogx(gammas, svm_te, "s-", label="Test")
ax[2].set_xlabel("gamma")
ax[2].set_title("SVM RBF: accuracy vs gamma")
for a in ax:
    a.set_ylabel("Accuracy")
    a.legend()
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/overfitting_complexity.png")
plt.close()

print("\n===== Overfitting / model complexity =====")
print("KNN  k=1 : train=%.4f test=%.4f" % (knn_tr[0], knn_te[0]))
print("KNN  k=%d: train=%.4f test=%.4f" % (best_k, knn_tr[ks.index(best_k)], knn_te[ks.index(best_k)]))
print("DT   depth=2 : train=%.4f test=%.4f" % (dt_tr[1], dt_te[1]))
print("DT   depth=25: train=%.4f test=%.4f" % (dt_tr[-1], dt_te[-1]))
for g, a, b in zip(gammas, svm_tr, svm_te):
    print("SVM  gamma=%-6g: train=%.4f test=%.4f" % (g, a, b))

# CNN training loss curve
plt.figure(figsize=(5, 3.5))
plt.plot(range(1, epochs + 1), train_losses)
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("CNN training loss")
plt.tight_layout()
plt.savefig("outputs/cnn_loss.png")
plt.close()

# ---------------- v) ROC curve analysis ----------------
plt.figure(figsize=(6, 5))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/roc_curves.png")
plt.close()

best = max(roc_data, key=lambda n: roc_data[n][2])
print(f"\nBest classifier based on ROC-AUC: {best} (AUC = {roc_data[best][2]:.4f})")

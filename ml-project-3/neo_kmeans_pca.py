"""
Q3: K-Means clustering (from scratch) and PCA (from scratch) on the
NASA Nearest Earth Objects dataset (archive(3)/neo_v2.csv).

a) K-Means
  i)   K = 2, 3, 4, 5, 6
  ii)  Within-Cluster Sum of Squares (WCSS) for each K
  iii) Elbow curve

b) PCA using covariance matrix + eigenvalue decomposition
  i)   eigenvalues and eigenvectors
  ii)  explained variance ratio
  iii) cumulative explained variance plot
  iv)  reduce data to 2 and 3 principal components
  v)   scatter plot of the first two principal components

Run:  python neo_kmeans_pca.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 10)
HERE = Path(__file__).resolve().parent
CSV = HERE / "archive(3)" / "neo_v2.csv"
OUT = HERE / "outputs"
# orbiting_body is always "Earth" and sentry_object is always False -> dropped
FEATURES = ["est_diameter_min", "est_diameter_max", "relative_velocity",
            "miss_distance", "absolute_magnitude"]


# ------------------------------------------------------------ data loading
def load_data():
    df = pd.read_csv(CSV)
    X = df[FEATURES].values.astype(float)
    # diameters are heavily right-skewed, log makes them usable for distances
    X[:, 0] = np.log(X[:, 0])
    X[:, 1] = np.log(X[:, 1])
    # standardise (z-score) so every feature has the same scale
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, df["hazardous"].values


# ------------------------------------------------------- a. K-Means (scratch)
class KMeans:
    def __init__(self, k, max_iter=300, tol=1e-6, n_init=5, seed=SEED):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.rng = np.random.default_rng(seed)

    def _distances(self, X, centroids):
        # squared Euclidean distance of every point to every centroid
        return ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)

    def _run_once(self, X):
        # random initialisation: pick k distinct data points as centroids
        centroids = X[self.rng.choice(len(X), self.k, replace=False)]
        for _ in range(self.max_iter):
            labels = self._distances(X, centroids).argmin(axis=1)
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                for j in range(self.k)])
            shift = np.abs(new_centroids - centroids).max()
            centroids = new_centroids
            if shift < self.tol:
                break
        labels = self._distances(X, centroids).argmin(axis=1)
        wcss = ((X - centroids[labels]) ** 2).sum()
        return centroids, labels, wcss

    def fit(self, X):
        # several random restarts, keep the one with lowest WCSS
        best = None
        for _ in range(self.n_init):
            result = self._run_once(X)
            if best is None or result[2] < best[2]:
                best = result
        self.centroids_, self.labels_, self.wcss_ = best
        return self


def run_kmeans(X):
    print("=" * 60)
    print("a) K-Means clustering from scratch")
    print("=" * 60)
    ks = [2, 3, 4, 5, 6]
    wcss = []
    for k in ks:
        km = KMeans(k).fit(X)
        wcss.append(km.wcss_)
        sizes = np.bincount(km.labels_, minlength=k)
        print(f"K={k}:  WCSS = {km.wcss_:12.2f}   cluster sizes = {sizes.tolist()}")
        if k == 2:
            km2 = km

    # iii) elbow curve
    plt.figure(figsize=(6, 4))
    plt.plot(ks, wcss, "o-")
    plt.xlabel("Number of clusters K")
    plt.ylabel("WCSS")
    plt.title("Elbow curve (K-Means from scratch)")
    plt.xticks(ks)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "elbow_curve.png", dpi=150)
    plt.close()

    pd.DataFrame({"K": ks, "WCSS": wcss}).to_csv(OUT / "wcss.csv", index=False)
    return km2


# ---------------------------------------------------------- b. PCA (scratch)
def run_pca(X, y):
    print("\n" + "=" * 60)
    print("b) PCA using covariance matrix and eigen decomposition")
    print("=" * 60)
    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / (len(Xc) - 1)

    # i) eigenvalues / eigenvectors (eigh: covariance matrix is symmetric)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]               # sort descending
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    eigvals = np.clip(eigvals, 0, None)             # remove tiny -1e-16 round-off

    print("Covariance matrix:")
    print(pd.DataFrame(cov, index=FEATURES, columns=FEATURES).round(4))
    print("\nEigenvalues:")
    print(np.round(eigvals, 4))
    print("\nEigenvectors (columns = PC1..PC5):")
    print(pd.DataFrame(eigvecs, index=FEATURES,
                       columns=[f"PC{i + 1}" for i in range(len(eigvals))]).round(4))

    # ii) explained variance ratio
    evr = eigvals / eigvals.sum()
    cum = np.cumsum(evr)
    print("\nExplained variance ratio:  ", np.round(evr, 4))
    print("Cumulative explained var.: ", np.round(cum, 4))

    # iii) cumulative explained variance plot
    pcs = np.arange(1, len(evr) + 1)
    plt.figure(figsize=(6, 4))
    plt.bar(pcs, evr, alpha=0.6, label="Individual")
    plt.step(pcs, cum, where="mid", color="red", label="Cumulative")
    plt.plot(pcs, cum, "ro")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title("Explained variance (PCA from scratch)")
    plt.xticks(pcs)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "cumulative_explained_variance.png", dpi=150)
    plt.close()

    # iv) reduce to 2 and 3 principal components
    X_pca2 = Xc @ eigvecs[:, :2]
    X_pca3 = Xc @ eigvecs[:, :3]
    print(f"\nOriginal shape: {X.shape}")
    print(f"Reduced to 2 PCs: {X_pca2.shape}  (variance kept = {cum[1]:.4f})")
    print(f"Reduced to 3 PCs: {X_pca3.shape}  (variance kept = {cum[2]:.4f})")
    pd.DataFrame(X_pca2, columns=["PC1", "PC2"]).to_csv(OUT / "pca_2d.csv", index=False)
    pd.DataFrame(X_pca3, columns=["PC1", "PC2", "PC3"]).to_csv(OUT / "pca_3d.csv", index=False)

    # v) visualise with first two PCs
    plt.figure(figsize=(7, 5))
    for label, color in [(False, "tab:blue"), (True, "tab:red")]:
        m = y == label
        plt.scatter(X_pca2[m, 0], X_pca2[m, 1], s=2, alpha=0.3, c=color,
                    label=f"hazardous={label}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("NEO data projected on first two principal components")
    plt.legend(markerscale=6)
    plt.tight_layout()
    plt.savefig(OUT / "pca_2d_scatter.png", dpi=150)
    plt.close()

    # 3-PC view as well
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(X_pca3[:, 0], X_pca3[:, 1], X_pca3[:, 2], s=1, alpha=0.3,
               c=np.where(y, "tab:red", "tab:blue"))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("NEO data on first three principal components")
    plt.tight_layout()
    plt.savefig(OUT / "pca_3d_scatter.png", dpi=150)
    plt.close()
    return X_pca2


def plot_clusters_on_pca(X_pca2, km):
    plt.figure(figsize=(7, 5))
    plt.scatter(X_pca2[:, 0], X_pca2[:, 1], s=2, alpha=0.3, c=km.labels_, cmap="viridis")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("K-Means (K=2) clusters shown on first two PCs")
    plt.tight_layout()
    plt.savefig(OUT / "kmeans_k2_on_pca.png", dpi=150)
    plt.close()


def main():
    OUT.mkdir(exist_ok=True)
    X, y = load_data()
    print(f"Loaded {len(X)} samples, {X.shape[1]} features: {FEATURES}\n")
    km2 = run_kmeans(X)
    X_pca2 = run_pca(X, y)
    plot_clusters_on_pca(X_pca2, km2)
    print(f"\nPlots and CSVs saved to: {OUT}")


if __name__ == "__main__":
    main()

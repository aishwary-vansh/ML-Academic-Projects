# Q3 – K-Means and PCA (from scratch) on NASA Nearest Earth Objects

Dataset: [NASA – Nearest Earth Objects (Kaggle)](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects)
– 90 836 asteroid close-approach records. File used: `archive(3)/neo_v2.csv`.

## Task

| Part | Requirement | Where in `neo_kmeans_pca.py` |
|------|-------------|------------------------------|
| a | K-Means clustering (from scratch) | `KMeans` class |
| a.i | Run for K = 2, 3, 4, 5, 6 | `run_kmeans()` |
| a.ii | Compute WCSS for each K | `KMeans.wcss_`, saved to `wcss.csv` |
| a.iii | Plot the elbow curve | `elbow_curve.png` |
| b | PCA using covariance matrix + eigen decomposition | `run_pca()` |
| b.i | Eigenvalues and eigenvectors | `np.linalg.eigh` on the covariance matrix |
| b.ii | Explained variance ratio | `eigvals / eigvals.sum()` |
| b.iii | Cumulative explained variance plot | `cumulative_explained_variance.png` |
| b.iv | Reduce to 2 and 3 principal components | `pca_2d.csv`, `pca_3d.csv` |
| b.v | Visualise with the first two PCs | `pca_2d_scatter.png` |

No library K-Means or PCA is used – only NumPy.

## Data preparation

- **Features (5):** `est_diameter_min`, `est_diameter_max`, `relative_velocity`, `miss_distance`, `absolute_magnitude`.
- Dropped: `id`, `name` (identifiers), `orbiting_body` (always "Earth"), `sentry_object` (always False),
  `hazardous` (kept only as a colour label for plots, not used for fitting).
- Both diameter columns are log-transformed (heavily right-skewed).
- All features are z-score standardised so distances are not dominated by `miss_distance`.

## a) K-Means

Algorithm (Lloyd's):

1. Pick K distinct data points at random as initial centroids.
2. Assign each point to its nearest centroid (squared Euclidean distance).
3. Move each centroid to the mean of its points (an empty cluster keeps its old centroid).
4. Repeat until the largest centroid shift is below `1e-6` (max 300 iterations).
5. Run 5 random restarts (`n_init=5`) and keep the one with the lowest WCSS.

WCSS = Σ over all points of ‖x − centroid(x)‖².

| K | WCSS | Cluster sizes |
|---|------|---------------|
| 2 | 248 006.20 | 40 082 / 50 754 |
| 3 | 201 013.14 | 21 869 / 36 877 / 32 090 |
| 4 | 162 911.76 | 27 679 / 18 853 / 25 318 / 18 986 |
| 5 | 141 722.23 | 21 126 / 18 520 / 13 364 / 21 608 / 16 218 |
| 6 | 126 255.42 | 10 811 / 15 977 / 13 393 / 15 401 / 14 878 / 20 376 |

WCSS falls with every extra cluster. The largest drops are K = 2→3 and 3→4; after K = 4 the curve flattens,
so **K ≈ 4** is a reasonable elbow. The elbow is soft – the data has no sharply separated groups.

## b) PCA

Steps: centre the data → covariance matrix `Xcᵀ Xc / (n − 1)` → `eigh` → sort eigenpairs in descending order →
project onto the top eigenvectors.

| PC | Eigenvalue | Explained variance | Cumulative |
|----|-----------|--------------------|-----------|
| 1 | 3.2957 | 65.91 % | 65.91 % |
| 2 | 1.0397 | 20.79 % | 86.71 % |
| 3 | 0.6646 | 13.29 % | 100.00 % |
| 4 | 0.0000 | 0.00 % | 100.00 % |
| 5 | 0.0000 | 0.00 % | 100.00 % |

- **PC1** – object size: equal loadings on both log-diameters (−0.54) and absolute magnitude (+0.54).
- **PC2** – mostly `miss_distance` (0.73) and `relative_velocity` (0.58).
- **PC3** – contrast between `relative_velocity` (+0.76) and `miss_distance` (−0.64).

**Why two eigenvalues are 0:** in this dataset both diameter estimates are computed from `absolute_magnitude`,
so after the log transform `log(est_diameter_min)`, `log(est_diameter_max)` and `absolute_magnitude` are exact
linear functions of each other (correlation ±1.0 in the covariance matrix). The data really has only 3 independent
directions, so 3 PCs keep 100 % of the variance, and 2 PCs keep 86.7 %.

Reduced shapes: `(90836, 5)` → `(90836, 2)` and `(90836, 3)`.

In `pca_2d_scatter.png`, hazardous objects sit at the large-size (negative PC1) side, which matches
PC1 being the size axis. `kmeans_k2_on_pca.png` shows the K = 2 clusters split mainly along PC1 as well.

## Output files (`outputs/`)

| File | Content |
|------|---------|
| `wcss.csv` | WCSS for K = 2–6 |
| `elbow_curve.png` | elbow curve |
| `cumulative_explained_variance.png` | individual + cumulative explained variance |
| `pca_2d.csv`, `pca_3d.csv` | data projected onto 2 and 3 PCs |
| `pca_2d_scatter.png` | PC1 vs PC2, coloured by `hazardous` |
| `pca_3d_scatter.png` | PC1 / PC2 / PC3 3-D view |
| `kmeans_k2_on_pca.png` | K = 2 cluster labels shown on PC1 vs PC2 |

## How to run

```bash
pip install numpy pandas matplotlib
python neo_kmeans_pca.py
```

Paths are resolved relative to the script, so it can be run from any directory. Runtime: under a minute.

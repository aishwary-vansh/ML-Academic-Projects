# Machine Learning Projects

A collection of three machine learning assignments. Each project has its own folder, code, outputs and README with full details. Click a project below to open its README.

| Project | Topic | Technique | Dataset |
|---------|-------|-----------|---------|
| [ML Project 1 – Classifier Comparison](ml-project-1/README.md) | Binary classification (pulsar detection) | KNN, Decision Tree, SVM (RBF), CNN | UCI HTRU2 |
| [ML Project 2 – Regression Comparison](ml-project-2/README.md) | Regression (CO concentration prediction) | Linear (gradient descent), Polynomial, Ridge, Lasso | UCI Air Quality |
| [ML Project 3 – K-Means and PCA](ml-project-3/README.md) | Unsupervised learning (asteroid clustering) | K-Means and PCA, implemented from scratch | NASA Nearest Earth Objects |

## Project overview

### [Project 1 – Classifier Comparison](ml-project-1/README.md)

Compares four classifiers on the HTRU2 dataset of 17,898 pulsar candidates. The data is imbalanced (about 9 % pulsars), so the project relies on precision, recall, F1 and ROC-AUC rather than accuracy alone. It also studies overfitting by varying model complexity. The CNN has the highest ROC-AUC.

### [Project 2 – Regression Comparison](ml-project-2/README.md)

Predicts hourly CO concentration from gas sensor and weather readings. It compares gradient-descent linear regression, polynomial regression with degree chosen by cross-validation, and Ridge and Lasso regularisation. The project shows the underfitting-to-overfitting trend. Ridge (degree 4) gives the best test result.

### [Project 3 – K-Means and PCA](ml-project-3/README.md)

Clusters NASA near-Earth asteroid records with K-Means and reduces them with PCA. Both algorithms are written with NumPy only. The elbow curve suggests about four clusters. The first two principal components keep about 87 % of the variance.

## Requirements

Each project lists its own dependencies. Most use NumPy, pandas, scikit-learn and matplotlib. Project 1 also needs PyTorch.

## Running the projects

Open the project folder's README for setup and run instructions. Each script uses paths relative to its own folder.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.adaline_bias_absorbed_gd import AdalineBaisAbsorbed
from src.logistic_regression_bias_absorbed_gd import LogisticRegressionBiasAbsorbedGD


# -------------------------------------------------------
# Utility: Standardize features
# -------------------------------------------------------
def standardize(X):
    X_std = np.copy(X)
    X_std = (X_std - X_std.mean(axis=0)) / X_std.std(axis=0)
    return X_std


# -------------------------------------------------------
# IRIS DATASET (Setosa vs Versicolor)
# -------------------------------------------------------
print("Loading Iris dataset...")

iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris_df = pd.read_csv(iris_url, header=None)

# Select first 100 samples (Setosa + Versicolor)
y_iris = iris_df.iloc[0:100, 4].values
y_iris = np.where(y_iris == "Iris-setosa", 0, 1)

X_iris = iris_df.iloc[0:100, [2, 3]].values  # Petal length, width
X_iris_std = standardize(X_iris)


# -------------------------------------------------------
# WINE DATASET (Class 1 vs Class 2)
# -------------------------------------------------------
print("Loading Wine dataset...")

wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
wine_df = pd.read_csv(wine_url, header=None)

# Select class 1 and 2
wine_subset = wine_df[(wine_df[0] == 1) | (wine_df[0] == 2)]

y_wine = wine_subset.iloc[:, 0].values
y_wine = np.where(y_wine == 1, 0, 1)

X_wine = wine_subset.iloc[:, 1:3].values  # First two features
X_wine_std = standardize(X_wine)


# -------------------------------------------------------
# Hyperparameters (SAME for both models)
# -------------------------------------------------------
eta = 0.05
epochs = 100


# -------------------------------------------------------
# Train on IRIS
# -------------------------------------------------------
print("Training on Iris...")

ada_iris = AdalineBaisAbsorbed(eta=eta, n_iter=epochs)
ada_iris.fit(X_iris_std, y_iris)

log_iris = LogisticRegressionBiasAbsorbedGD(eta=eta, n_iter=epochs)
log_iris.fit(X_iris_std, y_iris)


# -------------------------------------------------------
# Train on WINE
# -------------------------------------------------------
print("Training on Wine...")

ada_wine = AdalineBaisAbsorbed(eta=eta, n_iter=epochs)
ada_wine.fit(X_wine_std, y_wine)

log_wine = LogisticRegressionBiasAbsorbedGD(eta=eta, n_iter=epochs)
log_wine.fit(X_wine_std, y_wine)


# -------------------------------------------------------
# Plot Comparison
# -------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# IRIS
ax[0].plot(range(1, epochs + 1), ada_iris.losses_,
           label="Adaline (Bias Absorbed)")
ax[0].plot(range(1, epochs + 1), log_iris.losses_,
           label="Logistic Regression (Bias Absorbed)")
ax[0].set_title("Iris Dataset (Binary)")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].legend()

# WINE
ax[1].plot(range(1, epochs + 1), ada_wine.losses_,
           label="Adaline (Bias Absorbed)")
ax[1].plot(range(1, epochs + 1), log_wine.losses_,
           label="Logistic Regression (Bias Absorbed)")
ax[1].set_title("Wine Dataset (Binary)")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
ax[1].legend()

plt.tight_layout()
plt.show()

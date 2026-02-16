import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from src.logistic_regression_bias_absorbed_gd import LogisticRegressionBiasAbsorbedGD


# -----------------------------------------------------
# Load Wine Dataset (Classes 1 and 2 Only)
# -----------------------------------------------------
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df = pd.read_csv(s, header=None)

# Keep only classes 1 and 2
df = df[(df[0] == 1) | (df[0] == 2)]

# Features (drop class column)
X = df.iloc[:, 1:].values

# Binary labels: class 1 -> 0, class 2 -> 1
y = np.where(df.iloc[:, 0].values == 1, 0, 1)


# -----------------------------------------------------
# Standardize Features
# -----------------------------------------------------
X_std = np.copy(X)
X_std = (X_std - X_std.mean(axis=0)) / X_std.std(axis=0)


# -----------------------------------------------------
# Hyperparameters (SAME for all methods)
# -----------------------------------------------------
eta = 0.01
n_iter = 100


# -----------------------------------------------------
# 1. Full Batch Gradient Descent
# -----------------------------------------------------
gd = LogisticRegressionBiasAbsorbedGD(eta=eta, n_iter=n_iter)

start = time.time()
gd.fit(X_std, y)
gd_time = time.time() - start


# -----------------------------------------------------
# 2. Stochastic Gradient Descent
# -----------------------------------------------------
sgd = LogisticRegressionBiasAbsorbedGD(eta=eta, n_iter=n_iter)

start = time.time()
sgd.fit_sgd(X_std, y)
sgd_time = time.time() - start


# -----------------------------------------------------
# 3. Mini-Batch SGD
# -----------------------------------------------------
mb = LogisticRegressionBiasAbsorbedGD(eta=eta, n_iter=n_iter)

start = time.time()
mb.fit_minibatch(X_std, y, batch_size=32)
mb_time = time.time() - start


# -----------------------------------------------------
# Plot Loss Convergence
# -----------------------------------------------------
plt.figure(figsize=(8,6))
plt.plot(gd.losses_, label='Full GD')
plt.plot(sgd.losses_, label='SGD')
plt.plot(mb.losses_, label='Mini-Batch (32)')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.title('Wine Dataset: GD vs SGD vs Mini-Batch')
plt.legend()
plt.show()


# -----------------------------------------------------
# Print Timing Results
# -----------------------------------------------------
print("=== Time Comparison ===")
print(f"Full GD Time:        {gd_time:.4f} seconds")
print(f"SGD Time:            {sgd_time:.4f} seconds")
print(f"Mini-Batch Time:     {mb_time:.4f} seconds")

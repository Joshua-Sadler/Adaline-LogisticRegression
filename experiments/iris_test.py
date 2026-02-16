import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Import Models
from src.adaline_bias_absorbed_gd import AdalineBaisAbsorbed
from src.adaline_gd import AdalineGD
from src.perceptron import Perceptron
from src.logistic_regression_gd import LogisticRegressionGD
from src.logistic_regression_bias_absorbed_gd import LogisticRegressionBiasAbsorbedGD


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)

    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X[y == cl, 0],
                    X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')


# Load Iris dataset
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(s, header=None, encoding='utf-8')

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

X_log = df.iloc[0:100, [2, 3]].values

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada01 = AdalineBaisAbsorbed(n_iter=15, eta=0.1).fit(X, y)
ax[0].plot(range(1, len(ada01.losses_) + 1), np.log10(ada01.losses_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Mean squared error)')
ax[0].set_title('Adaline - Learning rate 0.1')

ada02 = AdalineBaisAbsorbed(n_iter=15, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada02.losses_) + 1), ada02.losses_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Mean squared error')
ax[1].set_title('Adaline - Learning rate 0.0001')

# plt.savefig('images/02_11.png', dpi=300)
plt.show()

# standardize features
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# Train Adaline Textbook Model
ada1 = AdalineGD(eta=0.5, n_iter=20)
ada1.fit(X_std, y)

# Plot loss
plt.plot(range(1, len(ada1.losses_) + 1), ada1.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Adaline Loss on Iris')
plt.show()

# Train Adaline With Baised Absorbed Model
ada2 = AdalineBaisAbsorbed(eta=0.5, n_iter=20)
ada2.fit(X_std, y)

# Plot loss
plt.plot(range(1, len(ada2.losses_) + 1), ada2.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Adaline Bais Absorbed Loss on Iris')
plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

# plt.savefig('images/02_07.png', dpi=300)
plt.show()

# Check accuracy of ada
predictions = ada2.predict(X_std)
accuracy = (predictions == y).mean()
print("Accuracy:", accuracy)

X_log_std = np.copy(X_log)
X_log_std[:, 0] = (X_log[:, 0] - X_log[:, 0].mean()) / X_log[:, 0].std()
X_log_std[:, 1] = (X_log[:, 1] - X_log[:, 1].mean()) / X_log[:, 1].std()

lrgd = LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)
lrgd.fit(X_log_std, y)

plot_decision_regions(X_log_std, y, classifier=lrgd)

plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.title('Logistic Regression (Binary)')
plt.show()

lrgd2 = LogisticRegressionBiasAbsorbedGD(eta=0.3, n_iter=1000, random_state=1)
lrgd2.fit(X_log_std, y)

plot_decision_regions(X_log_std, y, classifier=lrgd2)

plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.title('Logistic Regression (Binary) Absorbed')
plt.show()

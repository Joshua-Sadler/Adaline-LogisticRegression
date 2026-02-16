import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.perceptron import Perceptron


# -----------------------------------------------------
# One-vs-Rest Multiclass Perceptron
# -----------------------------------------------------
class PerceptronOvR:
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        self.classifiers_ = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.classifiers_ = []

        for cls in self.classes_:
            # Create binary labels for this class
            y_binary = np.where(y == cls, 1, 0)

            clf = Perceptron(eta=self.eta, n_iter=self.n_iter)
            clf.fit(X, y_binary)

            self.classifiers_.append(clf)

        return self

    def predict(self, X):
        # Collect activation scores
        activations = np.column_stack([
            clf.net_input(X) for clf in self.classifiers_
        ])

        # Choose class with highest activation
        return self.classes_[np.argmax(activations, axis=1)]


# -----------------------------------------------------
# Plotting Function
# -----------------------------------------------------
def plot_decision_regions(X, y, classifier, resolution=0.02):

    markers = ('o', 's', '^')
    colors = ('red', 'blue', 'green')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    grid = np.array([xx1.ravel(), xx2.ravel()]).T
    lab = classifier.predict(grid)
    lab = lab.reshape(xx1.shape)

    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X[y == cl, 0],
                    X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')


# -----------------------------------------------------
# Load Full Iris Dataset (All 3 Classes)
# -----------------------------------------------------
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(s, header=None)

X = df.iloc[0:150, [2, 3]].values  # petal length & width
y_raw = df.iloc[0:150, 4].values

# Convert string labels to integers
label_map = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}

y = np.array([label_map[label] for label in y_raw])


# -----------------------------------------------------
# Standardize Features
# -----------------------------------------------------
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# Shuffle dataset
rgen = np.random.RandomState(1)
indices = rgen.permutation(len(X_std))
X_std = X_std[indices]
y = y[indices]

# -----------------------------------------------------
# Train OvR Perceptron
# -----------------------------------------------------
ovr = PerceptronOvR(eta=0.1, n_iter=50)
ovr.fit(X_std, y)

# Accuracy
predictions = ovr.predict(X_std)
accuracy = (predictions == y).mean()
print("Multiclass Accuracy:", accuracy)


# -----------------------------------------------------
# Plot
# -----------------------------------------------------
plot_decision_regions(X_std, y, classifier=ovr)

plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.title('Multiclass Iris Classification (Perceptron OvR)')
plt.legend(loc='upper left')
plt.show()

pred = ovr.predict(X_std)
print("Accuracy:", (pred == y).mean())

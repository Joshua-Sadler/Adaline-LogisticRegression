import numpy as np

class LogisticRegressionBiasAbsorbedGD:
    """Gradient descent-based logistic regression classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after training.
    losses_ : list
       Log loss function values in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    # =====================================================
    # 3️⃣Full Batch Gradient Descent
    # =====================================================
    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : Instance of LogisticRegressionGD

        """
        # Add bias column (first column of ones)
        X_aug = np.c_[np.ones(X.shape[0]), X]

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X_aug.shape[1])
        self.losses_ = []

        for i in range(self.n_iter):

            # Shuffle each epoch
            indices = rgen.permutation(len(X_aug))
            X_aug = X_aug[indices]
            y = y[indices]

            net_input = self.net_input(X_aug)
            output = self.activation(net_input)

            errors = (y - output) 
            # Gradient update (NO factor 2 in logistic regression)
            self.w_ += self.eta * X_aug.T.dot(errors) / X_aug.shape[0]
            
            loss = (-y.dot(np.log(output)) - (1 - y).dot(np.log(1 - output))) / X_aug.shape[0]
            
            self.losses_.append(loss)
        return self
    
    # =====================================================
    # 2️⃣ STOCHASTI StochasticGradient Descent (SGD)
    # =====================================================
    def fit_sgd(self, X, y):
        X_aug = np.c_[np.ones(X.shape[0]), X]

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X_aug.shape[1])
        self.losses_ = []

        for _ in range(self.n_iter):

            indices = rgen.permutation(len(X_aug))
            X_aug = X_aug[indices]
            y = y[indices]

            for xi, target in zip(X_aug, y):
                net_input = np.dot(xi, self.w_)
                output = self.activation(net_input)
                error = target - output

                self.w_ += self.eta * xi * error

            # Compute loss after epoch
            output_full = self.activation(np.dot(X_aug, self.w_))
            loss = (-y.dot(np.log(output_full)) -
                    (1 - y).dot(np.log(1 - output_full))) / len(y)

            self.losses_.append(loss)

        return self

    # =====================================================
    #  Mini-Batch SGD
    # =====================================================
    def fit_minibatch(self, X, y, batch_size=32):
        X_aug = np.c_[np.ones(X.shape[0]), X]

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X_aug.shape[1])
        self.losses_ = []

        for _ in range(self.n_iter):

            indices = rgen.permutation(len(X_aug))
            X_aug = X_aug[indices]
            y = y[indices]

            for start_idx in range(0, len(X_aug), batch_size):
                end_idx = start_idx + batch_size

                X_batch = X_aug[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]

                net_input = np.dot(X_batch, self.w_)
                output = self.activation(net_input)

                errors = y_batch - output

                self.w_ += self.eta * X_batch.T.dot(errors) / len(y_batch)

            # Compute loss after epoch
            output_full = self.activation(np.dot(X_aug, self.w_))
            loss = (-y.dot(np.log(output_full)) -
                    (1 - y).dot(np.log(1 - output_full))) / len(y)

            self.losses_.append(loss)

        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_)

    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        X_aug = np.c_[np.ones((X.shape[0], 1)), X]
        return np.where(self.activation(self.net_input(X_aug)) >= 0.5, 1, 0)

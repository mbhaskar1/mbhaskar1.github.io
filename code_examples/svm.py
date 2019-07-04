import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs
from cvxopt import matrix, solvers


def kernel(name, x, y):
    if name == 'linear':
        return np.dot(x, y.T)
    if name == 'poly':
        return (1 + np.dot(x, y.T)) ** 3


class SVM:
    def __init__(self):
        pass

    def fit(self, X, y, kernel_name='linear'):
        # Store for later use
        self.X = X
        self.y = y
        self.m = X.shape[0]
        self.kernel_name = kernel_name

        # Create optimization problem matrices
        P = np.empty((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                P[i, j] = y[i] * y[j] * kernel(kernel_name, X[i], X[j])
        q = -np.ones((self.m, 1))
        G = -np.eye(self.m)
        h = np.zeros((self.m, 1))
        A = y.reshape((1, self.m))
        b = np.zeros((1, 1))

        # Convert to CVXOPT matrix format
        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A.astype('double'))
        b = matrix(b)

        # Solve Optimization Problem
        sol = solvers.qp(P, q, G, h, A, b)
        self.lambdas = np.array(sol['x']).reshape(self.m)

        # Calculate b
        SV = np.where(self.lambdas > 1e-4)[0][0]
        self.b = y[SV] - sum(self.lambdas * y * kernel(kernel_name, X, X[SV]))

    # Plot scatterplot of data and contour plot of SVM
    def plot(self):
        x_min = min(self.X[:, 0]) - 0.5
        x_max = max(self.X[:, 0]) + 0.5
        y_min = min(self.X[:, 1]) - 0.5
        y_max = max(self.X[:, 1]) + 0.5
        step = 0.02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
        d = np.concatenate((xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)), axis=1)

        Z = self.b + np.sum(
            self.lambdas.reshape((self.m, 1)) * self.y.reshape((self.m, 1)) * kernel(self.kernel_name, self.X, d),
            axis=0)
        Z = Z.reshape(xx.shape)

        fig, ax = plt.subplots()
        sns.scatterplot(x=self.X[:, 0], y=self.X[:, 1], hue=self.y, ax=ax, palette='winter')
        ax.contour(xx, yy, Z, levels=[-1, 0, 1])
        fig.show()

    # Predict class of given data point
    def predict(self, u):
        if self.b + sum(self.lambdas * self.y * kernel(self.kernel_name, self.X, u)) >= 0:
            return 1
        else:
            return -1


svm = SVM()

# Linearly Separable Example
X = np.array([[0, 2], [0, 0], [2, 1], [3, 4], [4, 3]])
y = np.array([-1, -1, -1, 1, 1])
svm.fit(X, y)
svm.plot()

# Non-linearly Separable Example
X_2 = np.array([[1, 0], [0, 1], [2, 1], [1, 2]])
y_2 = np.array([-1, 1, 1, -1])
svm.fit(X_2, y_2, kernel_name='poly')
svm.plot()

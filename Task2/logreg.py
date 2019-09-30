from scipy.special import expit
import numpy as np
from sklearn.base import BaseEstimator


class LogReg(BaseEstimator):
    def __init__(self, lambda_1=0.0, lambda_2=1.0, gd_type='stochastic',
                 tolerance=1e-4, max_iter=1000, w0=None, alpha=1e-2):
        """
        lambda_1: L1 regularization param
        lambda_2: L2 regularization param
        gd_type: 'full' or 'stochastic'
        tolerance: for stopping gradient descent
        max_iter: maximum number of steps in gradient descent
        w0: np.array of shape (d) - init weights
        alpha: learning rate
        """
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.gd_type = gd_type
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w0 = w0
        self.alpha = alpha
        self.w = None
        self.loss_history = None
        self.spent_time = None

    def fit(self, X, y):
        """
        X: np.array of shape (l, d)
        y: np.array of shape (l)
        ---
        output: self
        """
        self.loss_history = []
        self.spent_time = []
        if self.w0 is None:
            self.w = np.random.rand(X.shape[1]) - 0.5
        else:
            self.w = self.w0
        iternum = 1
        gr_X = X
        gr_y = y
        while iternum <= self.max_iter:
            if self.gd_type == 'stochastic':
                idx = np.random.randint(X.shape[0] - 1)
                gr_X = X[idx:idx + 1]
                gr_y = y[idx:idx + 1]
            start = time()
            step = - self.alpha * self.calc_gradient(gr_X, gr_y)
            self.w += step
            self.loss_history = np.append(
                self.loss_history, self.calc_loss(X, y))
            end = time()
            self.spent_time = np.append(self.spent_time, end - start)
            if (np.sum(step**2)**0.5) < self.tolerance:
                #                 print("Ended with alpha")
                break
            iternum += 1
        return self

    def predict_proba(self, X):
        """
        X: np.array of shape (l, d)
        ---
        output: np.array of shape (l, 2) where
        first column has probabilities of -1
        second column has probabilities of +1
        """
        if self.w is None:
            raise Exception('Not trained yet')
        probs = np.empty((X.shape[0], 2))
        probs[:, 1] = expit(np.dot(X, self.w))
        probs[:, 0] = 1 - probs[:, 1]
        return probs

    def calc_gradient(self, X, y):
        """
        X: np.array of shape (l, d) (l can be equal to 1 if stochastic)
        y: np.array of shape (l)
        ---
        output: np.array of shape (d)
        """
        yy = y.reshape(y.shape[0], 1)
        w = self.w.reshape(self.w.shape[0], 1)
        grad = - yy * X
        temp = expit(-y * np.dot(X, self.w)).reshape(X.shape[0], 1)
        grad = grad * temp
        grad = np.mean(grad, axis=0)
        grad += self.lambda_2 * self.w
        return grad

    def calc_loss(self, X, y):
        """
        X: np.array of shape (l, d)
        y: np.array of shape (l)
        ---
        output: float
        """
        yy = y.reshape(y.shape[0], 1)
        w = self.w.reshape(self.w.shape[0], 1)
        add = np.logaddexp(0, -y * np.dot(X, self.w))
        loss = np.sum(add, axis=0)
        loss /= X.shape[0]
        loss += self.lambda_2 * 0.5 * np.dot(self.w, self.w)
        return loss

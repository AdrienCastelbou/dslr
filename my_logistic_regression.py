import numpy as np
import random
import sys
from utils import progressbar


def check_matrices(matrices):
    for matrix in matrices:
        if type(matrix) != np.ndarray:
            raise Exception("Bad Type")


class MyLogisticRegression:

    supported_penalties = ["l2"]
    supported_optimization_methods = ["GD", "SGD", "MBGD"]


    def __init__(self, theta, alpha=0.001, max_iter=1000, penality="l2", lambda_=1.0):
        try:
            self.alpha = alpha
            self.max_iter = max_iter
            self.theta = theta.astype(float)
            self.penality = penality
            self.lambda_ = lambda_ if penality in self.supported_penalties else 0
        except Exception as e:
            print(e, file=sys.stderr)
            return None


    def l2(self):
        try:
            check_matrices([self.theta])
            prime_theta = np.array(self.theta)
            prime_theta[0][0] = 0
            l2 = float(prime_theta.T.dot(prime_theta))
            return l2
        except Exception as e:
            print(e, file=sys.stderr)
            return None


    def gradient(self, x, y, method="GD"):
        try:
            check_matrices([x, y, self.theta])
            if not method in self.supported_optimization_methods:
                raise Exception("Error in gradient : Optimization Method not supported")
            if method == "SGD":
                pos = np.random.choice(range(0, len(x)))
                x = x[pos].reshape(1, -1)
                y = y[pos].reshape(-1, 1)
            elif method == "MBGD":
                batch_size = int(np.ceil(5 / 100 * x.shape[0]))
                pos = random.sample(range(0, len(x)), batch_size)
                x = x[pos]
                y = y[pos]
            l = len(x)
            y_hat = self.predict_(x)
            x = np.hstack((np.ones((x.shape[0], 1)), x))
            prime_theta = np.array(self.theta)
            prime_theta[0][0] = 0
            nabla_J = (x.T.dot(y_hat - y) + self.lambda_ * prime_theta )/ l
            return nabla_J
        except Exception as e:
            print(e, file=sys.stderr)
            return None

    def fit_(self, x, y, method="GD"):
        try:
            check_matrices([x, y, self.theta])
            if not method in self.supported_optimization_methods:
                raise Exception("Error in fit_ : Optimization Method not supported")
            for i in progressbar(range(self.max_iter)):
                nabla_J = self.gradient(x, y, method)
                self.theta = self.theta - self.alpha * nabla_J
            return self.theta
        except Exception as e:
            print(e, file=sys.stderr)
            return None
    
    def predict_(self, x):
        try:
            check_matrices([x])
            if not len(x) or not len(self.theta):
                raise Exception("Error in predict_ : Bad parameters")
            extended_x = np.hstack((np.ones((x.shape[0], 1)), x))
            return 1 / (1 + np.e ** - extended_x.dot(self.theta))
        except Exception as e:
            print(e, file=sys.stderr)
            return None


    def loss_elem_(self, y, y_hat):
        try:
            check_matrices([y, y_hat])
            if y.ndim == 1:
                y = y.reshape(y.shape[0], -1)
            if y_hat.ndim == 1:
                y_hat = y_hat.reshape(y_hat.shape[0], -1)
            if y.shape[1] != 1 or y_hat.shape[1] != 1:
                raise Exception("Error in loss_elem_ : Wrong dimensions for parameters")
            return y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
        except Exception as e:
            print(e, file=sys.stderr)
            return None
    

    def loss_(self, y, y_hat):
        try:
            check_matrices([y, y_hat])
            if y.shape[1] != 1 or y.shape != y_hat.shape:
                raise Exception("Error in loss_ : Wrong dimensions for parameters")
            l = len(y)
            v_ones = np.ones((l, 1))
            loss = - float(y.T.dot(np.log(y_hat + 1e-15)) + (v_ones - y).T.dot(np.log(1 - y_hat + 1e-15))) / l
            return loss + (self.lambda_ / (2 * y.shape[0])) * self.l2()
        except Exception as e:
            print(e, file=sys.stderr)
            return None
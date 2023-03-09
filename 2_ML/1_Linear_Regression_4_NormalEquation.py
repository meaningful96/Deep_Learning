"""
Created on Meaningful96

Lucent lux tua
"""

import numpy as np
import matplotlib.pyplot as plt

def linear_regression_normal_equation(X, y):
    ones = np.ones((X.shape[0], 1))
    X = np.append(ones, X, axis=1)
    W = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))
    return W


if __name__ == "__main__":
    # Run a small test example: y = 5x (approximately)
    m, n = 500, 1
    X = np.random.rand(m, n)
    y = 5 * X + np.random.randn(m, n) * 0.1
    W = linear_regression_normal_equation(X, y)
    X = X.reshape(500,1)
    X_Pad = np.column_stack([np.ones([500,1]), X])
    y_pred = W.T.dot(X_Pad.T)
    plt.plot(X.reshape(1,500), y.T, 'r.')
    plt.plot(X.reshape(1,500), y_pred, 'b.')    

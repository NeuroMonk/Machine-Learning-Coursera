from __future__ import division
import numpy as np
from sigmoid import sigmoid

def costFunction(X, y, theta):
    J = 0
    m = len(y.T)
    h = sigmoid(np.dot(X, theta.T));
    J = (-1/m) * np.sum( np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1-h))  )
  
    return J



X = np.array([[1, 8, 1, 6],
              [1, 3, 5, 7],
              [1, 4, 9, 2]])
y = np.array([[1, 0, 1]]);
theta = np.array([[-2, -1, 1, 2]]);


print costFunction(X, y, theta)
from __future__ import division
import numpy as np
from sigmoid import sigmoid

def lrCostFunction(X, y, theta, l):

    theta = theta[np.newaxis, : ]

    J = 0
    m = len(y)
    h = sigmoid(np.dot(X, theta.T));
    J = ( (-1/m) * np.sum( np.dot(y.T, np.log(h)) + np.dot((1 - y.T), np.log(1-h))  ) )
    reg = (l/(2*m) * sum(np.delete(theta.T, 0, 0)  ** 2))
    J = J + reg   

    return J

def lrGrad(X, y, theta, l):

    theta = theta[np.newaxis, : ]
    m = len(y)
    h = sigmoid(np.dot(X, theta.T));
    
    theta[0][0] = 0
    grad = (1/m) * np.sum((h - y) * X, axis = 0) + (l/m) * theta
   
    return grad.T

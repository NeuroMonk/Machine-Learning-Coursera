from __future__ import division
import numpy as np
from sigmoid import sigmoid


def costFunctionReg(X, y, theta, l):

    theta = theta[np.newaxis, : ]

    J = 0
    m = len(y)
    h = sigmoid(np.dot(X, theta.T));
    J = ( (-1/m) * np.sum( np.dot(y.T, np.log(h)) + np.dot((1 - y.T), np.log(1-h))  ) ) 
    #should not regulirize theta(0), 
    reg = (l/(2*m) * sum(np.delete(theta.T, 0, 0)  ** 2))
    J = J + reg   

    #grad = (1/m) * np.sum((h - y) * X, axis = 0)

    return J

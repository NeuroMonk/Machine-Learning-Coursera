from __future__ import division
import numpy as np
from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y); 
    J_history = np.zeros((num_iters, 1));

    for iter in range(num_iters):
        h = np.sum((np.dot(X, theta) - y) * X, axis = 0)
        theta = theta - ((alpha/m) * h)[np.newaxis].T;
        J_history[iter] = computeCost(X, y, theta)
    return theta, J_history


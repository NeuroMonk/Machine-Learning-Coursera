from __future__ import division
import numpy as np

def computeCost(X, y, theta):
    m = len(y) 
    J = 0
    h = np.dot(X, theta); 
    J = (1/(2*m)) * np.sum(np.power(h-y, 2));
    return J
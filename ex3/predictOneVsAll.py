from __future__ import division
import numpy as np
from sigmoid import sigmoid

def predictOneVsAll(X, theta):
    p = sigmoid(np.dot(X, theta.T))
    return np.argmax(p, axis= 1)[np.newaxis]
    
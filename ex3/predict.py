from __future__ import division
import numpy as np
from sigmoid import sigmoid

def predict(X, Theta1, Theta2):
    m, n = X.shape 

    a1 = X
    z2 = a1.dot(Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.concatenate((np.ones((m, 1)), a2), axis=1)
    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)
    return np.argmax(a3, axis= 1)[np.newaxis]


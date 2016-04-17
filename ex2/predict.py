from __future__ import division
import numpy as np
from sigmoid import sigmoid


def predict(X, theta):
	m = X.shape[0]
	p = np.zeros((m, 1))
	p = sigmoid(X.dot(theta.T))
	p = p >= 0.5
	return p

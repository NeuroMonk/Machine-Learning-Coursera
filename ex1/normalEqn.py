import numpy as np

def normalEqn(X, y):
	return np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)
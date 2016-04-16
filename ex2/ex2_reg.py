from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from plotData import plotData
from plotDecisionBoundary import plotDecisionBoundary
from mapFeature import mapFeature
from costFunctionReg import costFunctionReg

data = np.loadtxt("ex2data2.txt", usecols=(0,1,2), delimiter=',',dtype=None)

X = data[:, 0:2]
y = data[:, 2]
y = y[:, np.newaxis]
l = 1

m, n = X.shape
plotData(X, y)

X = mapFeature(X[:, 0][np.newaxis].T, X[:, 1][np.newaxis].T)
m, n = X.shape

theta = np.zeros((1, n))

#find out why is there so huge difference between fmin and fmin_bfgs ?
#fmin gives totaly wrong result
options = {'full_output': True, 'retall': True}
theta, cost, _, _, _, _, _, allvecs = fmin_bfgs(lambda t: costFunctionReg(X, y, t, l), theta, maxiter=400, **options)
plotDecisionBoundary(X, y, theta)
plt.show()


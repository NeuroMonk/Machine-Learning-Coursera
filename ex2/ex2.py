from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from plotData import plotData
from costFunction import costFunction
from plotDecisionBoundary import plotDecisionBoundary

data = np.loadtxt("ex2data1.txt", usecols=(0,1,2), delimiter=',',dtype=None)

X = data[:, 0:2]
y = data[:, 2]
y = y[:, np.newaxis]

m, n = X.shape
plotData(X, y)
plt.show()

X = np.concatenate((np.ones((m, 1)), X), axis =1 )
theta = np.zeros((1, n+ 1))

#costFunction(X, y, theta)

options = {'full_output': True, 'maxiter': 400}
theta , cost, _, _, _ = fmin(lambda t: costFunction(X, y, t), theta, **options)

plotDecisionBoundary(X, y, theta)
plt.show()




from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData

data = np.loadtxt("ex2data1.txt", usecols=(0,1,2), delimiter=',',dtype=None)

X = data[:, 0:2]
y = data[:, 2]
y = y[:, np.newaxis]

plotData(X, y)





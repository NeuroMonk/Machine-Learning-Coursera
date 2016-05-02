from __future__ import division
import numpy as np
import scipy.io as sio
from displayData import displayData

data = sio.loadmat("ex4data1.mat")
X = np.array(data['X'])
y = np.array(data['y'])

weights = sio.loadmat("ex4weights.mat")
Theta1 = np.array(weights["Theta1"])
Theta2 = np.array(weights["Theta2"])



#displayData(X)


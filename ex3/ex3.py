from __future__ import division
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from displayData import displayData
from oneVsAll import oneVsAll

#there was an issue with scipy 0.17.0. 
#downgrade to 0.16.0 if loadmat would crash programm
data = sio.loadmat("ex3data1.mat")
X = np.array(data['X'])
y = np.array(data['y'])

input_layer_size = 400;
num_labels = 10;

m, n = X.shape
l = 0.1;


# rand_indices =  np.random.permutation(m)
# sel = X[rand_indices[0:101], :]

#displayData(X)

print oneVsAll(X, y, 10, l).shape






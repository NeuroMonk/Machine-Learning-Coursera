from __future__ import division
import numpy as np
import scipy.io as sio
from displayData import displayData
from predict import predict

input_layer_size = 400;
hidden_layer_size = 25;
num_labels = 10

data = sio.loadmat("ex3weights.mat")
Theta1 = data["Theta1"]
Theta2 = data["Theta2"]

data = sio.loadmat("ex3data1.mat")
X = np.array(data['X'])
y = np.array(data['y'])

X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1);

pred = predict(X, Theta1, Theta2)
print np.mean((pred.T + 1  == y)) * 100

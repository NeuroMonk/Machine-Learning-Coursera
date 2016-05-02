from __future__ import division
from nnCostFunction import nnCostFunction
import numpy as np

il = 2; # input layer
hl = 2; # hidden layer
nl = 4; # number og labels
nn = np.array([range(1,18+1)]) / 10 # nn_params
X = np.cos([[1, 2], [3, 4], [5, 6]])
y = np.array([[4], [2], [3]])
l = 4

J, grad = nnCostFunction(nn, il, hl, nl, X, y, l)
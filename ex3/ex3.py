from __future__ import division
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from displayData import displayData 
from oneVsAll import oneVsAll, oneVsAll_fmin_cg
from predictOneVsAll import predictOneVsAll

#there was an issue with scipy 0.17.0. 
#downgrade to 0.16.0 if loadmat would crash programm
data = sio.loadmat("ex3data1.mat")
X = np.array(data['X'])
y = np.array(data['y'])

#remap 0 number to zero column of the array
#because of the difference in range() function
#between octave and python
y[(y == 10)] = 0

input_layer_size = 400;
num_labels = 10;

m, n = X.shape
l = 0.1;

rand_indices =  np.random.permutation(m)
sel = X[rand_indices[0:101], :]

#displayData(X)

'''
Here I splited the code into two pices
first oneVsAll made using fmin_bfgs, as it was in ex2
seconf oneVsAll function made using fmin_cg as is has to be according 
the pdf instructions of ex3. The reason why i'v done this to see the
computational difference between fmin_bfgs and fmin_cg
results:
    fmin_bfgs took 303.6s
    fmin_cg took 23.6s
    
    the fmin_cg is much more faster
'''

 
#print oneVsAll(X, y, 10, l).shape
#np.set_printoptions(threshold=np.nan)

#thetas = oneVsAll_fmin_cg(X, y, 10, l)
#sio.savemat("thetas.mat", {'thetas': thetas})

X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1);

data = sio.loadmat("thetas.mat")
thetas = np.array(data['thetas'])

pred = predictOneVsAll(X, thetas)
print np.mean((pred.T == y)) * 100


from __future__ import division
import numpy as np
from scipy.optimize import fmin_cg
from scipy.optimize import fmin_bfgs
from lrCostFunction import lrCostFunction, lrGrad, lrCostFunction_fmin_cg, lrGrad_fmin_cg

def oneVsAll(X, y, num_labels, l):

    
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1);
    m, n = X.shape
    all_theta = np.zeros((num_labels, n))

    for x in range(0, num_labels):
    	initial_theta = np.zeros((n))
    	#print fmin_cg(lambda t: lrCostFunction(X, y == x, initial_theta, l) )
    	options = {'full_output': True, 'retall': True}
    	theta, cost, _, _, _, _, _, allvecs = fmin_bfgs(lambda t: lrCostFunction(X, y == x, t, l), initial_theta, maxiter=50, **options)
        all_theta[x, :] = theta

    return all_theta


def oneVsAll_fmin_cg(X, y, num_labels, l):

    
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1);
    m, n = X.shape
    all_theta = np.zeros((num_labels, n))

    for x in range(num_labels):
    	initial_theta = np.zeros((n))
        
        theta = fmin_cg(
            f = lrCostFunction_fmin_cg,
            fprime= lrGrad_fmin_cg,
            x0 = initial_theta, 
            maxiter= 50, 
            args =(X, y==x, l)
            )

        all_theta[x, :] = theta

    return all_theta

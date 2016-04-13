from __future__ import division
import numpy as np

np.set_printoptions(suppress=True)
#Octave std and numpy std gives different results
#http://stackoverflow.com/questions/7482205/precision-why-do-matlab-and-python-numpy-give-so-different-outputs
# to get the same result as octave ddof=1 should be set
def featureNormalize(X):
    X_norm = X
    mu = np.mean(X, 0)
    sigma = np.std(X, axis = 0, ddof=1)
    
    X_norm = X_norm - mu
    X_norm = X_norm/sigma
    
    return X_norm, mu, sigma


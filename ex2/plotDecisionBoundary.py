from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plotData import plotData
from mapFeature import mapFeature
from matplotlib import cm

def plotDecisionBoundary(X, y, theta):

    if X.shape[1] <= 3:
        plotData(X[:, 1:], y)
        plot_x = np.array([[np.min(X[:, 2]), np.max(X[:,2])]])
        plot_y = (-1./theta[2])*(theta[1]*plot_x + theta[0])
        plt.plot(np.squeeze(plot_x), np.squeeze(plot_y))
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((u.shape[0], v.shape[0]))


        for i in range(u.shape[0]):
            for j in range(v.shape[0]):
                z[i, j] = mapFeature( np.array( [[u[i]]] ) ,  np.array( [[v[j]]]  ) ).dot(theta)

        cset = plt.contour(u, v, z.T, [0, 100])
        cset.collections[1].set_label('Decision Boundary')

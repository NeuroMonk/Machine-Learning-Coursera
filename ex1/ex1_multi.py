import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from computeCost import computeCost
from gradientDescent import gradientDescent
from featureNormalize import featureNormalize
from normalEqn import normalEqn

data = np.loadtxt("ex1data2.txt", usecols=(0,1,2), delimiter=',',dtype=None)
X = data[:, 0:2]
y = data[:, 2]
y = y[:, np.newaxis]
X, mu, sigma = featureNormalize(X)

m = len(y)

fig1 = plt.figure(0)
ax1 = fig1.gca(projection='3d')
ax1.scatter(X[:, 0], X[:, 1], y, c='r', marker='o')

X = np.concatenate((np.ones((m, 1)), X), axis =1 )

theta = np.zeros((3, 1)) # initialize fitting parameters

alpha = 0.2
iterations = 200

theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
fig2 = plt.figure(1)
plt.plot(range(0,np.size(J_history)), J_history, '-b');

fig3 = plt.figure(2)
ax3 = fig3.gca(projection='3d')
ax3.scatter(X[:, 1], X[:, 2], np.dot(X,theta), c='r', marker='x')


print np.dot(np.array([1, (1650-mu[0])/sigma[0], (3-mu[1])/sigma[1]])	, theta)
theta = normalEqn(X, y)
print np.dot(np.array([1, (1650-mu[0])/sigma[0], (3-mu[1])/sigma[1]])	, theta)


plt.show()

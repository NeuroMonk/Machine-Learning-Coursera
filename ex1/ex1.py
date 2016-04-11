import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from computeCost import computeCost
from gradientDescent import gradientDescent


data = np.loadtxt("ex1data1.txt", usecols=(0,1), delimiter=',',dtype=None)
X = data[:, 0]
y = data[:, 1]
X = X[:, np.newaxis]
y = y[:, np.newaxis]

m = len(y)

plt.plot(X, y, 'rx')
plt.ylabel("Profit in $10,000s")
plt.xlabel("Population of city in 10,000s'")
#plt.show()

X = np.concatenate((np.ones((m, 1)), X), axis =1 )

theta = np.zeros((2, 1)) # initialize fitting parameters

iterations = 1500
alpha = 0.01


theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

plt.plot(X[:,1], np.dot(X,theta), '-')

predict1 = np.dot([1, 3.5], theta) * 10000
predict2 = np.dot([1, 7], theta)* 10000

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)


J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i,j] = computeCost(X, y, t)

theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals.T, rstride=10, cstride=10, cmap=cm.coolwarm)
#cset = ax.contour(theta0_vals, theta1_vals, J_vals.T)

plt.show()
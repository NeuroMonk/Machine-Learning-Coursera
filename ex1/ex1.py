import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from computeCost import computeCost
from gradientDescent import gradientDescent

data = np.loadtxt("ex1data1.txt", usecols=(0,1),delimiter=',',dtype=None)
X = np.matrix(data[:, 0]).T; 
y = np.matrix(data[:, 1]).T;

m = len(y)

plt.plot(X, y, 'rx');
plt.ylabel("Profit in $10,000s")
plt.xlabel("Population of city in 10,000s'")

X = np.column_stack((np.ones((m, 1)), X)) # Add a column of ones to x
theta = np.zeros((2, 1)) # initialize fitting parameters

iterations = 1500;
alpha = 0.01;

theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

plt.plot(X[:,1], X*theta, '-')

predict1 = [1, 3.5] * theta;
predict2 = [1, 7] * theta;

theta0_vals = np.linspace(-10, 10, 100);
theta1_vals = np.linspace(-1, 4, 100);

J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.matrix([theta0_vals[i], theta1_vals[j]]).T
        J_vals[i,j] = computeCost(X, y, t);

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals)

plt.show()
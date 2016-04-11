import numpy as np
from computeCost import computeCost
from gradientDescent import gradientDescent
# https://www.coursera.org/learn/machine-learning/discussions/5wftpZnyEeWKNwpBrKr_Fw

#ans should be  11.9450
J_1 = computeCost(np.array([[1, 2], [1, 3], [1, 4], [1, 5]]), 
	              np.array([[7],[6],[5],[4]]),
	              np.array([[0.1], [0.2]])
	              )

#ans should be 7.0175
J_2 = computeCost(np.array([[1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 6]]), 
	              np.array([[7],[6],[5],[4]]),
	              np.array([[0.1], [0.2], [0.3]])
	              )

# theta =   5.2148 -0.5733
# J_hist(1) = 5.9794
# J_hist(1000) = 0.85426
theta_1, J_hist_1= gradientDescent(np.array([[1, 5], [1, 2], [1, 4], [1, 5]]), 
					  np.array([[1], [6], [4], [2]]),
	                  np.array([[0], [0]]),
	                  0.01,
	                  1000);


print ("====gradientDescent Test Case 1====\ntheta = %f, %f \nJ_hist(1): %f, \n\
J_hist(1000): %f" % (theta_1[0], theta_1[1], J_hist_1[0], J_hist_1[999]))


theta_2, J_hist_2 = gradientDescent(np.array([[1, 5], [1, 2]]),
	np.array([[1], [6]]),
	np.array([[0.5], [0.5]]),
	0.1,
	10);

print ("====gradientDescent Test Case 2====\ntheta = %f, %f \nJ_hist(1): %f, \n\
J_hist(1000): %f" % (theta_2[0], theta_2[1], J_hist_2[0], J_hist_2[0]))
import numpy as np
from computeCost import computeCost
from gradientDescent import gradientDescent

print computeCost(np.matrix('1, 2; 1, 3; 1, 4; 1, 5'), 
	              np.matrix('7;6;5;4'),
	              np.matrix('0.1; 0.2')
	              )

#ans should be  11.9450

print computeCost(np.matrix('1, 2, 3; 1, 3, 4; 1, 4, 5; 1, 5, 6'), 
	              np.matrix('7;6;5;4'),
	              np.matrix('0.1; 0.2; 0.3')
	              )

#ans should be 7.0175

print gradientDescent(np.matrix('1, 5; 1, 2; 1, 4; 1, 5'), 
					  np.matrix('1, 6, 4, 2'),
	                  np.matrix('0, 0'),
	                  0.01,
	                  1000);
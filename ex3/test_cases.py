import numpy as np
import sys
from lrCostFunction import lrCostFunction

if __name__ == "__main__":
    theta = np.array([-2, -1, 1, 2])

    X = np.array([[1, 8, 1, 6], 
	              [1, 3, 5, 7],
	              [1, 4, 9, 2]])

    y = np.array([[1], [0], [1]]) >= 0.5

    l = 3

    try:
        assert( np.around( lrCostFunction(X, y, theta, l), decimals=4).tolist() == [7.6832] )
    except:
	    sys.exit("Unit test failed")


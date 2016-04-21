import numpy as np
import sys
from sigmoid import sigmoid
from costFunctionReg import costFunctionReg

if __name__ == "__main__":

    magic3 = np.array([[8, 1, 6], 
                       [3, 5, 7],
                       [4, 9, 2]])
    
    X = np.concatenate( (np.ones((3, 1)),  magic3), axis = 1)
    y = np.array([[1, 0, 1]]).T
    theta = np.array([-2, -1, 1, 2])
    
    try:
        assert( sigmoid(12000000) == 1)
        assert( sigmoid(-25000) == 0)
        assert( sigmoid(0) == 0.5)
        assert( np.around(sigmoid(np.array([4, 5, 6])), decimals = 4).tolist() == [0.9820, 0.9933, 0.9975] )
    
        assert(np.around( costFunctionReg(X, y, theta, 0), decimals=4).tolist() == [4.6832])
        assert(np.around( costFunctionReg(X, y, theta, 3), decimals=4).tolist() == [7.6832])
        
    except:
        sys.exit("Unit test failed")
    
    sys.exit(0)
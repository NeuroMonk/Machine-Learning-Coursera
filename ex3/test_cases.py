import numpy as np
import sys
from lrCostFunction import lrCostFunction, lrGrad
from oneVsAll import oneVsAll, oneVsAll_fmin_cg
from predict import predict
import scipy.io as sio

if __name__ == "__main__":
    
    ''' lrCostFunction check '''

    theta = np.array([-2, -1, 1, 2])

    magic3 = np.array([[8, 1, 6], 
                  [3, 5, 7],
                  [4, 9, 2]])

    X = np.concatenate((np.ones((magic3.shape[0], 1)), magic3), axis=1);
    y = np.array([[1], [0], [1]]) >= 0.5

    l = 3

    try:
        assert( np.around( lrCostFunction(X, y, theta, l), decimals=4).tolist() == [7.6832] )
        assert( np.around( lrGrad(X, y, theta, l), decimals =5).tolist()  ==  [[0.31722], [-0.12768], [2.64812], [4.23787]] )

    except:
        sys.exit("Unit test failed")

    ''' oneVsAll check '''
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=6)
    X = np.r_[ magic3 , np.sin(range(1, 4))[np.newaxis], np.cos(range(1, 4))[np.newaxis]]
    y = np.array([[0, 1, 1, 0, 2]]).T
    num_labels = 3
    l = 0.1
    
    print oneVsAll(X, y, num_labels, l)
    print "\n"
    print oneVsAll_fmin_cg(X, y, num_labels, l)

    ''' nn Predict check '''

    Theta1 =  np.array([[0.00000, 0.90930, -0.75680],
                        [0.47943,   0.59847,  -0.97753],
                        [0.84147,   0.14112,  -0.95892],
                        [0.99749,  -0.35078,  -0.70554] ])
        
    Theta2 =  np.array([[0.00000,   0.93204,   0.67546,  -0.44252,  -0.99616],
                        [0.29552,   0.99749,   0.42738,  -0.68777,  -0.92581],
                        [0.56464,   0.97385,   0.14112,  -0.87158,  -0.77276],
                        [0.78333,   0.86321,  -0.15775,  -0.97753,  -0.55069]])
        
    
    X = np.reshape(np.sin(range(1,17)), (2, 8)).T
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1);
    
    
    print predict(X, Theta1, Theta2) +1 








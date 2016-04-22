import numpy as np
import sys
from lrCostFunction import lrCostFunction, lrGrad
from oneVsAll import oneVsAll, oneVsAll_fmin_cg
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





